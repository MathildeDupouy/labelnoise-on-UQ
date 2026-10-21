from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.Models.tools import toggle_dropout
from src.Utils.vocabulary import (
    AU,
    DATALOADER,
    DEVICE,
    EU,
    FN,
    INDEX,
    LABELS,
    MODEL,
    NAME,
    NB_INFERENCE,
    PRED_POS,
    PREDICTED,
    REPRESENTATION,
    TEST,
    TRAIN,
    TRAIN_LABELS,
    TU,
    UQ,
    VAL,
)


def one_infer(input, obj_config, exp_path, model_name="last_model.pt"):
    """
    Infer input through the model in the exp_path under the name model_name.
    obj_config is only used for the device attribute.
    """
    exp_path = Path(exp_path)
    assert exp_path.exists(), f"Path {exp_path} doesn't exist"
    # Load model (not the one from the config file)
    device = obj_config[DEVICE] if torch.cuda.is_available() else torch.device("cpu")
    with open(exp_path/model_name, 'rb') as f:
        model = torch.load(f, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    input = input.to(device)
    pred = model(input) #Shape B, C, H, W
    pred_final = [pred[i].detach().cpu() for i in range(len(pred))]


    return pred_final

def inference(obj_config):
    """
    Return predictions (and representations if the model has a latent representation) for a model and a dataloader.
    The model in the obj_config need to be the trained one.


    Arguments:
    ----------
    obj_config: dict of Object. A configuration dictionary with instantiated objects. Keys DEVICE, MODEL, TRAIN_LABELS and DATALOADER are used.

    Returns:
    ---------
    predictions:
        [mode_i] ('train', 'test', 'val')
            |_ 'dataset_index': array
            |_ [label_name_i] ('soft class', 'class', 'hard pos', ...)
                |_ 'labels': array
                |_ 'predicted': array
                |_ 'uncertainty'
                    |_ [uncertainty_name_i] : list
    
    representations:
        mode
            representations: array
            labels
                label_key: array
            index: array
    """
    # Load useful config parameters
    device = obj_config[DEVICE]
    model = obj_config[MODEL].to(device)
    has_latent = hasattr(model, 'latent_representation')
    n_inference = obj_config[NB_INFERENCE]
    
    modes = [TRAIN, VAL, TEST]
    predictions = {TRAIN: {}, VAL: {}, TEST: {}}

    # Initialize representations
    if has_latent:
        print(f"[inference] model {model.__class__.__name__} has a latent representation.")
        representations = {
            TRAIN: {REPRESENTATION: [], LABELS: {}, INDEX: []},
            VAL: {REPRESENTATION: [], LABELS: {}, INDEX: []},
            TEST: {REPRESENTATION: [], LABELS: {}, INDEX: []},
        }

    # Initialize predictions
    for mode in modes:
        for i in range(len(obj_config[TRAIN_LABELS][LABELS])):
            predictions[mode][obj_config[TRAIN_LABELS][LABELS][i]] = {
                LABELS: [], 
                PREDICTED: [], 
                UQ: defaultdict(lambda: {AU: [], EU: [], TU: []})
                }
            predictions[mode][INDEX] = []
            if has_latent:
                representations[mode][LABELS][obj_config[TRAIN_LABELS][LABELS][i]] = []

    #---- Infer
    model.eval()
    # Activate dropout at inference only if asked
    if n_inference > 1 and any(["mcdropout" in uq_method[NAME].lower() for uq_method in obj_config[UQ]]):
        print("[inference] Activating dropout at inference time")
        toggle_dropout(model, activate=True)
    else:
        print("[inference] No dropout at inference time (model in eval mode)")
    
    for mode in modes:
        if mode in obj_config[DATALOADER].keys():
            with torch.inference_mode():
                for i, batch in tqdm(enumerate(obj_config[DATALOADER][mode])) :
                    # Load batch
                    x, labels, info = batch
                    
                    # Infer n_inference times
                    pred = []
                    if type(x) in (tuple, list):
                        x = [xi.to(device) for xi in x]
                        for _ in range(n_inference):
                            pred.append(model(*x))
                    else:
                        x = x.to(device)
                        for _ in range(n_inference):
                            pred.append(model(x))

                    # Store predictions for each train labels that exist in the dataset
                    # for i, label_key in enumerate(set(obj_config[TRAIN_LABELS][LABELS]).intersection(set(labels.keys()))): # NO because not sure that the list is in the same order
                    for i, label_key in enumerate(obj_config[TRAIN_LABELS][LABELS]):
                        if label_key in set(labels.keys()):
                            pred_pos = obj_config[TRAIN_LABELS][PRED_POS][i]

                            pred_stack_i = [pred[i][pred_pos].cpu().numpy() for i in range(n_inference)] # (n_inference, label_output_size (B, K))
                            pred_stack_i = np.stack(pred_stack_i, axis=-1) # Shape [n_inference, B, K]
                            
                            # Store prediction
                            predictions[mode][label_key][PREDICTED].append(pred_stack_i)
                            predictions[mode][label_key][LABELS].append(info[LABELS][label_key].cpu()) # Clean labels are stored
                        
                    # Compute uncertainty
                    for uq_method in obj_config[UQ]:
                        for label_key in uq_method[LABELS]:
                            preds = predictions[mode][label_key][PREDICTED][-1] # Predictions of current batch (last one stored in predictions) for a given label
                            # Apply softmax if predictions were stored without
                            if not preds[0, :, 0].sum() == 1.:
                                preds = torch.nn.functional.softmax(torch.Tensor(preds), dim=1).numpy()
                            for uq_type in [AU, EU, TU]:
                                predictions[mode][label_key][UQ][uq_method[NAME].lower()][uq_type] += list(uq_method[FN][uq_type](preds))

                    predictions[mode][INDEX].append(info[INDEX])
                    
                    # Store last representation
                    #! Assume that dropout does not modify the representation
                    if has_latent:
                        representations[mode][REPRESENTATION].append(model.latent_representation.cpu().numpy())
                        for key in set(representations[mode][LABELS].keys()).intersection(set(labels.keys())):
                            representations[mode][LABELS][key].append(info[LABELS][key].cpu()) # Clean labels are stored
                        representations[mode][INDEX].append(info[INDEX])

    # Concatenating predictions
    for mode, mode_dict in predictions.items():
        for label_key, label_val in mode_dict.items():
            if label_key == INDEX:
                if len(label_val) > 0:
                    predictions[mode][label_key] = np.concatenate(label_val, 0)
            else:
                for key, val in label_val.items():
                    if key == UQ:
                        predictions[mode][label_key][UQ] = dict(predictions[mode][label_key][UQ]) # Get rid of defaultdicts, that are not treated by pickle
                    elif len(val) > 0: # If this mode is used, this should be the case
                        # Predicted shape will be [N, D1,...Dn, n_inference]
                        predictions[mode][label_key][key] = np.concatenate(val, 0)

    # Concatenating representations
    if has_latent:
        for mode, mode_dict in representations.items():
            for label_key, label_val in mode_dict.items():
                if label_key == LABELS:
                    for key, val in label_val.items():
                        if len(val) :
                            representations[mode][label_key][key] = np.concatenate(val, 0)
                else:
                    if len(label_val) > 0:  # If this mode is used, this should be the case
                        # Representation shape will be [N, Z1,...Zn]
                        representations[mode][label_key] = np.concatenate(label_val, 0)
        
        return predictions, representations
    else:
        return predictions



###############################################################################
###############################################################################

if __name__ == "__main__":
    """
    cf. Scripts/infer.py
    """