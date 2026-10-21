import json
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
import tqdm

from src.Utils.vocabulary import (
    CHECKPOINT_FREQ,
    CONTRASTIVE,
    DATALOADER,
    DEVICE,
    INDEX,
    LABELS,
    LOSS,
    MODEL,
    NB_EPOCHS,
    OPTIMIZER,
    PRED_POS,
    PREDICTED,
    REPRESENTATION,
    TEST,
    TOTAL,
    TRAIN,
    TRAIN_LABELS,
    VAL,
    WEIGHT,
)


def train(exp_config, run_path, last_model_name = "last_model.pt"):
    """
    Method to train a model on the training dataset, and evaluate the losses values on the validation dataset for a given number of epochs.
    It stores the files "[last_model_name]", "losses.json", predictions_xx.pkl (and representation_xs.pkl if relevant). 
    If the configuration file contains a CHECKPOINT_FREQ key, models are saved at a given frequency.
    Detailed return files:
    predictions:
        mode
            label_i
                Labels: array
                Predicted: array
            index: array
    
    representations:
        mode
            representations: array
            labels
                label_key: array
            index: array

    Loss_values: 
        mode
            loss_i: [...]
            total: [...]
            
    Arguments:
    ----------
    exp_config: dict of Object
    Configuration dictionary of instantiated objects
    
    run_path: string of Path
    Path to the run folder
    
    last_model_name: str
    name for the last model file
    """
    # Get objects from config
    run_path = Path(run_path)
    device = exp_config[DEVICE]
    model = exp_config[MODEL].to(device)
    optimizer = exp_config[OPTIMIZER]
    
    # Determine "validation" set (switch to test if no validation, for monitoring purposes during training, and for checkpoint predictions)
    has_val = (VAL in exp_config[DATALOADER].keys() and len(exp_config[DATALOADER][VAL]) > 0)
    has_test = (TEST in exp_config[DATALOADER].keys() and len(exp_config[DATALOADER][TEST]) > 0)
    val_set = VAL if has_val else TEST
    print(f"[train] Validation set used is {val_set}.")
    
    # Initialize predictions and loss values
    predictions = {TRAIN: {}}
    loss_values = {TRAIN: {TOTAL:[]}}
    if hasattr(model, 'latent_representation'):
        representations = {
            TRAIN: {REPRESENTATION: [], INDEX: []},
        }
    else:
        print("Model does not have a representation")
    if has_val or has_test:
        predictions[val_set] = {}
        loss_values[val_set] = {TOTAL:[]}
        if hasattr(model, 'latent_representation'):
            representations[val_set] = {REPRESENTATION: [], INDEX: []}

    # Initialize losses and predictions associated with each loss term
    for i, loss_fn in enumerate(exp_config[TRAIN_LABELS][LOSS]):
        loss_values[TRAIN][loss_fn.__class__.__name__] = []
        predictions[TRAIN][exp_config[TRAIN_LABELS][LABELS][i]] = {LABELS: [], PREDICTED: []}
        predictions[TRAIN][INDEX] = []
        if len(exp_config[DATALOADER][TEST]) > 0:
            loss_values[val_set][loss_fn.__class__.__name__] = []
            predictions[val_set][exp_config[TRAIN_LABELS][LABELS][i]] = {LABELS: [], PREDICTED: []}
            predictions[val_set][INDEX] = []

    # List of checkpoint epochs
    checkpoints = [exp_config[NB_EPOCHS] - 1] # Last epoch is always a checkpoint
    if CHECKPOINT_FREQ in exp_config.keys():
        # Checkpoints are each checkpoint_freq, with 0 and last epoch excluded
        checkpoints = [i for i in range(exp_config[CHECKPOINT_FREQ] - 1, exp_config[NB_EPOCHS], exp_config[CHECKPOINT_FREQ])]
        if exp_config[NB_EPOCHS] - 1 not in checkpoints:
            checkpoints.append(exp_config[NB_EPOCHS] - 1)
    print("[train] checkpoints epochs:", checkpoints)

    #--------------Training loop
    epoch_bar = tqdm.tqdm(range(exp_config[NB_EPOCHS]))
    for epoch in epoch_bar:
        #---------------- Training
        model.train()
        
        # Track mean losses
        mean_loss = {key:0 for key in loss_values[TRAIN].keys()}
        mean_loss[TOTAL] = 0
        #TODO consider using AMP for GPU computation optimization (see torch.cuda.amp.GradScaler)
        for i_batch, batch in enumerate(exp_config[DATALOADER][TRAIN]) :
            epoch_bar.set_description(f"Batch {i_batch + 1}/{len(exp_config[DATALOADER][TRAIN])}")

            x, labels, info = batch # get batch elements

            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            if type(x) in (tuple, list): # Several inputs are given (for example contrastive learning)
                x = tuple(t.to(device) for t in x)
                pred = model(*x)
            else: # Forward pass on one input
                x = x.to(device)
                pred = model(x)
            
            # ----- Compute losses -----
            loss = 0
            for i_loss, loss_fn in enumerate(exp_config[TRAIN_LABELS][LOSS]):
                label_key = exp_config[TRAIN_LABELS][LABELS][i_loss]
                pred_pos = exp_config[TRAIN_LABELS][PRED_POS][i_loss]
                weight = exp_config[TRAIN_LABELS][WEIGHT][i_loss]

                # Contrastive case: prediction arguments only
                if label_key == CONTRASTIVE:
                    assert isinstance(pred_pos, (tuple, list))
                    loss_inputs = ()
                    for pred_posi in pred_pos:
                        loss_inputs += (pred[pred_posi],)

                # Supervised case: prediction + target
                else:
                    target = labels[label_key].to(device) # Target is potentially a noisy label
                    loss_inputs = (pred[pred_pos], target)
                
                # Compute weighted loss
                loss_i = weight * loss_fn(*loss_inputs)
                loss = loss + loss_i

                # ------- Store predictions for checkpoints -------
                #! Predictions stored are for the model in train mode, and before the optimization step of the given epoch
                if epoch in checkpoints:
                    if isinstance(pred_pos, (tuple, list)): # Contrastive case
                        # Store tuple of predictions
                        preds = [pred[p].detach().cpu().numpy() for p in pred_pos]
                        predictions[TRAIN][label_key][PREDICTED].append(preds)
                        predictions[TRAIN][label_key][LABELS].append(np.empty(1))
                    else: # Supervised case
                        predictions[TRAIN][label_key][PREDICTED].append(pred[pred_pos].detach().cpu().numpy())
                        predictions[TRAIN][label_key][LABELS].append(labels[label_key]) #! In the predictions file, noisy labels are stored
                
                # Accumulate mean loss
                mean_loss[loss_fn.__class__.__name__] += loss_i.item() #! Loss naming via class name is fragile if same loss is used with different params or if loss is wrapped

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Uncomment this line to monitor the number of elements and number of nan elements
            # print("Parameters", sum(p.numel() for p in model.parameters() if p.requires_grad), any(torch.isnan(p).any() for p in model.parameters() if p.requires_grad))

            # ------- Store representations for checkpoints -------
            if epoch in checkpoints:
                # Store dataset indices of batch
                predictions[TRAIN][INDEX].append(info[INDEX])
                
                # Store representation
                if hasattr(model, 'latent_representation'):
                    representations[TRAIN][REPRESENTATION].append(model.latent_representation.detach().cpu().numpy())
                    representations[TRAIN][INDEX].append(info[INDEX])

            # Average loss over batches and store
            mean_loss[TOTAL] += loss.item()
    
        # Store mean loss
        for loss_key, loss_val in mean_loss.items():
            loss_values[TRAIN][loss_key].append(loss_val / len(exp_config[DATALOADER][TRAIN]))

        #------------ Validation
        if len(exp_config[DATALOADER][val_set]) > 0:
            model.eval() # Disable dropout, batch norm updates...

            # Track mean losses
            mean_loss = {key:0 for key in loss_values[val_set].keys()}
            mean_loss[TOTAL] = 0

            with torch.inference_mode(): # Inference mode (faster and safer than no_grad)
                for i_batch, batch in enumerate(exp_config[DATALOADER][val_set]) :
                    epoch_bar.set_description(f"Test {i_batch + 1}/{len(exp_config[DATALOADER][val_set])}")
                    
                    x, labels, info = batch

                    # Forward pass
                    if type(x) in (tuple, list):
                        x = tuple(t.to(device) for t in x)
                        pred = model(*x)
                    else:
                        x = x.to(device)
                        pred = model(x)
                    
                    # ----- Compute losses -----
                    loss = 0
                    for i_loss, loss_fn in enumerate(exp_config[TRAIN_LABELS][LOSS]):
                        label_key = exp_config[TRAIN_LABELS][LABELS][i_loss]
                        weight = exp_config[TRAIN_LABELS][WEIGHT][i_loss]
                        pred_pos = exp_config[TRAIN_LABELS][PRED_POS][i_loss]

                        # Contrastive case: prediction arguments only
                        if label_key == CONTRASTIVE:
                            assert isinstance(pred_pos, (tuple, list))
                            loss_inputs = tuple(pred[p] for p in pred_pos)

                        # Supervised case: prediction + target
                        else:
                            target = info[LABELS][label_key].to(device) # Target is the not noisy label
                            loss_inputs = (pred[pred_pos], target)

                        # Compute weighted loss
                        loss_i = weight * loss_fn(*loss_inputs)
                        loss = loss + loss_i

                        # ------- Store predictions for checkpoints -------
                        if epoch in checkpoints:
                            if isinstance(pred_pos, (tuple, list)): # Contrastive case
                                # Store tuple of predictions
                                preds = tuple(pred[p].cpu().numpy() for p in pred_pos)
                                predictions[val_set][label_key][PREDICTED].append(preds)
                                predictions[val_set][label_key][LABELS].append(np.empty(1))
                            else: # Supervised case
                                predictions[val_set][label_key][PREDICTED].append(pred[pred_pos].cpu().numpy())
                                predictions[val_set][label_key][LABELS].append(info[LABELS][label_key].cpu()) # Predictions store the not noisy label

                        # Accumulate mean loss
                        mean_loss[loss_fn.__class__.__name__] += loss_i.item()

                    # ------- Store representations for checkpoints -------
                    if epoch in checkpoints:
                        # Store dataset indices of batch
                        predictions[val_set][INDEX].append(info[INDEX])
                        
                        # Store last representation
                        if hasattr(model, 'latent_representation'):
                            representations[val_set][REPRESENTATION].append(model.latent_representation.cpu().numpy())
                            representations[val_set][INDEX].append(info[INDEX])
                
                    # Accumulate total loss
                    mean_loss[TOTAL] += loss.item()

                # Average loss over batches and store
                for loss_key, loss_val in mean_loss.items():
                    loss_values[val_set][loss_key].append(loss_val / len(exp_config[DATALOADER][val_set]))
                epoch_bar.set_postfix({"Loss": mean_loss})

        #------------------ Checkpoints
        if epoch in checkpoints:
            # Saving model state and optimizer state
            checkpoint_path = run_path/last_model_name if epoch == exp_config[NB_EPOCHS] - 1 else run_path/f"model_ep{epoch}.pt"
                
            torch.save(model.state_dict(), checkpoint_path)
            torch.save(optimizer.state_dict(), run_path/f"optimizer_ep{epoch}.pt")

            # Saving predictions file
            for mode, mode_dict in predictions.items():
                for label_key, label_val in mode_dict.items():
                    if label_key == INDEX:
                        # Flatten list of indices
                        predictions[mode][label_key] = np.concatenate(label_val, axis=0)
                    else:
                        for key, val in label_val.items():
                            # Check if first element is tuple/list (contrastive/multi-output case)
                            if isinstance(val[0], (list, tuple)):
                                # Convert each batch element to np.array and concatenate along axis 0
                                concatenated = []
                                for i in range(len(val[0])):  # For each output in tuple
                                    arrays_to_concat = [np.array(v[i]) for v in val]
                                    concatenated.append(np.concatenate(arrays_to_concat, axis=0))
                                predictions[mode][label_key][key] = tuple(concatenated)
                            else:
                                # Standard supervised case
                                predictions[mode][label_key][key] = np.concatenate(val, axis=0)
            with (run_path/f"predictions_ep{epoch}.pkl").open("wb") as f:
                pkl.dump(predictions, f)

            # Reinitialize predictions
            if epoch < exp_config[NB_EPOCHS]-1:
                for i, loss_fn in enumerate(exp_config[TRAIN_LABELS][LOSS]):
                    loss_values[TRAIN][loss_fn.__class__.__name__] = []
                    predictions[TRAIN][exp_config[TRAIN_LABELS][LABELS][i]] = {LABELS: [], PREDICTED: []}
                    predictions[TRAIN][INDEX] = []
                    if len(exp_config[DATALOADER][TEST]) > 0:
                        loss_values[val_set][loss_fn.__class__.__name__] = []
                        predictions[val_set][exp_config[TRAIN_LABELS][LABELS][i]] = {LABELS: [], PREDICTED: []}
                        predictions[val_set][INDEX] = []

            # Saving representations file
            if hasattr(model, 'latent_representation'):
                for mode, mode_dict in representations.items():
                    for label_key, label_val in mode_dict.items():
                        if label_key == LABELS:
                            for key, val in label_val.items():
                                representations[mode][label_key][key] = np.concatenate(val, 0)
                        else:
                            representations[mode][label_key] = np.concatenate(label_val, 0)
                with (run_path/f"representations_ep{epoch}.pkl").open("wb") as f:
                    pkl.dump(representations, f)
                # Reinitialize representations
                representations = {
                    TRAIN: {REPRESENTATION: [], INDEX: []},
                    val_set: {REPRESENTATION: [], INDEX: []},
                }

    #------------------------- Saving
    assert run_path.exists()
    with (run_path/"losses.json").open("w") as f:
        json.dump(loss_values, f, indent = 4)

    return loss_values

###############################################################################
###############################################################################

if __name__ == "__main__":
    print("""See corresponding scripts:
          - train_exp to train from an experiment number
          - train_config to train from a configuration file
          """)