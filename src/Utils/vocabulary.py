"""
Define constant strings for the application.
"""
#--------- CONSTANT VARIABLES
MAX_ITER_WHILE = 50

#--------- GENERAL
TOTAL = "total"

# Splits
TRAIN = "train"
VAL = "validation"
TEST = "test"
ALL = "all"


# Prediction
PREDICTED = "predicted"

#--------- CONFIG KEYS
# General
NAME = "name"
DESCRIPTION = "description"
INFO = "info"

# Training
DEVICE = "device"
NB_EPOCHS = "nb_epochs"
CHECKPOINT_FREQ = "checkpoint frequency"

# Model
MODEL = "model"
ARGS = "args"

# Optimization
OPTIMIZER = "optimizer"
LR = "lr"
BATCH_SIZE = "batch size"
WEIGHT_DECAY = "weight_decay"
#SCHEDULER

# Data
DATASET = "dataset"
DATA_PATH = "data path"
DATASET_TYPE = "dataset type"
TRAIN_LABELS = "train labels"
TEST_LABELS = "test labels"
#AUG
DATALOADER = "dataloader"
INDEX = "dataset index"
ID = "ID"
NOISY = "noisy"
# Labels
CONTRASTIVE = "contrastive"
UNSUPERVISED = "unsupervised"

# Loss and mertics
LOSS = "loss"
METRIC = "metric"
WEIGHT = "weight"
PRED_POS = "prediction position"

# Evaluation
EVALUATION = "evaluation"
LABELS = "labels"

# Representation
REPRESENTATION = "representation"

# Projection
PROJECTION = "projection"

#--------- Uncertainty quantification
UQ = "uncertainty"
FN = "function"
TU = "total uncertainty"
EU = "epistemic uncertainty"
AU = "aleatoric uncertainty"
NB_INFERENCE = "nb inference"


#--------- GENERAL SIGNAL MANAGEMENT
# General
SIGNAL = "signal" # PTB, ESR
TEST_SIZE = "test size"
VAL_SIZE = "validation size"

# Labels
HARD_TARGET = "hard-target" # PTB, MNIST

#--------- PTB MANAGEMENT
# Paths
NORMAL_PATH = "normal path"
ABNORMAL_PATH = "abnormal path"

index2target_PTB = {
    0: "normal",
    1: "abnormal"
}

#--------- ESR MANAGEMENT
BINARY_TARGET = "binary-target"
MULTI_TARGET = "multi-target"
CSV_PATH = "csv path"

index2multitarget_ESR = {
    0: "(intra) seizure",
    1: "(intra) epileptogenic zone, no seizure",
    2: "(intra) healthy area, no seizure",
    3: "(extra) eyes closed",
    4: "(extra) eyes open"
}
multitarget_index = {value:key for key, value in index2multitarget_ESR.items()}

index2binarytarget_ESR = {
    0: "seizure",
    1: "no seizure"
}
binarytarget_index = {value:key for key, value in index2binarytarget_ESR.items()}