from Model import *

CUP_PLOT_PATH= f"{PLOT_PATH}/Cup"
CUP_RESULTS_PATH= f"{RESULTS_PATH}/Cup"
CUP_MODEL_PATH= f"{MODEL_PATH}/Cup"
CUP_CSV_RESULTS_PATH = f"{CUP_RESULTS_PATH}/CSV"
DATASET_PATH_CUP = f"{DATASET_PATH}/CUP"


USE_KFOLD_CUP = True
OPTIMIZER_CUP = 1
BATCH_SIZE = 128

DATASET_PATH_CUP_TR = f"{DATASET_PATH_CUP}/ML-CUP24-TR.csv"
DATASET_PATH_CUP_TS = f"{DATASET_PATH_CUP}/ML-CUP24-TS.csv"

KFOLD_NUM_CUP = 5
VAL_SPLIT_CUP = 0.15
TEST_SPLIT_CUP = 0.15
DATA_SHUFFLE_SEED_CUP = 42

STANDARDIZE = False