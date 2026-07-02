#!/bin/bash

MODE=${26}
DATA=${27}
TRAINNN=${28}

if [ $DATA = "generate" ]; then
    cd datasets/
    python ucr_time_series.py $1 $2 $6 $7 $8 $9 ${10} ${11} ${12} ${13} 0 -1    1 ${22} ${14} ${23} ${24} ${25} $MODE
    python ucr_time_series.py $1 $3 $6 $7 $8 $9 ${15} ${16} 0     2     0 -1    0 ${22} ${14} ${23} ${24} ${25} $MODE
    python ucr_time_series.py $1 $4 $6 $7 $8 $9 0     2     ${17} ${18} 0 -1    0 ${22} ${14} ${23} ${24} ${25} $MODE
    python ucr_time_series.py $1 $5 $6 $7 $8 $9 0     2     ${17} ${18} 1 ${19} 0 ${22} ${14} ${23} ${24} ${25} $MODE
    cd ..
fi

MODEL_NAME="${TFN_MODEL_NAME:-all}"
if [ $TRAINNN = "train" ]; then
    if [ "${TFN_E2E:-0}" = "1" ]; then
        # End-to-end: pass test dataset path for label loading
        python train_nn.py $1$2 $MODEL_NAME ${20} ${21} ${22} $MODE $1$4
    else
        python train_nn.py $1$2 $MODEL_NAME ${20} ${21} ${22} $MODE
    fi
fi

python analysis_nn.py $MODEL_NAME $1$2 $1$3 $1$4 ${20} ${22} $MODE
python analysis_nn.py $MODEL_NAME $1$2 $1$3 $1$5 ${20} ${22} $MODE
