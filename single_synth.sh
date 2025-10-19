#!/bin/bash

MODE=${26}
DATA=${27}
TRAINNN=${28}

if [ $DATA = "generate" ]; then
    cd datasets/
    python synthetic.py $1 $2 $6 $7 $8 $9 ${10} ${11} ${12} ${13} 0 -1    1 ${22} ${14} ${23} ${24} ${25} $MODE
    python synthetic.py $1 $3 $6 $7 $8 $9 ${15} ${16} 0     2     0 -1    0 ${22} ${14} ${23} ${24} ${25} $MODE 
    python synthetic.py $1 $4 $6 $7 $8 $9 0     2     ${17} ${18} 0 -1    0 ${22} ${14} ${23} ${24} ${25} $MODE 
    python synthetic.py $1 $5 $6 $7 $8 $9 0     2     ${17} ${18} 1 ${19} 0 ${22} ${14} ${23} ${24} ${25} $MODE 
    cd ..
fi

if [ $TRAINNN = "train" ]; then
    python train_nn.py $1$2 ripsnet_$1$2 ${20} ${21} ${22} $MODE
fi

python analysis_nn.py ripsnet_$1$2 $1$2 $1$3 $1$4 ${20} ${22} $MODE
python analysis_nn.py ripsnet_$1$2 $1$2 $1$3 $1$5 ${20} ${22} $MODE
