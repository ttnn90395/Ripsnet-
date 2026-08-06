#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

# Run experiments sequentially with nohup
while IFS=' ' read -r ds model t flag; do
    jname="shape_${ds}_${model}_t${t}_${flag#--}"
    if [ -f "results/${jname}.json" ]; then
        echo "SKIP: $jname"
        continue
    fi
    echo "RUN: $jname"
    python train_shape.py "$ds" "$model" 20 "$t" "$flag" > "logs/${jname}.out" 2> "logs/${jname}.err"
    echo "DONE: $jname"
done < /tmp/missing_2d.txt
echo "ALL DONE"
