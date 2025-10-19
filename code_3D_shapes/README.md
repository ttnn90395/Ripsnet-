# RipsNet: 3D shape experiments

For the necessary requirements of the python environment see `requirements.txt`.

First, download the `ModelNet10` dataset from [here](https://modelnet.cs.princeton.edu/).
Next, in the file `./datasets/modelnet.py`, add the full path to the downloaded dataset in the definition of the variable `data_path` (line `344`).

To generate the persistence images, train RipsNet and evaluate the classifiers, run:
`python launcher_MN.py`

This will produce the file `./results/PI/modelnet10/analysis_results.csv` which contains all relevant results.

Run `python ./helper_fctns/compute_statistics.py` in order to produce the results reported in the respective tables of paper.

To train the pointnet baseline, run:
`python ./baselines/pointnet.py`

Run `python ./baselines/pointnet_evaluation.py` to obtain the results of the pointnet baseline.



