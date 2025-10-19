In order to reproduce the synthetic and time series experiments of the article, run the following lines of code in a terminal:

`for i in 0 1 2 3 4 5 6 7 8 9; do python launch_expe.py synth laptop generate train try$i; done`
`for i in 0 1 2 3 4 5 6 7 8 9; do python launch_expe.py ucr   laptop generate train try$i; done`

Beware that you need first to:

1. download the UCR data sets at https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/

2. make sure that you input the right path to these data sets on your machine at line 55 of "datasets/ucr_time_series.py" and line 6 of "launch_expe.py" 

Our code requires (in addition to standard Python packages such as NumPy, TensorFlow, Scikit-Learn) the following modules: 

Gudhi (https://gudhi.inria.fr/python/latest/)

Velour (https://github.com/raphaeltinarrage/velour)


