
import os
import subprocess as sp
from pathlib import Path
import sys
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))
print(sys.path)
from helper_fctns.get_names import get_dirs_results, get_dir_model, get_dir_data, get_suffix_dataset, command_data_generation


cwd = os.getcwd()
on_cluster = False
oar_sub_string = 'bash'
data_base_dir = os.path.join(cwd, 'datasets/saved_datasets')
submission_script_dir = os.path.join(cwd, 'submission_scripts')
Path(data_base_dir).mkdir(parents=True, exist_ok=True)
Path(submission_script_dir).mkdir(parents=True, exist_ok=True)

def write_bash_script(submission_file, representation='PI', num_epochs=800,
                      num_samples=1024, pctn=0.15, hd=1, n=True, PIsz=50,
                      wgt_name='linear', scaling_factor=1.0, learning_rate=5e-4,
                      normalize_pc=True,
                      sample_even=False, model_name='RN', bulk_train=False, bulk_analysis=False,
                      train_dataset_name='', modelnet_choice=10, on_cluster=on_cluster, loss_name='mse',
                      data_type='modelnet', generate_data=True,
                      resolution=100, num_landscapes=5, overwrite_RN=False, do_CV=False, num_data_gen_folds=1,
                      dropout=0, regularization=0, batch_size=32,
                      ):
    data = f'{data_type}{modelnet_choice}'
    num_classes = modelnet_choice
    kwargs_data_dir = {
                       'representation': representation,
                       'data': data,
                       'dataset_choice': 'rn_train',
                       'hom_dim': hd,
                       'PIsz': PIsz,
                       'scaling_factor': scaling_factor,
                       'wgt_name': wgt_name,
                       'normalize_vect': n,
                       'normalize_pc': normalize_pc,
                       'num_samples': num_samples,
                       'pct_noise': 0,
                       'sample_even': sample_even,
                       'ts_name': '',
                       'tde_dim': 3,
                       'skipauto': 10000,
                       'delay': 1,
                       'resolution': 100,
                       'num_landscapes': 5,
                       'on_cluster': on_cluster,
    }

    data_dir_rn_train = get_dir_data(**kwargs_data_dir)
    kwargs_data_dir.update({'dataset_choice': 'ml_train'})
    data_dir_ml_train = get_dir_data(**kwargs_data_dir)
    kwargs_data_dir.update({'dataset_choice': 'test', 'pct_noise': pctn})
    data_dir_test = get_dir_data(**kwargs_data_dir)

    kwargs_suffix = {
                     'representation': representation,
                     'data': data,
                     'hom_dim': hd,
                     'PIsz': PIsz,
                     'scaling_factor': scaling_factor,
                     'wgt_name': wgt_name,
                     'normalize_vect': n,
                     'normalize_pc': normalize_pc,
                     'num_samples': num_samples,
                     'pct_noise': 0,
                     'sample_even': sample_even,
                     'ts_name': '',
                     'tde_dim': 3,
                     'skipauto': 10000,
                     'delay': 1,
                     'resolution': 100,
                     'num_landscapes': 5,
                     'fold': 0,
                     }

    data_files_rn_train, data_files_ml_train, data_files_test = [], [], []
    for f in range(num_data_gen_folds):
        kwargs_suffix.update({'fold': f, 'pct_noise': 0})
        suffix_0 = get_suffix_dataset(**kwargs_suffix)
        kwargs_suffix.update({'pct_noise': pctn})
        suffix_n = get_suffix_dataset(**kwargs_suffix)

        data_file_rn_train = os.path.join(data_dir_rn_train, suffix_0)
        data_files_rn_train.append(data_file_rn_train)
        data_file_ml_train = os.path.join(data_dir_ml_train, suffix_0)
        data_files_ml_train.append(data_file_ml_train)
        data_file_test     = os.path.join(data_dir_test, suffix_n)
        data_files_test.append(data_file_test)

    kwargs_model = {
                     'representation': representation,
                     'data': data,
                     'hom_dim': hd,
                     'PIsz': PIsz,
                     'scaling_factor': scaling_factor,
                     'wgt_name': wgt_name,
                     'normalize_pc': normalize_pc,
                     'num_samples': num_samples,
                     'sample_even': sample_even,
                     'ts_name': '',
                     'tde_dim': 3,
                     'skipauto': 10000,
                     'delay': 1,
                     'resolution': resolution,
                     'num_landscapes': num_landscapes,
                     'learning_rate': learning_rate,
                     'num_epochs': num_epochs,
                     'loss_name': loss_name,
                     'CV': do_CV,
                     'batch_size': batch_size,
                     'regularization': regularization,
                     'dropout':dropout,
                     }

    trained_RN_model_dir, trained_RN_model_name = get_dir_model(**kwargs_model)

    kwargs_data_gen = {
                     'representation': representation,
                     'data': data,
                     'dataset_choice': 'rn_train',
                     'modelnet_choice': modelnet_choice,
                     'hom_dim': hd,
                     'PIsz': PIsz,
                     'scaling_factor': scaling_factor,
                     'pct_noise': pctn,
                     'wgt_name': wgt_name,
                     'normalize_pc': normalize_pc,
                     'num_samples': num_samples,
                     'sample_even': sample_even,
                     'ts_name': '',
                     'tde_dim': 3,
                     'skipauto': 10000,
                     'delay': 1,
                     'resolution': resolution,
                     'num_landscapes': num_landscapes,
                     'num_classes': num_classes,
                     'fold': 0
                     }
    generation_command_rn_train = ''
    generation_command_ml_train = ''
    generation_command_test = ''

    for f in range(num_data_gen_folds):
        kwargs_data_gen.update({'fold': f})
        generation_command_rn_train += f'{command_data_generation(**kwargs_data_gen)}\n'
        kwargs_data_gen.update({'dataset_choice': 'ml_train'})
        generation_command_ml_train += f'{command_data_generation(**kwargs_data_gen)}\n'
        kwargs_data_gen.update({'dataset_choice': 'test'})
        generation_command_test += f'{command_data_generation(**kwargs_data_gen)}\n'
        kwargs_data_gen.update({'dataset_choice': 'rn_train'})



    if n:
        normalize = '-n'
    else:
        normalize = ''
    if overwrite_RN:
        overwrite_RN_str = '-o'
    else:
        overwrite_RN_str = ''
    if do_CV:
        do_CV_str = '-cv'
    else:
        do_CV_str = ''
    if normalize_pc:
        norm_pc_str = '-npc'
    else:
        norm_pc_str = ''
    if bulk_train:
        bt = ' -b '
    else:
        bt = ''
    if bulk_analysis:
        ba = ' -b '
    else:
        ba = ''
    if os.path.exists(submission_file): # Making sure to start with an empty file.
        os.remove(submission_file)

    poetry = ''

    if not generate_data:
        generation_command_rn_train = ''
        generation_command_ml_train = ''
        generation_command_test     = ''

    else:
        generation_command = ''

    ## Fix whitespace in string issue:
    if not isinstance(hd, int) and (len(hd) >= 1):
        hd_str = (" ").join(str(x) for x in hd)
    else:
        hd_str = str(hd)


    analysis_command = ''
    for f in range(num_data_gen_folds):
        analysis_command += f'{poetry}python analysis_nn.py -d modelnet -r {representation} -mnc {modelnet_choice} -mn {trained_RN_model_name} -dPIp {data_files_rn_train[f]} -dtrn {data_files_ml_train[f]} -dtst {data_files_test[f]}{ba} {normalize} -hd {hd_str} -f {f} -do {dropout} -reg {regularization} -bs {batch_size}\n'


    with open(submission_file, 'w') as sscr:
        sscr.write(
            '# !/bin/bash\n'
            #f'{activate_env}\n'
            #'cd ..\n'
            f'{generation_command_rn_train}\n'
            
            f'{generation_command_ml_train}\n'
            
            f'{generation_command_test}\n'
            
            f'{poetry}python train_nn.py -d modelnet -r {representation} -mnc {modelnet_choice} -dn {data_files_rn_train[0]} {normalize} -e {num_epochs} -lr {learning_rate} -mn {model_name}{bt} {overwrite_RN_str} {do_CV_str} -do {dropout} -reg {regularization} -bs {batch_size}\n'

            f'{analysis_command}'
        )

################ Setting the Parameters:
generate_data    = True
num_data_gen_folds = 10
overwrite_RN     = False
do_CV            = True
num_epochs       = 1600
regularization   = 0.0
dropout          = 0.0
batch_size       = 200
modelnet_choices = [10]
#num_classes     = modelnet_choice
num_samples      = 1024
normalize        = True
sample_even      = False
normalize_pc     = True
homdims          = [[0, 1]]
pct_noise_list   = [0.02, 0.05, 0.15, 0.1, 0.25, 0.5]
learning_rates   = [0.005]

# PI parameters:
wgt_names       = ['quadratic']
PI_sizes        = [25]
scaling_factors = [1.0]


# # LS parameters:
resolutions = [150]
num_landscapes = [5]


# Chose the types of representations: ['PI']
representations = ['PI']
############################################

################ Creating and running the submission file:

for representation in representations:
    if representation == 'LS':
        wgt_names = [1]
        scaling_factors = [1]
        PI_sizes = [1]
    if representation == 'PI':
        resolutions = [1]
        num_landscapes = [1]
    for PIsz in PI_sizes:
        for hd in homdims:
            for pctn in pct_noise_list:
                for wgt_name in wgt_names:
                    for sf in scaling_factors:
                        for num_ls in num_landscapes:
                            for resolution in resolutions:
                                for lr in learning_rates:
                                    for modelnet_choice in modelnet_choices:
                                        num_classes = modelnet_choice
                                        if representation == 'PI':
                                            ## Fix whitespace in string:
                                            hd_str = ('').join([str(h) for h in hd])

                                            suffix = f'PI_modelnet{modelnet_choice}_homdim{hd_str}_num_pts_{num_samples}_pctn{pctn}_wgt{wgt_name}_lr{lr}_normalization_{int(normalize)}_PIsz{PIsz}_scl_fct_{sf}_samp_even{int(sample_even)}_norm_pc{int(normalize_pc)}_bs_{batch_size}_reg_{regularization}_do_{dropout}'

                                        elif representation == 'LS':
                                            suffix = f'LS_modelnet{modelnet_choice}_homdim{hd_str}_num_pts_{num_samples}_pctn{pctn}lr{lr}_normalization_{int(normalize)}_res_{resolution}_num_ls_{num_ls}_samp_even{int(sample_even)}_norm_pc{int(normalize_pc)}_bs_{batch_size}_reg_{regularization}_do_{dropout}'

                                        submission_file = os.path.join(submission_script_dir,
                                                                       f'submission_script_{suffix}_num_folds_{num_data_gen_folds}.sh')
                                        write_bash_script(num_epochs=num_epochs,
                                                          representation=representation,
                                                          num_samples=num_samples,
                                                          hd=hd, pctn=pctn, n=True,
                                                          normalize_pc = normalize_pc,
                                                          wgt_name=wgt_name,
                                                          scaling_factor=sf,
                                                          learning_rate=lr,
                                                          submission_file=submission_file,
                                                          modelnet_choice=modelnet_choice,
                                                          on_cluster=on_cluster,
                                                          PIsz=PIsz,
                                                          resolution=resolution,
                                                          num_landscapes=num_ls,
                                                          sample_even=sample_even,
                                                          generate_data=generate_data,
                                                          do_CV=do_CV,
                                                          regularization=regularization,
                                                          dropout=dropout,
                                                          batch_size=batch_size,
                                                          overwrite_RN=overwrite_RN,
                                                          num_data_gen_folds=num_data_gen_folds,
                                                          )

                                        print(f'ModelNet choice: modelnet{modelnet_choice}')
                                        command = f'chmod 777 {submission_file}'
                                        sp.run(command, shell=True, check=True)
                                        command = f"""{oar_sub_string} {submission_file}"""
                                        sp.run(command, shell=True, check=True)

