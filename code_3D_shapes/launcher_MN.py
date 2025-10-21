import os
import subprocess as sp
from pathlib import Path

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
                      normalize_pc=True, sample_even=False, model_name='RN',
                      bulk_train=False, bulk_analysis=False, modelnet_choice=10,
                      loss_name='mse', generate_data=True,
                      resolution=100, num_landscapes=5, overwrite_RN=False,
                      do_CV=False, num_data_gen_folds=1,
                      dropout=0, regularization=0, batch_size=32):
    
    if os.path.exists(submission_file):
        os.remove(submission_file)
    
    # Flags
    normalize_flag = "-n" if n else ""
    overwrite_flag = "-o" if overwrite_RN else ""
    do_CV_flag = "-cv" if do_CV else ""
    norm_pc_flag = "-npc" if normalize_pc else ""
    bulk_train_flag = "-b" if bulk_train else ""
    bulk_analysis_flag = "-b" if bulk_analysis else ""
    
    hd_str = " ".join(str(h) for h in hd) if isinstance(hd, (list, tuple)) else str(hd)

    with open(submission_file, "w") as f:
        f.write("#!/bin/bash\n\n")

        # --- DATA GENERATION ---
        if generate_data:
            for fold in range(num_data_gen_folds):
                # RN train
                f.write(f"python data_generation.py "
                        f"--representation {representation} "
                        f"--dataset rn_train "
                        f"--modelnet-choice {modelnet_choice} "
                        f"--num-samples {num_samples} "
                        f"--pct-noise 0 "
                        f"--hom-dim {hd_str} "
                        f"--PIsz {PIsz} "
                        f"--scaling-factor {scaling_factor} "
                        f"--wgt-name {wgt_name} "
                        f"--normalize {normalize_flag} "
                        f"--normalize-pc {norm_pc_flag} "
                        f"--sample-even {int(sample_even)} "
                        f"--resolution {resolution} "
                        f"--num-landscapes {num_landscapes} "
                        f"--fold {fold}\n")
                # ML train
                f.write(f"python data_generation.py "
                        f"--representation {representation} "
                        f"--dataset ml_train "
                        f"--modelnet-choice {modelnet_choice} "
                        f"--num-samples {num_samples} "
                        f"--pct-noise 0 "
                        f"--hom-dim {hd_str} "
                        f"--PIsz {PIsz} "
                        f"--scaling-factor {scaling_factor} "
                        f"--wgt-name {wgt_name} "
                        f"--normalize {normalize_flag} "
                        f"--normalize-pc {norm_pc_flag} "
                        f"--sample-even {int(sample_even)} "
                        f"--resolution {resolution} "
                        f"--num-landscapes {num_landscapes} "
                        f"--fold {fold}\n")
                # Test
                f.write(f"python data_generation.py "
                        f"--representation {representation} "
                        f"--dataset test "
                        f"--modelnet-choice {modelnet_choice} "
                        f"--num-samples {num_samples} "
                        f"--pct-noise {pctn} "
                        f"--hom-dim {hd_str} "
                        f"--PIsz {PIsz} "
                        f"--scaling-factor {scaling_factor} "
                        f"--wgt-name {wgt_name} "
                        f"--normalize {normalize_flag} "
                        f"--normalize-pc {norm_pc_flag} "
                        f"--sample-even {int(sample_even)} "
                        f"--resolution {resolution} "
                        f"--num-landscapes {num_landscapes} "
                        f"--fold {fold}\n")

        # --- TRAINING ---
        f.write(f"python train_nn.py "
                f"--data-type modelnet "
                f"--representation {representation} "
                f"--modelnet-choice {modelnet_choice} "
                f"--dataset {representation} "
                f"--num-samples {num_samples} "
                f"--pct-noise {pctn} "
                f"--hom-dim {hd_str} "
                f"--PIsz {PIsz} "
                f"--scaling-factor {scaling_factor} "
                f"--wgt-name {wgt_name} "
                f"{normalize_flag} {norm_pc_flag} "
                f"--sample-even {int(sample_even)} "
                f"--num-epochs {num_epochs} "
                f"--learning-rate {learning_rate} "
                f"--batch-size {batch_size} "
                f"--dropout {dropout} "
                f"--regularization {regularization} "
                f"--model-name {model_name} "
                f"{overwrite_flag} {do_CV_flag} {bulk_train_flag}\n")

        # --- ANALYSIS ---
        for fold in range(num_data_gen_folds):
            f.write(f"python analysis_nn.py "
                    f"--data-type modelnet "
                    f"--representation {representation} "
                    f"--modelnet-choice {modelnet_choice} "
                    f"--dataset rn_train "
                    f"--model-name {model_name} "
                    f"--fold {fold} "
                    f"--dropout {dropout} "
                    f"--regularization {regularization} "
                    f"--batch-size {batch_size} "
                    f"--hom-dim {hd_str} "
                    f"{normalize_flag} {norm_pc_flag} {bulk_analysis_flag}\n")


###########################
# PARAMETERS
###########################
generate_data = True
num_data_gen_folds = 10
overwrite_RN = False
do_CV = True
num_epochs = 1600
regularization = 0.0
dropout = 0.0
batch_size = 200
modelnet_choices = [10]
num_samples = 1024
normalize = True
sample_even = False
normalize_pc = True
homdims = [[0, 1]]
pct_noise_list = [0.02, 0.05, 0.15, 0.1, 0.25, 0.5]
learning_rates = [0.005]
wgt_names = ['quadratic']
PI_sizes = [25]
scaling_factors = [1.0]
resolutions = [150]
num_landscapes = [5]
representations = ['PI']

###########################
# CREATE AND RUN SCRIPTS
###########################
for representation in representations:
    if representation == 'LS':
        wgt_names, scaling_factors, PI_sizes = [1], [1], [1]
    if representation == 'PI':
        resolutions, num_landscapes = [1], [1]

    for PIsz in PI_sizes:
        for hd in homdims:
            for pctn in pct_noise_list:
                for wgt_name in wgt_names:
                    for sf in scaling_factors:
                        for num_ls in num_landscapes:
                            for resolution in resolutions:
                                for lr in learning_rates:
                                    for modelnet_choice in modelnet_choices:
                                        hd_str = ''.join(str(h) for h in hd)
                                        suffix = (f"{representation}_modelnet{modelnet_choice}_hd{hd_str}_"
                                                  f"pts{num_samples}_pct{pctn}_wgt{wgt_name}_lr{lr}_"
                                                  f"norm{int(normalize)}_PIsz{PIsz}_sf{sf}_even{int(sample_even)}_"
                                                  f"npc{int(normalize_pc)}_bs{batch_size}_reg{regularization}_do{dropout}")
                                        
                                        submission_file = os.path.join(
                                            submission_script_dir,
                                            f"submission_script_{suffix}_num_folds_{num_data_gen_folds}.sh"
                                        )

                                        write_bash_script(
                                            submission_file=submission_file,
                                            representation=representation,
                                            num_epochs=num_epochs,
                                            num_samples=num_samples,
                                            hd=hd,
                                            pctn=pctn,
                                            n=normalize,
                                            normalize_pc=normalize_pc,
                                            PIsz=PIsz,
                                            wgt_name=wgt_name,
                                            scaling_factor=sf,
                                            learning_rate=lr,
                                            model_name='RN',
                                            modelnet_choice=modelnet_choice,
                                            overwrite_RN=overwrite_RN,
                                            do_CV=do_CV,
                                            batch_size=batch_size,
                                            dropout=dropout,
                                            regularization=regularization,
                                            num_data_gen_folds=num_data_gen_folds,
                                            generate_data=generate_data,
                                            resolution=resolution,
                                            num_landscapes=num_ls
                                        )

                                        # chmod + run
                                        sp.run(f"chmod 777 {submission_file}", shell=True, check=True)
                                        sp.run(f"{oar_sub_string} {submission_file}", shell=True, check=True)
