import os

def get_suffix_dataset(representation='PI',
                       data='modelnet10',
                       hom_dim=0,
                       PIsz=50,
                       scaling_factor=1.0,
                       wgt_name='linear',
                       normalize_vect=True,
                       normalize_pc=True,
                       num_samples=1024,
                       pct_noise=0,
                       sample_even=False,
                       ts_name='',
                       tde_dim=3,
                       skipauto=10000,
                       delay=1,
                       resolution=100,
                       num_landscapes=5,
                       fold=0,
                       ):
    ## Fix whitespace in string issue:
    if not isinstance(hom_dim, int) and (len(hom_dim) >= 1):
        hom_dim = ('').join([str(h) for h in hom_dim])

    if data[:-2] == 'modelnet':
        if representation == 'LS':
            suffix = f'{data}_homdim_{hom_dim}' + f'_num_pts_{num_samples}' +\
                     f'_pct_noise_{pct_noise}' +\
                     f'_samp_even{int(sample_even)}' + f'_norm_pc{int(normalize_pc)}' + \
                     f'_res_{resolution}_num_ls_{num_landscapes}'
        elif representation == 'PI':
            suffix = f'{data}_homdim_{hom_dim}' + f'_num_pts_{num_samples}' +\
                     f'_pct_noise_{pct_noise}' + f'_PIsz_{PIsz}' + f'_wgt_name_{wgt_name}' +\
                     f'_scl_fct_{scaling_factor}' +\
                     f'_samp_even{int(sample_even)}' + f'_norm_pc{int(normalize_pc)}'
        else:
            raise ValueError(f'Illegal representation chosen: {representation}')
    elif data == 'ucr_timeseries':
        if representation == 'LS':
            suffix = ts_name + f'_tdedim_{tde_dim}_skipauto_{skipauto}_delay_{delay}_homdim_{hom_dim}' +\
                    f'_pct_noise_{pct_noise}' + f'_res_{resolution}_num_ls_{num_landscapes}'
        elif representation == 'PI':
            suffix = f'_tdedim_{tde_dim}_skipauto_{skipauto}_delay_{delay}_homdim_{hom_dim}' +\
                     f'_pct_noise_{pct_noise}' + f'_PIsz_{PIsz}' + f'_wgt_name_{wgt_name}' +\
                     f'_scl_fct_{scaling_factor}'
        else:
            raise ValueError(f'Illegal representation chosen: {representation}')
    else:
        raise ValueError(f'Illegal dataset choice: {data}')
    suffix += f'_fold_{fold}'
    return suffix

def get_dir_data(representation='PI',
                       data='modelnet10',
                       dataset_choice='rn_train',
                       hom_dim=0,
                       PIsz=50,
                       scaling_factor=1.0,
                       wgt_name='linear',
                       normalize_vect=True,
                       normalize_pc=True,
                       num_samples=1024,
                       pct_noise=0,
                       sample_even=False,
                       ts_name='',
                       tde_dim=3,
                       skipauto=10000,
                       delay=1,
                       resolution=100,
                       num_landscapes=5,
                       on_cluster=True,
                 ):
    cwd = os.getcwd()
    data_base_dir = os.path.join(cwd, 'datasets/saved_datasets')

    ## Fix whitespace in string issue:
    if not isinstance(hom_dim, int) and (len(hom_dim) >= 1):
        hom_dim = ('').join([str(h) for h in hom_dim])  # str(hom_dim).replace(" ", "")

    if data[:-2] == 'modelnet':
        if representation == 'LS':
            data_dir = os.path.join(data_base_dir,
                                    representation,
                                    data,
                                    dataset_choice,
                                    f'homdim_{hom_dim}',
                                    f'num_pts_{num_samples}',
                                    f'pctnoise_{pct_noise}',
                                    f'res_{resolution}',
                                    f'num_ls_{num_landscapes}',
                                    f'samp_even{int(sample_even)}',
                                    f'norm_pc{int(normalize_pc)}'
                                    )
        elif representation == 'PI':
            data_dir = os.path.join(data_base_dir,
                                    representation,
                                    data,
                                    dataset_choice,
                                    f'homdim_{hom_dim}',
                                    f'num_pts_{num_samples}',
                                    f'pctnoise_{pct_noise}',
                                    f'wgtname_{wgt_name}',
                                    f'scl_fct_{scaling_factor}',
                                    f'samp_even{int(sample_even)}',
                                    f'norm_pc{int(normalize_pc)}'
                                    )
        else:
            raise ValueError(f'Illegal representation chosen: {representation}')

    elif data == 'ucr_timeseries':
        if representation == 'LS':
            data_dir = os.path.join(data_base_dir,
                                    representation,
                                    data,
                                    ts_name,
                                    dataset_choice,
                                    f'tdedim_{tde_dim}',
                                    f'skipauto_{skipauto}',
                                    f'delay_{delay}',
                                    f'homdim_{hom_dim}',
                                    f'pctnoise_{pct_noise}',
                                    f'res_{resolution}',
                                    f'num_ls_{num_landscapes}',
                                    )

        elif representation == 'PI':
            data_dir = os.path.join(data_base_dir,
                                    representation,
                                    data,
                                    ts_name,
                                    dataset_choice,
                                    f'tdedim_{tde_dim}',
                                    f'skipauto_{skipauto}',
                                    f'delay_{delay}',
                                    f'homdim_{hom_dim}',
                                    f'pctnoise_{pct_noise}',
                                    f'wgtname_{wgt_name}',
                                    f'scl_fct_{scaling_factor}',
                                    )
        else:
            raise ValueError(f'Illegal representation chosen: {representation}')

    else:
        raise ValueError(f'Illegal dataset choice: {data}')

    return data_dir

def get_dir_model(representation='PI',
                  data='modelnet10',
                  hom_dim=0,
                  PIsz=50,
                  scaling_factor=1.0,
                  wgt_name='linear',
                  normalize_pc=True,
                  num_samples=1024,
                  sample_even=False,
                  ts_name='',
                  tde_dim=3,
                  skipauto=10000,
                  delay=1,
                  resolution=100,
                  num_landscapes=5,
                  on_cluster=True,
                  model_name='RN',
                  learning_rate=1e-4,
                  num_epochs=1000,
                  loss_name='mse',
                  dropout=0,
                  regularization=0,
                  batch_size=32,
                  CV=False,
                 ):
    cwd = os.getcwd()
    model_base_dir = os.path.join(cwd, f'models/{representation}')

    model_dir = os.path.join(model_base_dir, data)

    ## Fix whitespace in string issue:
    if not isinstance(hom_dim, int) and (len(hom_dim) >= 1):
        hom_dim = ('').join([str(h) for h in hom_dim])  # str(hom_dim).replace(" ", "")

    if data[:-2] == 'modelnet':
        if representation == 'LS':
            name_suffix = f'_num_pts_{num_samples}' + \
                          f'_res_{resolution}_numlscp_{num_landscapes}_samp_even{int(sample_even)}' + \
                          f'_norm_pc{int(normalize_pc)}_epochs_{num_epochs}_lr_{learning_rate}' + \
                          f'_reg_{regularization}_loss_{loss_name}_homdim_{hom_dim}_cv_{int(CV)}'
        elif representation == 'PI':
            name_suffix = f'_num_pts_{num_samples}_PIsz_{PIsz}' + \
                          f'_wgt_{wgt_name}_scl_fct_{scaling_factor}_samp_even{int(sample_even)}' + \
                          f'_norm_pc{int(normalize_pc)}_epochs_{num_epochs}_lr_{learning_rate}' + \
                          f'_reg_{regularization}_loss_{loss_name}_homdim_{hom_dim}_cv_{int(CV)}'
        else:
            raise ValueError(f'Illegal representation chosen: {representation}')
    elif data == 'ucr_timeseries':
        model_dir = os.path.join(model_dir, ts_name)
        if representation == 'LS':
            name_suffix = f'_res_{resolution}_numlscp_{num_landscapes}' + \
                          f'_epochs_{num_epochs}_lr_{learning_rate}_reg_{regularization}' +\
                          f'_loss_{loss_name}_homdim_{hom_dim}_tdedim_{tde_dim}' + \
                          f'_skipauto_{skipauto}_delay_{delay}_cv_{int(CV)}'
        elif representation == 'PI':
            name_suffix = f'_PIsz_{PIsz}_wgt_{wgt_name}_scl_fct_{scaling_factor}' +\
                          f'_epochs_{num_epochs}_lr_{learning_rate}_reg_{regularization}' +\
                          f'_loss_{loss_name}_homdim_{hom_dim}_tdedim_{tde_dim}' + \
                          f'_skipauto_{skipauto}_delay_{delay}_cv_{int(CV)}'
        else:
            raise ValueError(f'Illegal representation chosen: {representation}')
    else:
        raise ValueError(f'Illegal dataset choice: {data}')

    model_name = model_name + name_suffix + f'_dropout_{dropout}_batch_size_{batch_size}'
    return model_dir, model_name

def get_dirs_results(representation='PI',
                       data='modelnet10',
                       ts_name='',
                       on_cluster=True,
                    ):
    cwd = os.getcwd()
    results_base_dir = os.path.join(cwd, f'results/{representation}/{data}')

    figures_output_dir = os.path.join(results_base_dir, 'analysis', 'figures', ts_name)

    return results_base_dir, figures_output_dir

def command_data_generation(
                            representation='PI',
                            data='modelnet10',
                            dataset_choice='rn_train',
                            modelnet_choice=10,
                            hom_dim=0,
                            PIsz=50,
                            scaling_factor=1.0,
                            pct_noise=0,
                            wgt_name='linear',
                            normalize=True,
                            normalize_pc=True,
                            num_samples=1024,
                            sample_even=False,
                            ts_name='',
                            tde_dim=3,
                            skipauto=10000,
                            delay=1,
                            resolution=100,
                            num_landscapes=5,
                            on_cluster=True,
                            num_classes=10,
                            fold=0,
                            ):

    norm_pc_str = ''
    sample_even_str = ''
    if data[:-2] == 'modelnet':
        python_filename = 'datasets/modelnet.py'
        assert(data[-2:] == str(num_classes))
        if normalize_pc:
            norm_pc_str = '-npc'
        if sample_even:
            sample_even_str = '-se'

    elif data == 'ucr_timeseries':
        python_filename = 'datasets/ucr_timeseries.py'
    else:
        raise ValueError(f'Illegal data chosen: {data}')
    assert(representation in ['PI', 'LS'])
    # else:
    #     raise ValueError(f'Illegal representation chosen: {representation}')
    # if on_cluster:
    #     activate_env = 'conda activate py39\n'
    #     poetry = ''
    # else:
    #     activate_env = ''
    #     poetry = 'poetry '
    poetry = ''

    if normalize:
        normalize_str = '-n'
    else:
        normalize_str = ''

    ## Fix hom_dim list issue:
    if not isinstance(hom_dim, int) and (len(hom_dim) >= 1):
        hom_dim = (" ").join(str(hd) for hd in hom_dim)

    if representation == 'PI':
        if data == 'ucr_timeseries':
            generation_command = f'{poetry}python {python_filename} -r {representation} -dc {dataset_choice} -dn {ts_name} -td {tde_dim} -sa {skipauto} -d {delay} -hd {hom_dim} -pctn {pct_noise} -PIsz {PIsz} -wn {wgt_name} -sf {scaling_factor} {normalize_str} -cl {num_classes}'
        else:
            generation_command = f'{poetry}python {python_filename} -r {representation} -dc {dataset_choice} -mnc {modelnet_choice} -ns {num_samples} -hd {hom_dim} -pctn {pct_noise} -PIsz {PIsz} -wn {wgt_name} -sf {scaling_factor} {normalize_str} {norm_pc_str} {sample_even_str}'
    elif representation == 'LS':
        if data == 'ucr_timeseries':
            generation_command = f'{poetry}python {python_filename} -r {representation} -dc {dataset_choice} -dn {ts_name} -td {tde_dim} -sa {skipauto} -d {delay} -hd {hom_dim} -pctn {pct_noise} {normalize_str} -cl {num_classes} -rs {resolution} -nls {num_landscapes} {norm_pc_str}'
        else:
            generation_command = f'{poetry}python {python_filename} -r {representation} -dc {dataset_choice} -mnc {modelnet_choice} -ns {num_samples} -hd {hom_dim} -pctn {pct_noise} {normalize_str} -rs {resolution} -nls {num_landscapes} {norm_pc_str} {sample_even_str}'

    generation_command += f' -f {fold}\n'
    return generation_command
