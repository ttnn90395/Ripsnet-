# Training of the RN

import dill as pck
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, CosineSimilarity, MeanSquaredError
from tensorflow.keras import regularizers, layers
from IPython.display import SVG
import gudhi as gd
import gudhi.representations
from tqdm import tqdm
import pandas as pd
from time import time
import os
import argparse
import csv
from pathlib import Path
import sys
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))
print(sys.path)
from helper_fctns.get_names import get_dirs_results, get_dir_model, get_dir_data, get_suffix_dataset
from helper_fctns.create_ripsnet import create_ripsnet


# GPU support:
use_GPU = False
if use_GPU:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def custom_loss(y_true, y_pred):
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred) + 100 * tf.keras.losses.mean_absolute_error(y_true, y_pred)
    return loss


parser = argparse.ArgumentParser()

parser.add_argument(
    '-d',
    '--data',
    required=False,
    type=str,
    default='ucr_timeseries',
    help='Which data to use. Can be in ["ucr_timeseries", "modelnet"]'
)
parser.add_argument(
    '-dn',
    '--dataset_name',
    required=False,
    type=str,
    default='',
    help='Dataset name.'
)
parser.add_argument(
    '-mn',
    '--model_name',
    required=False,
    type=str,
    default='RN',
    help='Model name.'
)
parser.add_argument(
    '-tsn',
    '--time_series_name',
    required=False,
    type=str,
    default='',
    help='Name of the UCR time series.'
)
parser.add_argument(
    '-ln',
    '--loss_name',
    required=False,
    type=str,
    default='mse',
    help='Name of the loss used to train the model.'
)
parser.add_argument(
    '-n',
    '--normalize',
    required=False,
    action = 'store_true',
    help='Control whether to normalize or not.',
)
parser.add_argument(
    '-o',
    '--overwrite',
    required=False,
    action = 'store_true',
    help='Option to choose whether an existing model should be overwritten or not.',
)
parser.add_argument(
    '-cv',
    '--cross_validate',
    required=False,
    action = 'store_true',
    help='Option to choose whether do do cross validation.',
)
parser.add_argument(
    '-e',
    '--num_epochs',
    type=int,
    required=False,
    default=1000,
    help='Number of epochs used to train the rips net.',
)
parser.add_argument(
    '-lr',
    '--learning_rate',
    type=float,
    required=False,
    default=5e-3,
    help='Learning rate used to train the rips net.',
)
parser.add_argument(
    '-do',
    '--dropout',
    type=float,
    required=False,
    default=0,
    help='Dropout used to train the rips net.',
)
parser.add_argument(
    '-reg',
    '--regularization',
    type=float,
    required=False,
    default=0,
    help='l2 regularization used to train the rips net.',
)
parser.add_argument(
    '-bs',
    '--batch_size',
    type=int,
    required=False,
    default=32,
    help='Batch size used to train the rips net.',
)
parser.add_argument(
    '-b',
    '--bulk',
    required=False,
    action = 'store_true',
    help='Control whether to bulk analyze or not.',
)
parser.add_argument(
    '-r',
    '--representation',
    required=False,
    type=str,
    default='PI',
    help='Choice of the persistence diagram representation; persistence image/landscape. Must be in: ["PI", "LS"].',
)
parser.add_argument(
    '-mnc',
    '--model_net_choice',
    required=False,
    type=int,
    default=10,
    help='Choice between modelnet40 and modelnet10. Must be in [10, 40]',
)

args = parser.parse_args()

data_name = args.data
dataset_name = args.dataset_name
model_name = args.model_name
num_epochs = args.num_epochs
time_series_name = args.time_series_name
normalize = args.normalize
bulk_training = args.bulk
learning_rate = args.learning_rate
loss_name = args.loss_name
representation = args.representation
model_net_choice = args.model_net_choice
overwrite = args.overwrite
do_CV = args.cross_validate
dropout = args.dropout
batch_size = args.batch_size
regularization = args.regularization

model_net = False
assert(data_name in ['ucr_timeseries', 'modelnet'])
if data_name == 'modelnet':
    model_net = True
    data_name = data_name + str(model_net_choice)

#check if the execution is on the cluster or not:
on_cluster = False

# define directories to save the results:
cwd = os.getcwd()
dataset_dir = os.path.join(cwd, f'datasets/saved_datasets/{representation}{data_name}/{time_series_name}/rn_train')


ts_out_dir = ''
if time_series_name != '' or model_net:
    ts_out_dir = data_name


results_dir, _ = get_dirs_results(representation=representation,
                                  data=data_name,
                                  ts_name=time_series_name,
                                  on_cluster=on_cluster)
Path(results_dir).mkdir(parents=True, exist_ok=True)

history_save_dir = os.path.join(results_dir, ts_out_dir, 'history', time_series_name)
Path(history_save_dir).mkdir(parents=True, exist_ok=True)
figures_save_dir = os.path.join(results_dir, ts_out_dir, 'figures', time_series_name)
Path(figures_save_dir).mkdir(parents=True, exist_ok=True)
loss_save_dir = os.path.join(results_dir, ts_out_dir, 'loss_curves', time_series_name)
Path(loss_save_dir).mkdir(parents=True, exist_ok=True)

print('figures save dir: ', figures_save_dir)

# Collect the training results in the following file:
results_file = os.path.join(results_dir, ts_out_dir, 'training_results.csv')
# Pick the loss function:
loss_dict = {'mse': MeanSquaredError(), 'mae': MeanAbsoluteError(), 'coss': CosineSimilarity(), 'msloge': MeanSquaredLogarithmicError()}
loss = loss_dict[loss_name]


print(sys.argv)

# Load data
if bulk_training:
    dataset_names = [d for d in os.listdir(dataset_dir) if d[-4:]=='.pkl']
else:
    if dataset_name[:-4] == '.pkl':
        dataset_names = [dataset_name]
    else:
        dataset_names = [dataset_name + ".pkl"]

for ds_name in dataset_names:
    #ds_name = os.path.split(ds_name)[1]
    ds_name = ds_name[:-4]
    ds_file = os.path.join(dataset_dir, ds_name + ".pkl")

    print('\nDataset used for training the ripsnet:\n', ds_file)
    assert(os.path.isfile(ds_file))

    data = pck.load(open(ds_file, 'rb'))


    homdim = data['homdim']
    if isinstance(homdim, int):
        homdim = [homdim]
    num_homdims = len(homdim)
    homdim_str= ('').join([str(h) for h in homdim])
    pct_noise = data['pct_noise']
    data_train = data["data_train"]
    vectorization_train = data["vectorization_train"]
    data_test = data["data_test"]
    label_test = data['label_test']
    vectorization_test = data["vectorization_test"]
    vectorization_param = data["vectorization_params"]
    num_classes = data['num_classes']
    scaling_factor = data['scaling_factor']
    if model_net:
        sample_even = data['sample_even']
        normalize_pc = data['normalize_pc']


    N_sets_train = len(data_train)
    N_sets_test = len(data_test)
    vectorization_size = int(vectorization_train.shape[1])
    dim = data_train[0].shape[1]

    print("N_sets_train:       ", N_sets_train)
    print("N_sets_test:        ", N_sets_test)
    print("vectorization_size: ", vectorization_size)

    results_dict = {
                    'N train sets': N_sets_train,
                    'N test sets': N_sets_test,
                    'homdim': str(homdim).replace(',', " "),
                    'normalize': normalize,
                    'pct noise': pct_noise,
                    'num_classes': num_classes,
                    }

    if not model_net:
        skip = data["skip"]
        skipauto = data['skipauto']
        delay = data["delay"]
        tde_dim = data["tde_dim"]
        len_TS = data["TS length"]
        maxd = data['RC_max_dist']
        results_dict.update({
            'Name': os.path.split(dataset_name)[1],
            'tde dim': tde_dim,
            'skip': skip,
            'skipauto': skipauto,
            'delay': delay,
            'TS length': len_TS,
            'RC_max_dist': maxd,
        })
    else:
        num_samples = data['num_pts']
        results_dict.update({
            'normalize_pc': normalize_pc,
            'sample_even': sample_even,
        })

    if representation == 'PI':
        PI_size = vectorization_param[homdim[0]]['resolution'][0]

        results_dict.update({
            'PI size': PI_size,
            'bandwidth': vectorization_param[homdim[0]]['bandwidth'],
            'weight': vectorization_param[homdim[0]]['weight'],
        })
    else:
        resolution = vectorization_param[homdim[0]]['resolution']
        num_landscapes = vectorization_param[homdim[0]]['num_landscapes']
        results_dict.update({
            'resolution': resolution,
            'num_landscapes': num_landscapes,
        })


    data_train = tf.ragged.constant([[list(c) for c in list(data_train[i])] for i in range(len(data_train))], ragged_rank=1)
    data_test = tf.ragged.constant([[list(c) for c in list(data_test[i])] for i in range(len(data_test))], ragged_rank=1)


### Normalize the PIs:
if normalize:
    for hdidx in range(num_homdims):
        st_idx = hdidx * int(vectorization_size / num_homdims)
        end_idx = (hdidx + 1) * int(vectorization_size / num_homdims)

        MV = np.max(vectorization_train[:, st_idx:end_idx])
        vectorization_train[:, st_idx:end_idx] /= MV
        vectorization_test[:, st_idx:end_idx]  /= MV



if representation == 'PI':
    weight_str = vectorization_param[homdim[0]]["weight"]
    resolution = ''
    num_landscapes = ''

    if model_net:
        name_suffix = f'num_pts_{num_samples}_PI_sz_{PI_size}_wgt_{weight_str}_scl_fct_{scaling_factor}_samp_even{int(sample_even)}_norm_pc{int(normalize_pc)}_epochs_{num_epochs}_lr_{learning_rate}_loss_{loss_name}_homdim_{homdim_str}'
    else:
        name_suffix = f'PI_sz_{PI_size}_wgt_{weight_str}_scl_fct_{scaling_factor}_epochs_{num_epochs}_lr_{learning_rate}_loss_{loss_name}_homdim_{homdim_str}_tdedim_{tde_dim}_skipauto_{skipauto}_delay_{delay}'
else:
    name_suffix = f'res_{resolution}_numlscp_{num_landscapes}_epochs_{num_epochs}_lr_{learning_rate}_loss_{loss_name}_homdim_{homdim_str}'
    if not model_net:
        name_suffix += f'_tdedim_{tde_dim}_skipauto_{skipauto}_delay_{delay}'
    else:
        name_suffix += f'_samp_even{int(sample_even)}_norm_pc{int(normalize_pc)}'

# Setting the initial learning rate of the model:
initial_learning_rate = learning_rate

## Check if trained model already exists:
prefix = 27

##### Set parameters:
lr_schedule = initial_learning_rate

loss = MeanSquaredError()
batch_size = batch_size
loss_name = 'mse'
regularization = regularization
dropout = dropout

##### Set model save directory
kwargs_model_dir = {
                   'representation': representation,
                   'data': data_name,
                   'hom_dim': homdim,
                   'normalize_pc': normalize_pc,
                   'num_samples': num_samples,
                   'sample_even': sample_even,
                   'on_cluster': on_cluster,
                   'num_epochs': num_epochs,
                   'learning_rate': initial_learning_rate,
                   'CV': do_CV,
                   'dropout': dropout,
                   'regularization': regularization,
                   'batch_size': batch_size,
}

if representation == 'PI':
    kwargs_model_dir.update({'wgt_name': weight_str,# vectorization_param["weight"],
                             'PIsz': PI_size,
                             'scaling_factor': scaling_factor,
                             })
elif representation == 'LS':
    kwargs_model_dir.update({
        'resolution': resolution,
        'num_landscapes': num_landscapes,
    })

model_save_dir, model_name = get_dir_model(**kwargs_model_dir)
model_save_dir = os.path.join(model_save_dir, model_name)
Path(model_save_dir).mkdir(parents=True, exist_ok=True)

if os.path.isfile(os.path.join(model_save_dir, 'saved_model.pb')) and (not overwrite):
    print(f'The model\n {model_save_dir}\n already exists.\n Refusing to overwrite!\n')

else:
    print(f'Created model save directory:\n {model_save_dir}\n')

    ### Definiton and training of RN

    results_dict.update({
        'batch_size': batch_size,
        'loss_name': loss_name,
        'regularization': regularization,
        'dropout': dropout,
    })

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=350, min_delta=5e-6,
                                                      restore_best_weights=True)
    early_stopping_cv = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, min_delta=5e-6,
                                                         restore_best_weights=True)

    activation_fct = 'gelu'
    output_activation = 'hard_sigmoid'
    ragged_layers = [[200,100,50], [50, 20, 10]]
    dense_layers = [[50, 100, 200], [10, 20, 50]]
    tf.keras.backend.clear_session()

    num_models = len(ragged_layers)
    assert(num_models == len(dense_layers))

    if representation == 'PI':
        output_units = PI_size * PI_size * num_homdims
    elif representation == 'LS':
        output_units = resolution * num_landscapes * num_homdims
    else:
        raise ValueError('Invalid representation.')

    models  = []

    for idx in range(num_models):
        inputs, outputs = create_ripsnet(input_dimension=dim,
                                     ragged_layers=ragged_layers[idx],
                                     dense_layers=dense_layers[idx],
                                     output_units=output_units,
                                     activation_fct=activation_fct,
                                     output_activation=output_activation,
                                     dropout=dropout,
                                     kernel_regularization=regularization)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        adam = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
        model.compile(optimizer=adam, loss=loss, loss_weights=[1])
        # model.summary()
        models.append(model)

    # Train the model
    if not do_CV:
        best_index = 1
        best_CV_model = models[best_index]

    else:
        best_loss = np.inf
        num_folds = 3
        kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
        best_index = 0
        for mdl in tqdm(range(len(models)), file=sys.stdout, desc='cross validation'):
            mdl = models[idx]
            final_val_loss = 0
            for train, test in tqdm(kfold.split(vectorization_train), file=sys.stdout, desc='split training'):
                history = mdl.fit(tf.gather(data_train, train), vectorization_train[train],
                                  epochs=int(num_epochs * 0.75),
                                  validation_data=(tf.gather(data_train, test), vectorization_train[test]),
                                  batch_size=batch_size,
                                  callbacks=[early_stopping_cv],
                                  verbose=0)
                val_loss = history.history['val_loss'][-1]
                final_val_loss += val_loss / num_folds
                tf.keras.backend.clear_session()
            if final_val_loss < best_loss:
                best_loss = final_val_loss
                best_CV_model = mdl
                best_index = idx

    model = best_CV_model
    print('Best CV model:')
    model.summary()
    history = model.fit(data_train, vectorization_train,
                        epochs=num_epochs,
                        validation_data=(data_test, vectorization_test),
                        batch_size=batch_size,
                        callbacks=[early_stopping],
                        verbose=1)

    # Save the model
    model.save(model_save_dir)

    pck.dump(ds_file, open(os.path.join(model_save_dir, 'model_train_dataset_filename.pkl'), "wb"))
    pck.dump(results_dict, open(os.path.join(model_save_dir, 'results_dict.pkl'), "wb"))

    # Analysis of the results
    prediction = model.predict(data_test)

    print('figures save dir: ', figures_save_dir)

    # Plot the true PIs
    if representation == 'PI':
        reshape_target_size = [num_homdims * PI_size, PI_size]
    else:
        reshape_target_size = [num_landscapes, num_homdims * resolution]


    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if representation == 'PI':
            plt.imshow(np.flip(np.reshape(prediction[prefix + i], reshape_target_size), 0), vmin=0, vmax=1, cmap='jet')
            plt.colorbar()
            plt.suptitle('The corresponding predicted PI')
        else:
            for landscape_id in prediction[prefix + i].reshape(reshape_target_size):  # TODO: uber dirty.
                plt.plot(landscape_id)
            plt.suptitle('The corresponding predicted LS')
    figure_file = os.path.join(figures_save_dir, os.path.split(ds_name)[1] + name_suffix + '_predicted_vect_on_train.jpg')
    plt.savefig(figure_file)
    print(f'\nfigure file:\n{figure_file}')

    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if representation == 'PI':
            plt.imshow(np.flip(np.reshape(vectorization_test[prefix + i], reshape_target_size), 0), vmin=0, vmax=1, cmap='jet')
            plt.colorbar()
            plt.suptitle('The corresponding true PI')
        else:
            for landscape_id in vectorization_test[prefix + i].reshape(reshape_target_size):  # TODO: uber dirty.
                plt.plot(landscape_id)
            plt.suptitle('The corresponding true LS')
    plt.savefig(os.path.join(figures_save_dir, os.path.split(ds_name)[1] + name_suffix + '_true_vect_on_train.jpg'))
    print('\nImage saved to:\n', os.path.join(figures_save_dir, os.path.split(ds_name)[1] + name_suffix + '_true_vect_on_train.jpg'))

    #prefix = 38
    fig = plt.figure()
    fig.set_size_inches(6, 4.5)
    for i in range(9):
        ax = fig.add_subplot(3,3,i+1, projection='3d')
        x = data_test[prefix + i, :, 0]
        y = data_test[prefix + i, :, 1]
        z = data_test[prefix + i, :, 2]
        ax.scatter(x, y, z, s=0.2)
        ax.title.set_text(label_test[prefix + i])
        ax.axis('off')
    plt.suptitle('The corresponding point clouds')
    plt.savefig(os.path.join(figures_save_dir, 'point_clouds_on_train.jpg'),
                dpi=200)


    # Evaluation of the model and plot of the evolution of the loss
    history_dict = history.history
    pck.dump(history_dict, open(os.path.join(history_save_dir, model_name + "_train_history.pkl"), 'wb')) # + name_suffix

    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs[8:], loss[8:], 'g', label='Training loss')
    plt.plot(epochs[8:], val_loss[8:], 'b', label='Validation loss')
    plt.title('RN training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.text(0.8, 0.6,
             f'bs: {batch_size}, lr: {initial_learning_rate}\nreg: {regularization}, do: {dropout}\noutput_act: {output_activation}, act: {activation_fct}\nrlay: {ragged_layers}, denslay: {dense_layers}',
             fontsize=11,)
    plt.legend()
    plt.savefig(os.path.join(loss_save_dir, os.path.split(ds_name)[1] + '_loss_on_train.jpg'))

    plt.close('all')

    ## Saving the results:
    results_dict.update({'num epochs': num_epochs , 'learning rate': learning_rate, 'regularization': regularization,
                         'dropout': dropout,
                         'ragged_layers': str(ragged_layers[best_index]).replace(',', " "),
                         'dense_layers': str(dense_layers[best_index]).replace(',', " "),
                         'activation_fct': activation_fct,
                         'output_activation': output_activation,
                         'batch_size': batch_size,
                         'pc_normalization': normalize_pc,
                         'training loss': loss[-1],
                         'test loss': val_loss[-1],
                         'loss_name': loss_name,
                         'model_save_dir': model_save_dir,
                         'do_CV': do_CV})

    results_df = pd.DataFrame(results_dict, index=[0])
    add_header = False
    if not os.path.isfile(results_file):
        add_header = True
    results_df.to_csv(results_file, mode='a', header=add_header)

    assert(os.path.isfile(os.path.join(model_save_dir, 'saved_model.pb')))
    print('\nModel saved at:\n', model_save_dir)