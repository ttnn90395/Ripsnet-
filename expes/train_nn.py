# Training of the NN

import dill as pck
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import SVG
import gudhi as gd
import gudhi.representations
from tqdm import tqdm
import os
import sys
from sklearn.model_selection import KFold
from expes.utils import DenseRagged, PermopRagged

dataset_name = sys.argv[1]
model_name   = sys.argv[2]
normalize    = int(sys.argv[3])
num_epochs   = int(sys.argv[4])
PV_type      = sys.argv[5]
mode         = sys.argv[6]


print(sys.argv)

# Load data

data = pck.load(open('datasets/' + dataset_name + ".pkl", 'rb'))
data_train = data["data_train"]
PVs_train = data["PV_train"]
data_test = data["data_test"]
PVs_test = data["PV_test"]
PV_params = data["PV_params"][0]
homdim = data["hdims"]

N_sets_train = len(data_train)
N_sets_test = len(data_test)
PV_size = PV_params['resolution'][0] if PV_type == 'PI' else PV_params['resolution'] 
dim = data_train[0].shape[1]

data_train = tf.ragged.constant([[list(c) for c in list(data_train[i])] for i in range(len(data_train))], ragged_rank=1)
data_test = tf.ragged.constant([[list(c) for c in list(data_test[i])] for i in range(len(data_test))], ragged_rank=1)

# Normalize the PIs

if normalize:
    for hidx in range(len(homdim)):
        MPV = np.max(PVs_train[hidx])
        PVs_train[hidx] /= MPV
        PVs_test[hidx]  /= MPV

output_dim = sum([PVs_train[hidx].shape[1] for hidx in range(len(homdim))])

# Definiton of the NN

if dataset_name[:5] == 'synth':
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=200, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    optim = tf.keras.optimizers.Adamax(learning_rate=5e-4)
else:
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=200, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
    optim = tf.keras.optimizers.Adam(learning_rate=5e-4)

inputs = tf.keras.Input(shape=(None, dim), dtype ="float32", ragged=True)
x = DenseRagged(units=30, use_bias=True, activation='relu')(inputs)
x = DenseRagged(units=20, use_bias=True, activation='relu')(x)
x = DenseRagged(units=10, use_bias=True, activation='relu')(x)
x = PermopRagged()(x)
x = tf.keras.layers.Dense(50,  activation='relu')(x)
x = tf.keras.layers.Dense(100, activation='relu')(x)
x = tf.keras.layers.Dense(200, activation='relu')(x)
outputs = tf.keras.layers.Dense(output_dim, activation='sigmoid')(x)
model1 = tf.keras.Model(inputs=inputs, outputs=outputs)
model1.compile(optimizer=optim, loss="mse")  #contrastive_loss

inputs = tf.keras.Input(shape=(None, dim), dtype ="float32", ragged=True)
x = DenseRagged(units=30, use_bias=True, activation='gelu')(inputs)
x = DenseRagged(units=20, use_bias=True, activation='gelu')(x)
x = DenseRagged(units=10,  use_bias=True, activation='gelu')(x)
x = PermopRagged()(x)
x = tf.keras.layers.Dense(50,  activation='gelu')(x)
x = tf.keras.layers.Dense(100, activation='gelu')(x)
x = tf.keras.layers.Dense(200, activation='gelu')(x)
outputs = tf.keras.layers.Dense(output_dim, activation='gelu')(x)
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)
model2.compile(optimizer=optim, loss="mse")  #contrastive_loss

inputs = tf.keras.Input(shape=(None, dim), dtype ="float32", ragged=True)
x = DenseRagged(units=30, use_bias=True, activation='relu')(inputs)
x = PermopRagged()(x)
x = tf.keras.layers.Dense(200, activation='relu')(x)
outputs = tf.keras.layers.Dense(output_dim, activation='sigmoid')(x)
model3 = tf.keras.Model(inputs=inputs, outputs=outputs)
model3.compile(optimizer=optim, loss="mse")  #contrastive_loss

inputs = tf.keras.Input(shape=(None, dim), dtype ="float32", ragged=True)
x = DenseRagged(units=30, use_bias=True, activation='gelu')(inputs)
x = PermopRagged()(x)
x = tf.keras.layers.Dense(200, activation='gelu')(x)
outputs = tf.keras.layers.Dense(output_dim, activation='gelu')(x)
model4 = tf.keras.Model(inputs=inputs, outputs=outputs)
model4.compile(optimizer=optim, loss="mse")  #contrastive_loss


#model.summary()
#SVG(tf.keras.utils.model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
#tf.keras.utils.plot_model(model, to_file='Résultats/Résultats 2/model_multiple_circles.pdf', show_shapes=True)

list_models = [model1] if dataset_name[:5] == 'synth' else [model1, model2, model3, model4] 
best_CV_model, best_loss = list_models[0], np.inf

# Train the model 

if len(list_models) > 1:

    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    for mdl in list_models:
        final_val_loss = 0
        for train, test in kfold.split(np.hstack(PVs_train)):
            history = mdl.fit(tf.gather(data_train, train), np.hstack(PVs_train)[train], epochs=num_epochs, validation_data=(tf.gather(data_train, test), np.hstack(PVs_train)[test]), callbacks=[callback], verbose=0)
            val_loss = history.history['val_loss'][-1]
            final_val_loss += val_loss/num_folds
        if final_val_loss < best_loss:
            best_CV_model = mdl
            best_loss = final_val_loss
else:

    best_CV_model = list_models[0]

# Save the model
history = best_CV_model.fit(data_train, np.hstack(PVs_train), epochs=num_epochs, validation_data=(data_test, np.hstack(PVs_test)), callbacks=[callback], verbose=0)
best_CV_model.save('models/' + model_name)

# Study the results to see how the training went 

prediction = best_CV_model.predict(data_test)

prefix = 0

for hidx, hdim in enumerate(homdim):

    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if PV_type == 'PI':
            plt.imshow(np.flip(np.reshape(prediction[prefix + i][(hidx)*(PV_size*PV_size):(hidx+1)*(PV_size*PV_size)], [PV_size, PV_size]), 0), cmap='jet') #, vmin=0, vmax=1)
            plt.colorbar()
        elif PV_type == 'PL':
            nls = PV_params['num_landscapes']
            for lidx in range(nls):
                plt.plot(prediction[prefix + i][(hidx)*(PV_size*nls)+lidx*PV_size:(hidx)*(PV_size*nls)+(lidx+1)*PV_size])
    plt.suptitle('Train predicted PV in hdim ' + str(hdim))
    plt.savefig('results/' + dataset_name + '_predicted_PVs_h' + str(hdim) + '_on_train.png')

    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if PV_type == 'PI':
            plt.imshow(np.flip(np.reshape(PVs_test[hidx][prefix + i], [PV_size, PV_size]), 0), cmap='jet') #, vmin=0, vmax=1)
            plt.colorbar()
        elif PV_type == 'PL':
            nls = PV_params['num_landscapes']
            for lidx in range(nls):
                plt.plot(PVs_test[hidx][prefix + i][lidx*PV_size:(lidx+1)*PV_size]) 
    plt.suptitle('Train true PV in hdim ' + str(hdim))
    plt.savefig('results/' + dataset_name + '_true_PVs_h' + str(hdim) + '_on_train.png')

plt.figure()
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.scatter(data_test[prefix + i, :, 0], data_test[prefix + i, :, 1], s=3)
plt.suptitle('Train point cloud')
plt.savefig('results/' + dataset_name + '_point_clouds_on_train.png')

# Evaluation of the model and plot of the evolution of the loss 

history_dict = history.history
pck.dump(history_dict, open('results/' + model_name + "_train_history", 'wb'))

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs[:], np.log(loss[:]), 'bo', label='Training loss')
plt.plot(epochs[:], np.log(val_loss[:]), 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('results/' + dataset_name + '_loss_on_train.png')
