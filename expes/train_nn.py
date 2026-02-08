# Training of the NN

import dill as pck
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# Convert data to PyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_train_torch = [torch.FloatTensor(data_train[i]).to(device) for i in range(len(data_train))]
data_test_torch = [torch.FloatTensor(data_test[i]).to(device) for i in range(len(data_test))]

# Normalize the PIs
if normalize:
    for hidx in range(len(homdim)):
        MPV = np.max(PVs_train[hidx])
        PVs_train[hidx] /= MPV
        PVs_test[hidx]  /= MPV

output_dim = sum([PVs_train[hidx].shape[1] for hidx in range(len(homdim))])

# Convert PVs to PyTorch tensors
PVs_train_torch = [torch.FloatTensor(PVs_train[hidx]).to(device) for hidx in range(len(homdim))]
PVs_test_torch = [torch.FloatTensor(PVs_test[hidx]).to(device) for hidx in range(len(homdim))]
PVs_train_stacked = torch.hstack(PVs_train_torch).to(device)
PVs_test_stacked = torch.hstack(PVs_test_torch).to(device)

# Definition of the NN models

class DenseRaggedNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activations):
        super(DenseRaggedNet, self).__init__()
        self.dense_layers = nn.ModuleList()
        self.dense_layers.append(DenseRagged(hidden_dims[0], activation=activations[0]))
        for i in range(1, len(hidden_dims)):
            self.dense_layers.append(DenseRagged(hidden_dims[i], activation=activations[i]))
        self.permop = PermopRagged()
        
        # Dense layers after pooling
        self.final_layers = nn.ModuleList()
        final_activations = activations[len(hidden_dims):-1]
        prev_dim = hidden_dims[-1]
        for i, hidden_dim in enumerate([50, 100, 200]):
            self.final_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.final_layer = nn.Linear(prev_dim, output_dim)
        self.final_activation = nn.Sigmoid() if activations[-1] == 'sigmoid' else nn.Identity()
    
    def forward(self, x):
        for dense in self.dense_layers[:-1]:
            x = dense(x)
        x = self.permop(x)
        for layer in self.final_layers:
            x = torch.relu(layer(x))
        x = self.final_layer(x)
        x = self.final_activation(x)
        return x

class EarlyStopping:
    def __init__(self, patience=200, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# Create models
if dataset_name[:5] == 'synth':
    optim_lr = 5e-4
    optimizer_class = optim.Adamax
else:
    optim_lr = 5e-4
    optimizer_class = optim.Adam

def create_model1():
    return DenseRaggedNet(dim, output_dim, [30, 20, 10], ['relu', 'relu', 'relu'])

def create_model_full():
    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = DenseRagged(30, activation='relu')
            self.dense2 = DenseRagged(20, activation='relu')
            self.dense3 = DenseRagged(10, activation='relu')
            self.permop = PermopRagged()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 100)
            self.fc3 = nn.Linear(100, 200)
            self.fc_out = nn.Linear(200, output_dim)
        
        def forward(self, x):
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)
            x = self.permop(x)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.sigmoid(self.fc_out(x))
            return x
    return FullModel()

def create_model2():
    class Model2(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = DenseRagged(30, activation='gelu')
            self.dense2 = DenseRagged(20, activation='gelu')
            self.dense3 = DenseRagged(10, activation='gelu')
            self.permop = PermopRagged()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 100)
            self.fc3 = nn.Linear(100, 200)
            self.fc_out = nn.Linear(200, output_dim)
        
        def forward(self, x):
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.dense3(x)
            x = self.permop(x)
            x = torch.nn.functional.gelu(self.fc1(x))
            x = torch.nn.functional.gelu(self.fc2(x))
            x = torch.nn.functional.gelu(self.fc3(x))
            x = torch.nn.functional.gelu(self.fc_out(x))
            return x
    return Model2()

def create_model3():
    class Model3(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = DenseRagged(30, activation='relu')
            self.permop = PermopRagged()
            self.fc1 = nn.Linear(30, 200)
            self.fc_out = nn.Linear(200, output_dim)
        
        def forward(self, x):
            x = self.dense1(x)
            x = self.permop(x)
            x = torch.relu(self.fc1(x))
            x = torch.sigmoid(self.fc_out(x))
            return x
    return Model3()

def create_model4():
    class Model4(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = DenseRagged(30, activation='gelu')
            self.permop = PermopRagged()
            self.fc1 = nn.Linear(30, 200)
            self.fc_out = nn.Linear(200, output_dim)
        
        def forward(self, x):
            x = self.dense1(x)
            x = self.permop(x)
            x = torch.nn.functional.gelu(self.fc1(x))
            x = torch.nn.functional.gelu(self.fc_out(x))
            return x
    return Model4()

# Create and compile models
model1 = create_model1().to(device)
model2 = create_model2().to(device)
model3 = create_model3().to(device)
model4 = create_model4().to(device)

list_models = [model1] if dataset_name[:5] == 'synth' else [model1, model2, model3, model4]
best_CV_model, best_loss = list_models[0], np.inf

# Training function
def train_epoch(model, data, targets, optimizer, criterion):
    model.train()
    total_loss = 0
    for i in range(len(data)):
        optimizer.zero_grad()
        x = data[i].unsqueeze(0)
        output = model(x)
        target = torch.FloatTensor(targets[i:i+1]).to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data)

def evaluate(model, data, targets, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i in range(len(data)):
            x = data[i].unsqueeze(0)
            output = model(x)
            target = torch.FloatTensor(targets[i:i+1]).to(device)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(data)

criterion = nn.MSELoss()

# Train the model
if len(list_models) > 1:
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=0)
    indices = np.arange(len(data_train_torch))
    stacked_targets = np.hstack(PVs_train)
    
    for mdl in list_models:
        optimizer = optimizer_class(mdl.parameters(), lr=optim_lr)
        final_val_loss = 0
        fold_idx = 0
        for train_idx, test_idx in kfold.split(stacked_targets):
            early_stopping = EarlyStopping(patience=200, min_delta=1e-5)
            for epoch in range(num_epochs):
                train_loss = train_epoch(mdl, [data_train_torch[i] for i in train_idx], stacked_targets[train_idx], optimizer, criterion)
                val_loss = evaluate(mdl, [data_train_torch[i] for i in test_idx], stacked_targets[test_idx], criterion)
                if early_stopping(val_loss):
                    break
            final_val_loss += val_loss / num_folds
            fold_idx += 1
        if final_val_loss < best_loss:
            best_CV_model = mdl
            best_loss = final_val_loss
else:
    best_CV_model = list_models[0]

# Fine-tune on full training set
optimizer = optimizer_class(best_CV_model.parameters(), lr=optim_lr)
early_stopping = EarlyStopping(patience=200, min_delta=1e-5)
history_loss = []
history_val_loss = []

for epoch in range(num_epochs):
    train_loss = train_epoch(best_CV_model, data_train_torch, np.hstack(PVs_train), optimizer, criterion)
    val_loss = evaluate(best_CV_model, data_test_torch, np.hstack(PVs_test), criterion)
    history_loss.append(train_loss)
    history_val_loss.append(val_loss)
    if early_stopping(val_loss):
        break

# Save the model
torch.save(best_CV_model.state_dict(), 'models/' + model_name + '.pt')

# Study the results
best_CV_model.eval()
with torch.no_grad():
    predictions = []
    for i in range(len(data_test_torch)):
        x = data_test_torch[i].unsqueeze(0)
        pred = best_CV_model(x)
        predictions.append(pred.cpu().numpy())
    prediction = np.vstack(predictions)

prefix = 0

for hidx, hdim in enumerate(homdim):
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        if PV_type == 'PI':
            plt.imshow(np.flip(np.reshape(prediction[prefix + i][(hidx)*(PV_size*PV_size):(hidx+1)*(PV_size*PV_size)], [PV_size, PV_size]), 0), cmap='jet')
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
            plt.imshow(np.flip(np.reshape(PVs_test[hidx][prefix + i], [PV_size, PV_size]), 0), cmap='jet')
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

# Evaluation and loss plotting
history_dict = {
    'loss': history_loss,
    'val_loss': history_val_loss
}
pck.dump(history_dict, open('results/' + model_name + "_train_history", 'wb'))

loss = history_loss
val_loss = history_val_loss

epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs[:], np.log(loss[:]), 'bo', label='Training loss')
plt.plot(epochs[:], np.log(val_loss[:]), 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('results/' + dataset_name + '_loss_on_train.png')


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
