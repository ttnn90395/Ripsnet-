
"""
Title: Point cloud classification with PointNet
Author: [David Griffiths](https://dgriffiths3.github.io)
Date created: 2020/05/25
Last modified: 2020/05/26
Description: Implementation of PointNet for ModelNet10 classification.
"""
"""
# Point cloud classification
"""

"""
## Introduction
Classification, detection and segmentation of unordered 3D point sets i.e. point clouds
is a core problem in computer vision. This example implements the seminal point cloud
deep learning paper [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593). For a
detailed introduction on PointNet see [this blog
post](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a).
"""

"""
## Setup
If using colab first install trimesh with `!pip install trimesh`.
"""

import os
print(os.getcwd())
from pathlib import Path
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import dill as pck
from sklearn.preprocessing import LabelEncoder

from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
print(sys.path)

from helper_fctns.get_names import get_dir_data, get_suffix_dataset
from helper_fctns.create_pointnet import create_pointnet

tf.random.set_seed(1234)

# GPU support:
use_GPU = True
if use_GPU:
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


NUM_POINTS = 1024
#NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS = 200
suffix = f'_npts_{NUM_POINTS}_epochs_{NUM_EPOCHS}_bs_{BATCH_SIZE}_'


"""
Load the saved point clouds for train and test from files.
"""

# Check if the execution is on the cluster or not:
on_cluster = False


# Setting some parameters:
representation  = 'PI'
modelnet_choice = 10
data_name = 'modelnet' + str(modelnet_choice)
normalize_pc = True
homdims = [0,1]
PI_size = 25
wgt_name = 'quadratic'
pct_noise = 0.02
fold = 0

NUM_CLASSES = modelnet_choice

#Set the data directories:
cwd = os.getcwd()
data_base_dir = os.path.join(cwd, f'datasets/saved_datasets/{representation}/{data_name}')
model_save_dir = os.path.join(cwd, f'models/pointnet/{data_name}')
results_dir = os.path.join(cwd, f'results/pointnet/{data_name}')

figures_output_dir   = os.path.join(results_dir, 'figures')
Path(figures_output_dir).mkdir(parents=True, exist_ok=True)

kwargs_data_dir = {
                   'representation': representation,
                   'data': f'modelnet{modelnet_choice}',
                   'dataset_choice': 'ml_train',
                   'hom_dim': homdims,
                   'PIsz': PI_size,
                   #'scaling_factor': scaling_factor,
                   'wgt_name': wgt_name,
                   #'normalize_vect': normalize,
                   'normalize_pc': normalize_pc,
                   'num_samples': NUM_POINTS,
                   'pct_noise': 0,
                   #'sample_even': sample_even,
                   #'resolution': 100,
                   #'num_landscapes': 5,
                   'on_cluster': on_cluster,
}

kwargs_suffix = {
                 'representation': representation,
                 'data': f'modelnet{modelnet_choice}',
                 'hom_dim': homdims,
                 'PIsz': PI_size,
                 # 'scaling_factor': scaling_factor,
                 'wgt_name': wgt_name,
                 # 'normalize_vect': normalize,
                 'normalize_pc': normalize_pc,
                 'num_samples':NUM_POINTS,
                 'pct_noise': 0,
                 # 'sample_even': sample_even,
                 # 'resolution': 100,
                 # 'num_landscapes': 5,
                 'fold': fold,
                 }

suffix = get_suffix_dataset(**kwargs_suffix)
ml_train_data_dir = get_dir_data(**kwargs_data_dir)
# Path(ml_train_data_dir).mkdir(parents=True, exist_ok=True)
ml_train_data_file = os.path.join(ml_train_data_dir, suffix + '.pkl')
assert(os.path.isfile(ml_train_data_file))

kwargs_data_dir.update({'dataset_choice': 'test', 'pct_noise': pct_noise})
kwargs_suffix.update({'pct_noise': pct_noise})
test_data_dir = get_dir_data(**kwargs_data_dir)
suffix = get_suffix_dataset(**kwargs_suffix)
# Path(test_data_dir).mkdir(parents=True, exist_ok=True)
test_data_file = os.path.join(test_data_dir, suffix + '.pkl')
# print(test_data_file)
assert(os.path.isfile(test_data_file))

### Loading the data from files:
# Clean pointclouds for training:
data_trn = pck.load(open(ml_train_data_file, 'rb'))
pc_train_ml_train = data_trn["data_train"]
pc_test_ml_train = data_trn['data_test']
label_train_ml_train = data_trn["label_train"]
label_test_ml_train = data_trn["label_test"]

# noisy (and clean) pointclouds for evaluation:
data_eval = pck.load(open(test_data_file, 'rb'))
pc_test_eval = data_eval['data_test']
pc_test_eval_clean = data_eval["data_test_clean"]
label_test_eval = data_eval["label_test"]


### Encoding the labels:
le = LabelEncoder().fit(np.concatenate([label_train_ml_train, label_test_ml_train, label_test_eval]))
label_train_ml_train = le.transform(label_train_ml_train)
label_test_ml_train = le.transform(label_test_ml_train)
label_test_eval = le.transform(label_test_eval)


"""
Our data can now be read into a `tf.data.Dataset()` object. We set the shuffle buffer
size to the entire size of the dataset as prior to this the data is ordered by class.

"""

train_dataset = tf.data.Dataset.from_tensor_slices((pc_train_ml_train, label_train_ml_train))
test_dataset = tf.data.Dataset.from_tensor_slices((pc_test_ml_train, label_test_ml_train))
eval_dataset = tf.data.Dataset.from_tensor_slices((pc_test_eval, label_test_eval))
eval_dataset_clean = tf.data.Dataset.from_tensor_slices((pc_test_eval_clean, label_test_eval))

train_dataset = train_dataset.shuffle(len(pc_train_ml_train)).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(pc_test_ml_train)).batch(BATCH_SIZE)
eval_dataset = eval_dataset.shuffle(len(pc_test_eval)).batch(BATCH_SIZE)
eval_dataset_clean = eval_dataset_clean.shuffle(len(pc_test_eval_clean)).batch(BATCH_SIZE)

## Creating the pointnet model:
model = create_pointnet(NUM_CLASSES=NUM_CLASSES, NUM_POINTS=NUM_POINTS, BATCH_SIZE=BATCH_SIZE)
model.summary()

"""
### Train model
Once the model is defined it can be trained like any other standard classification model
using `.compile()` and `.fit()`.
"""

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)

# Save the model
model_name = f'pointnet{modelnet_choice}_num_points_{NUM_POINTS}_epochs_{NUM_EPOCHS}_batch_size_{BATCH_SIZE}_weights'
model.save_weights(os.path.join(model_save_dir, model_name))

"""
## Visualize predictions
We can use matplotlib to visualize our trained model performance.
"""

### Get predictions on noisy and clean evaluation data:
data = test_dataset.take(1)
data_eval = eval_dataset.take(1)
data_eval_clean = eval_dataset_clean.take(1)

points_eval, labels_eval = list(data_eval)[0]
points_eval = points_eval[:8, ...]
labels_eval = labels_eval[:8, ...]

points_eval_clean, labels_eval_clean = list(data_eval_clean)[0]
points_eval_clean = points_eval_clean[:8, ...]
labels_eval_clean = labels_eval_clean[:8, ...]


# run evaluation data through model
preds = model.predict(points_eval)
preds = tf.math.argmax(preds, -1)

# run clean evaluation data through model
preds_clean = model.predict(points_eval_clean)
preds_clean = tf.math.argmax(preds_clean, -1)

points_eval = points_eval.numpy()


# Get the evaluation accuracies of the pointnet model:

eval_loss_noisy, eval_acc_noisy =  model.evaluate(eval_dataset, verbose=0)
eval_loss_clean, eval_acc_clean =  model.evaluate(eval_dataset_clean, verbose=0)

print(f'\nPointnet evaluation - clean:     acc: {eval_acc_clean}, loss: {eval_loss_clean}\n')
print(f'Pointnet evaluation - noisy {pct_noise}: acc: {eval_acc_noisy}, loss: {eval_loss_noisy}')

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points_eval[i, :, 0], points_eval[i, :, 1], points_eval[i, :, 2], s=0.5)
    ax.set_title(
        "pred: {:}, label: {:}".format(
        le.inverse_transform(preds)[i], le.inverse_transform(labels_eval)[i]
            # CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
    plt.suptitle(f'Predictions on noisy evaluation data. Noise: {pct_noise}')
    plt.savefig(os.path.join(figures_output_dir, f'predictions_on_noisy_eval_{suffix}.jpg'))
# plt.show()
# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points_eval_clean[i, :, 0], points_eval_clean[i, :, 1], points_eval_clean[i, :, 2], s=0.5)
    ax.set_title(
        "pred: {:}, label: {:}".format(
            le.inverse_transform(preds_clean)[i], le.inverse_transform(labels_eval_clean)[i]
            # CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
    plt.suptitle('Predictions on clean evaluation data.')
plt.savefig(os.path.join(figures_output_dir, f'predictions_on_clean_eval.jpg'))
# plt.show()

### Plot loss curves:
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs[:], loss[:], 'g', label='Training loss')
plt.plot(epochs[:], val_loss[:], 'b', label='Validation loss')
plt.title('RN training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
Path(os.path.join(figures_output_dir, 'loss_curves')).mkdir(parents=True, exist_ok=True)
plt.savefig(os.path.join(figures_output_dir, 'loss_curves', f'pointnet{modelnet_choice}{suffix}loss_on_train.jpg'))

plt.close('all')

