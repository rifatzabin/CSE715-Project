import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"

# Constants
IMG_SIZE = 128
VOLUME_SLICES = 100
VOLUME_START_AT = 22
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',
    2: 'EDEMA',
    3: 'ENHANCING'
}

TRAIN_DATASET_PATH = '/mnt/SSD2/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'

# Dice Coefficient Metric
# def dice_coef(y_true, y_pred, smooth=1.0):
#     y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
#     y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef(y_true, y_pred, smooth=1.0):
    """
    Computes the average Dice coefficient for a multi-class segmentation task.

    Parameters:
    - y_true: Ground truth tensor (one-hot encoded, shape: [batch_size, H, W, num_classes]).
    - y_pred: Predicted tensor (softmax probabilities, shape: [batch_size, H, W, num_classes]).
    - smooth: Smoothing constant to avoid division by zero.

    Returns:
    - average_dice: Average Dice coefficient across all classes.
    """
    # Ensure both tensors have the same dtype (float32)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Compute intersection and sums for all classes simultaneously
    intersection = tf.reduce_sum(y_true * y_pred, axis=(0, 1, 2))
    y_true_sum = tf.reduce_sum(y_true, axis=(0, 1, 2))
    y_pred_sum = tf.reduce_sum(y_pred, axis=(0, 1, 2))

    # Compute Dice coefficient for each class
    dice = (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)

    # Average over all classes
    average_dice = tf.reduce_mean(dice)
    return average_dice

def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    """
    Computes the Dice coefficient for the necrotic/core class (class index 1).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(tf.abs(y_true[..., 1] * y_pred[..., 1]))
    return (2. * intersection) / (tf.reduce_sum(tf.square(y_true[..., 1])) + tf.reduce_sum(tf.square(y_pred[..., 1])) + epsilon)


def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    """
    Computes the Dice coefficient for the edema class (class index 2).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(tf.abs(y_true[..., 2] * y_pred[..., 2]))
    return (2. * intersection) / (tf.reduce_sum(tf.square(y_true[..., 2])) + tf.reduce_sum(tf.square(y_pred[..., 2])) + epsilon)


def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    """
    Computes the Dice coefficient for the enhancing tumor class (class index 3).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(tf.abs(y_true[..., 3] * y_pred[..., 3]))
    return (2. * intersection) / (tf.reduce_sum(tf.square(y_true[..., 3])) + tf.reduce_sum(tf.square(y_pred[..., 3])) + epsilon)



def precision(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())

def sensitivity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())

def specificity(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_negatives = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = tf.reduce_sum(tf.round(tf.clip_by_value(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())

# U-Net Model
def build_unet(input_layer, ker_init='he_normal', dropout=0.2):
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(input_layer)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv4)
    drop4 = Dropout(dropout)(conv4)

    up5 = UpSampling2D(size=(2, 2))(drop4)
    up5 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=ker_init)(up5)
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(merge5)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=ker_init)(up6)
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(merge6)
    conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv6)

    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=ker_init)(up7)
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(merge7)
    conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv7)

    output = Conv2D(4, 1, activation='softmax')(conv7)
    return Model(inputs=input_layer, outputs=output)


# Build the U-Net model
input_layer = Input((IMG_SIZE, IMG_SIZE, 2))
model = build_unet(input_layer)

# Compile the model with the required metrics
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=[
        'accuracy',
        tf.keras.metrics.MeanIoU(num_classes=4),
        dice_coef,
        precision,
        sensitivity,
        specificity,
        dice_coef_necrotic,
        dice_coef_edema,
        dice_coef_enhancing
    ]
)

# Plot Model
plot_model(model, show_shapes=True, dpi=70)

# Data Generator
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, ids, batch_size=1, img_size=IMG_SIZE, n_channels=2, shuffle=True):
        self.ids = ids
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return len(self.ids) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_ids = [self.ids[k] for k in indexes]
        return self.__data_generation(batch_ids)

    def __data_generation(self, batch_ids):
        X = np.zeros((self.batch_size * VOLUME_SLICES, self.img_size, self.img_size, self.n_channels))
        y = np.zeros((self.batch_size * VOLUME_SLICES, self.img_size, self.img_size, 4))
        for i, ID in enumerate(batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, ID)
            flair = nib.load(os.path.join(case_path, f"{ID}_flair.nii")).get_fdata()
            t1ce = nib.load(os.path.join(case_path, f"{ID}_t1ce.nii")).get_fdata()
            seg = nib.load(os.path.join(case_path, f"{ID}_seg.nii")).get_fdata()
            for j in range(VOLUME_SLICES):
                X[i * VOLUME_SLICES + j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (self.img_size, self.img_size))
                X[i * VOLUME_SLICES + j, :, :, 1] = cv2.resize(t1ce[:, :, j + VOLUME_START_AT], (self.img_size, self.img_size))
                mask = cv2.resize(seg[:, :, j + VOLUME_START_AT], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                y[i * VOLUME_SLICES + j] = tf.one_hot(mask, 4)
        return X / np.max(X), y

# Split Data
train_val_dirs = [
    d for d in os.listdir(TRAIN_DATASET_PATH)
    if os.path.isdir(os.path.join(TRAIN_DATASET_PATH, d))
    and os.path.exists(os.path.join(TRAIN_DATASET_PATH, d, f"{d}_seg.nii"))
]

train_ids, val_ids = train_test_split(train_val_dirs, test_size=0.2, random_state=42)

# Create Generators
train_gen = DataGenerator(train_ids)
val_gen = DataGenerator(val_ids)


from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

# Path to save the best model
model_checkpoint_path = "best_unet_brain_tumor_seg.keras"

# Callbacks
callbacks = [
    ModelCheckpoint(
        filepath=model_checkpoint_path,  # Path to save the model
        monitor='val_loss',             # Metric to monitor
        save_best_only=True,            # Save only the best model
        save_weights_only=False,        # Save the full model (architecture + weights)
        mode='min',                     # Look for the minimum value of val_loss
        verbose=1                       # Verbosity for logging
    ),
    CSVLogger('training.log'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
]

# Training
history = model.fit(train_gen, validation_data=val_gen, epochs=200, callbacks=callbacks)
