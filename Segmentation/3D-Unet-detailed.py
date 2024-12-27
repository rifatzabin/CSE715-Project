# %%

import os
import cv2
import glob
import PIL
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import data
from skimage.util import montage
import skimage.transform as skTrans
from skimage.transform import rotate
from skimage.transform import resize
from PIL import Image, ImageOps


# neural imaging
import nilearn as nl
import nibabel as nib
import nilearn.plotting as nlplt
#!pip install git+https://github.com/miykael/gif_your_nifti # nifti to gif
import gif_your_nifti.core as gif2nif


# ml libs
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras import layers
#from tensorflow.keras.layers import preprocessing
from tensorflow.keras.layers import Normalization

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


# %%

# DEFINE seg-areas
SEGMENT_CLASSES = {
    0 : 'NOT tumor',
    1 : 'NECROTIC/CORE', # or NON-ENHANCING tumor CORE
    2 : 'EDEMA',
    3 : 'ENHANCING' # original 4 -> converted into 3 later
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5
VOLUME_SLICES = 100
VOLUME_START_AT = 22 # first slice of volume that we will include


# %%

TRAIN_DATASET_PATH = '/mnt/SSD2/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = '/mnt/SSD2/archive//BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'

test_image_flair=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii').get_fdata()
test_image_t1=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1.nii').get_fdata()
test_image_t1ce=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
test_image_t2=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
test_mask=nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii').get_fdata()


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize = (20, 10))
slice_w = 25
ax1.imshow(test_image_flair[:,:,test_image_flair.shape[0]//2-slice_w], cmap = 'gray')
ax1.set_title('Image flair')
ax2.imshow(test_image_t1[:,:,test_image_t1.shape[0]//2-slice_w], cmap = 'gray')
ax2.set_title('Image t1')
ax3.imshow(test_image_t1ce[:,:,test_image_t1ce.shape[0]//2-slice_w], cmap = 'gray')
ax3.set_title('Image t1ce')
ax4.imshow(test_image_t2[:,:,test_image_t2.shape[0]//2-slice_w], cmap = 'gray')
ax4.set_title('Image t2')
ax5.imshow(test_mask[:,:,test_mask.shape[0]//2-slice_w])
ax5.set_title('Mask')
plt.show()


# %%

# Skip 50:-50 slices since there is not much to see
fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(test_image_t1[50:-50,:,:]), 90, resize=True), cmap ='gray')
plt.show()

# %%


# Skip 50:-50 slices since there is not much to see
fig, ax1 = plt.subplots(1, 1, figsize = (15,15))
ax1.imshow(rotate(montage(test_mask[60:-60,:,:]), 90, resize=True), cmap ='gray')
plt.show()


# %%

shutil.copy2(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii', './test_gif_BraTS20_Training_001_flair.nii')
gif2nif.write_gif_normal('./test_gif_BraTS20_Training_001_flair.nii')



# %%
niimg = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii')
nimask = nl.image.load_img(TRAIN_DATASET_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii')

fig, axes = plt.subplots(nrows=4, figsize=(30, 40))


nlplt.plot_anat(niimg,
                title='BraTS20_Training_001_flair.nii plot_anat',
                axes=axes[0])

nlplt.plot_epi(niimg,
               title='BraTS20_Training_001_flair.nii plot_epi',
               axes=axes[1])

nlplt.plot_img(niimg,
               title='BraTS20_Training_001_flair.nii plot_img',
               axes=axes[2])

nlplt.plot_roi(nimask,
               title='BraTS20_Training_001_flair.nii with mask plot_roi',
               bg_img=niimg,
               axes=axes[3], cmap='Paired')

plt.show()



# %%

# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    total_loss = 0
    for i in range(class_num):
        # Ensure both tensors are float32
        y_true_f = tf.keras.backend.cast(tf.keras.backend.flatten(y_true[:, :, :, :, i]), dtype='float32')
        y_pred_f = tf.keras.backend.cast(tf.keras.backend.flatten(y_pred[:, :, :, :, i]), dtype='float32')

        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) /
                (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))
        total_loss += loss
    total_loss = total_loss / class_num
    return total_loss


def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    y_true_f = tf.keras.backend.cast(y_true[:, :, :, :, 1], dtype='float32')
    y_pred_f = tf.keras.backend.cast(y_pred[:, :, :, :, 1], dtype='float32')

    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + epsilon)


def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    y_true_f = tf.keras.backend.cast(y_true[:, :, :, :, 2], dtype='float32')
    y_pred_f = tf.keras.backend.cast(y_pred[:, :, :, :, 2], dtype='float32')

    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + epsilon)


def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    y_true_f = tf.keras.backend.cast(y_true[:, :, :, :, 3], dtype='float32')
    y_pred_f = tf.keras.backend.cast(y_pred[:, :, :, :, 3], dtype='float32')

    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + epsilon)


def precision(y_true, y_pred):
    # Ensure consistent data types
    y_true = tf.cast(y_true, dtype='float32')
    y_pred = tf.cast(y_pred, dtype='float32')

    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    return true_positives / (predicted_positives + tf.keras.backend.epsilon())



# Computing Sensitivity
def sensitivity(y_true, y_pred):
    # Ensure consistent data types
    y_true = tf.cast(y_true, dtype='float32')
    y_pred = tf.cast(y_pred, dtype='float32')

    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    return true_positives / (possible_positives + tf.keras.backend.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    # Ensure consistent data types
    y_true = tf.cast(y_true, dtype='float32')
    y_pred = tf.cast(y_pred, dtype='float32')

    true_negatives = tf.reduce_sum(tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = tf.reduce_sum(tf.round(tf.clip_by_value(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + tf.keras.backend.epsilon())


# %%
IMG_SIZE=128


# %%

# source https://naomi-fridman.medium.com/multi-class-image-segmentation-a5cc671e647a

def build_unet(inputs, ker_init, dropout):
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv1)

    pool = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(pool)
    conv = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv)

    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(pool1)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv2)

    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(pool2)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv3)

    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv5)
    drop5 = Dropout(dropout)(conv5)

    up7 = Conv3D(256, (2, 2, 2), activation='relu', padding='same', kernel_initializer=ker_init)(
        UpSampling3D(size=(2, 2, 2))(drop5))
    merge7 = concatenate([conv3, up7])
    conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(merge7)
    conv7 = Conv3D(256, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv7)

    up8 = Conv3D(128, (2, 2, 2), activation='relu', padding='same', kernel_initializer=ker_init)(
        UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = concatenate([conv2, up8])
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(merge8)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv8)

    up9 = Conv3D(64, (2, 2, 2), activation='relu', padding='same', kernel_initializer=ker_init)(
        UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = concatenate([conv, up9])
    conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(merge9)
    conv9 = Conv3D(64, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv9)

    up = Conv3D(32, (2, 2, 2), activation='relu', padding='same', kernel_initializer=ker_init)(
        UpSampling3D(size=(2, 2, 2))(conv9))
    merge = concatenate([conv1, up])
    conv = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(merge)
    conv = Conv3D(32, (3, 3, 3), activation='relu', padding='same', kernel_initializer=ker_init)(conv)

    conv10 = Conv3D(4, (1, 1, 1), activation='softmax')(conv)

    return Model(inputs=inputs, outputs=conv10)


input_layer = Input((IMG_SIZE, IMG_SIZE, IMG_SIZE, 2))  # Update to 2 channels
model = build_unet(input_layer, 'he_normal', 0.2)
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing]
)


# %%

plot_model(model,
           show_shapes = True,
           show_dtype=False,
           show_layer_names = True,
           rankdir = 'TB',
           expand_nested = False,
           dpi = 70)


# %%

# lists of directories with studies
train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]

# file BraTS20_Training_355 has ill formatted name for for seg.nii file
train_and_val_directories.remove(TRAIN_DATASET_PATH + 'BraTS20_Training_355')


def pathListIntoIds(dirList):
    x = []
    for i in range(0, len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/') + 1:])
    return x


train_and_test_ids = pathListIntoIds(train_and_val_directories);

train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2)
train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15)


# %%

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples'
        # Initialize arrays for 3D volumes
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size, *self.dim, 4))

        for i, case_id in enumerate(Batch_ids):
            case_path = os.path.join(TRAIN_DATASET_PATH, case_id)

            # Load modalities
            flair = nib.load(os.path.join(case_path, f'{case_id}_flair.nii')).get_fdata()
            ce = nib.load(os.path.join(case_path, f'{case_id}_t1ce.nii')).get_fdata()
            seg = nib.load(os.path.join(case_path, f'{case_id}_seg.nii')).get_fdata()

            # Select slices and resize to fit the model
            flair = np.stack(
                [cv2.resize(flair[:, :, z], (self.dim[0], self.dim[1])) for z in range(self.dim[2])],
                axis=-1
            )
            ce = np.stack(
                [cv2.resize(ce[:, :, z], (self.dim[0], self.dim[1])) for z in range(self.dim[2])],
                axis=-1
            )
            seg = np.stack(
                [cv2.resize(seg[:, :, z], (self.dim[0], self.dim[1])) for z in range(self.dim[2])],
                axis=-1
            )

            # Assign channels for input
            X[i, :, :, :, 0] = flair
            X[i, :, :, :, 1] = ce

            # Generate masks
            seg[seg == 4] = 3  # Map label 4 to label 3
            mask = tf.one_hot(seg, 4)  # One-hot encode the segmentation mask
            y[i] = mask.numpy()

        return X / np.max(X), y

training_generator = DataGenerator(train_ids, dim=(128, 128, 128), n_channels=2)
valid_generator = DataGenerator(val_ids, dim=(128, 128, 128), n_channels=2)
test_generator = DataGenerator(test_ids, dim=(128, 128, 128), n_channels=2)


# %%

# show number of data for each dir
def showDataLayout():
    plt.bar(["Train", "Valid", "Test"],
            [len(train_ids), len(val_ids), len(test_ids)], align='center', color=['green', 'red', 'blue'])
    plt.legend()

    plt.ylabel('Number of images')
    plt.title('Data distribution')

    plt.show()


showDataLayout()


# %%

csv_logger = CSVLogger('training.log', separator=',', append=False)


callbacks = [
#     keras.callbacks.EarlyStopping(monitor='loss', min_delta=0,
#                               patience=2, verbose=1, mode='auto'),
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.000001, verbose=1),
#  keras.callbacks.ModelCheckpoint(filepath = 'model_.{epoch:02d}-{val_loss:.6f}.m5',
#                             verbose=1, save_best_only=True, save_weights_only = True)
        csv_logger
    ]


# %%

K.clear_session()

history = model.fit(
    training_generator,
    epochs=35,
    steps_per_epoch=len(training_generator),
    validation_data=valid_generator,
    callbacks=callbacks
)
model.save("model_x1_1.h5")



# %%

############ load trained model ################
model = keras.models.load_model('model_x1_1.h5',
                                   custom_objects={ 'accuracy' : tf.keras.metrics.MeanIoU(num_classes=4),
                                                   "dice_coef": dice_coef,
                                                   "precision": precision,
                                                   "sensitivity":sensitivity,
                                                   "specificity":specificity,
                                                   "dice_coef_necrotic": dice_coef_necrotic,
                                                   "dice_coef_edema": dice_coef_edema,
                                                   "dice_coef_enhancing": dice_coef_enhancing
                                                  }, compile=False)

history = pd.read_csv('training.log', sep=',', engine='python')

hist=history

############### ########## ####### #######

# hist=history.history

acc=hist['accuracy']
val_acc=hist['val_accuracy']

epoch=range(len(acc))

loss=hist['loss']
val_loss=hist['val_loss']

train_dice=hist['dice_coef']
val_dice=hist['val_dice_coef']

f,ax=plt.subplots(1,4,figsize=(16,8))

ax[0].plot(epoch,acc,'b',label='Training Accuracy')
ax[0].plot(epoch,val_acc,'r',label='Validation Accuracy')
ax[0].legend()

ax[1].plot(epoch,loss,'b',label='Training Loss')
ax[1].plot(epoch,val_loss,'r',label='Validation Loss')
ax[1].legend()

ax[2].plot(epoch,train_dice,'b',label='Training dice coef')
ax[2].plot(epoch,val_dice,'r',label='Validation dice coef')
ax[2].legend()

ax[3].plot(epoch,hist['mean_io_u'],'b',label='Training mean IOU')
ax[3].plot(epoch,hist['val_mean_io_u'],'r',label='Validation mean IOU')
ax[3].legend()

plt.show()



# %%

# mri type must one of 1) flair 2) t1 3) t1ce 4) t2 ------- or even 5) seg
# returns volume of specified study at `path`
def imageLoader(path):
    image = nib.load(path).get_fdata()
    X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels))
    for j in range(VOLUME_SLICES):
        X[j + VOLUME_SLICES * c, :, :, 0] = cv2.resize(image[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
        X[j + VOLUME_SLICES * c, :, :, 1] = cv2.resize(ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));

        y[j + VOLUME_SLICES * c] = seg[:, :, j + VOLUME_START_AT];
    return np.array(image)


# load nifti file at `path`
# and load each slice with mask from volume
# choose the mri type & resize to `IMG_SIZE`
def loadDataFromDir(path, list_of_files, mriType, n_images):
    scans = []
    masks = []
    for i in list_of_files[:n_images]:
        fullPath = glob.glob(i + '/*' + mriType + '*')[0]
        currentScanVolume = imageLoader(fullPath)
        currentMaskVolume = imageLoader(glob.glob(i + '/*seg*')[0])
        # for each slice in 3D volume, find also it's mask
        for j in range(0, currentScanVolume.shape[2]):
            scan_img = cv2.resize(currentScanVolume[:, :, j], dsize=(IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_AREA).astype('uint8')
            mask_img = cv2.resize(currentMaskVolume[:, :, j], dsize=(IMG_SIZE, IMG_SIZE),
                                  interpolation=cv2.INTER_AREA).astype('uint8')
            scans.append(scan_img[..., np.newaxis])
            masks.append(mask_img[..., np.newaxis])
    return np.array(scans, dtype='float32'), np.array(masks, dtype='float32')

# brains_list_test, masks_list_test = loadDataFromDir(VALIDATION_DATASET_PATH, test_directories, "flair", 5)



# %%
def predictByPath(case_path, case):
    files = next(os.walk(case_path))[2]
    X = np.empty((128, IMG_SIZE, IMG_SIZE, 2))  # Ensure depth is 128

    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_flair.nii')
    flair = nib.load(vol_path).get_fdata()

    vol_path = os.path.join(case_path, f'BraTS20_Training_{case}_t1ce.nii')
    ce = nib.load(vol_path).get_fdata()

    # Resize or pad depth to match 128
    for j in range(128):
        if j < flair.shape[2]:  # Use existing slices if available
            X[j, :, :, 0] = cv2.resize(flair[:, :, j], (IMG_SIZE, IMG_SIZE))
            X[j, :, :, 1] = cv2.resize(ce[:, :, j], (IMG_SIZE, IMG_SIZE))
        else:  # Pad with zeros if depth is less than 128
            X[j, :, :, 0] = np.zeros((IMG_SIZE, IMG_SIZE))
            X[j, :, :, 1] = np.zeros((IMG_SIZE, IMG_SIZE))

    # Add batch dimension
    X = np.expand_dims(X, axis=0)  # Shape becomes (1, 128, 128, 128, 2)

    return model.predict(X / np.max(X), verbose=1)


def showPredictsById(case, start_slice=60):
    path = f"/mnt/SSD2/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
    gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
    origImage = nib.load(os.path.join(path, f'BraTS20_Training_{case}_flair.nii')).get_fdata()
    p = predictByPath(path, case)[0]  # Access the first batch (3D volume)

    core = p[:, :, :, 1]
    edema = p[:, :, :, 2]
    enhancing = p[:, :, :, 3]

    plt.figure(figsize=(18, 50))
    f, axarr = plt.subplots(1, 6, figsize=(18, 50))

    for i in range(6):  # for each image, add brain background
        axarr[i].imshow(cv2.resize(origImage[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray",
                        interpolation='none')

    axarr[0].imshow(cv2.resize(origImage[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].title.set_text('Original image flair')
    curr_gt = cv2.resize(gt[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    axarr[1].imshow(curr_gt, cmap="Reds", interpolation='none', alpha=0.3)  # ,alpha=0.3,cmap='Reds'
    axarr[1].title.set_text('Ground truth')
    axarr[2].imshow(p[start_slice, :, :, 1:4], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[2].title.set_text('all classes')
    axarr[3].imshow(edema[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    axarr[4].imshow(core[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[5].imshow(enhancing[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[5].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
    plt.show()

# showPredictsById(case=test_ids[0][-3:])
# showPredictsById(case=test_ids[1][-3:])
# showPredictsById(case=test_ids[2][-3:])
# showPredictsById(case=test_ids[3][-3:])
# showPredictsById(case=test_ids[4][-3:])
# showPredictsById(case=test_ids[5][-3:])
# showPredictsById(case=test_ids[6][-3:])

# Assuming test_ids contains case IDs like ['001', '002', '003', ...]
for i in range(min(7, len(test_ids))):  # Visualize up to 7 cases
    showPredictsById(case=test_ids[i][-3:])


# %%

case = test_ids[3][-3:]
path = f"/mnt/SSD2/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
p = predictByPath(path, case)

# Remove the batch dimension from the prediction
p = np.squeeze(p)  # Shape becomes (128, 128, 128, 4)

core = p[:, :, :, 1]
edema = p[:, :, :, 2]
enhancing = p[:, :, :, 3]

i = 25  # Slice at
eval_class = 2  # 0: 'NOT tumor', 1: 'ENHANCING', 2: 'CORE', 3: 'WHOLE'

# Use only one class for per-class evaluation
gt[gt != eval_class] = 1
resized_gt = cv2.resize(gt[:, :, i + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

plt.figure()
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(resized_gt, cmap="gray")
axarr[0].title.set_text('ground truth')
axarr[1].imshow(p[i, :, :, eval_class], cmap="gray")
axarr[1].title.set_text(f'predicted class: {SEGMENT_CLASSES[eval_class]}')
plt.show()


# %%

model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics = ['accuracy',tf.keras.metrics.MeanIoU(num_classes=4), dice_coef, precision, sensitivity, specificity, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing] )
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_generator, batch_size=16, callbacks= callbacks)
print("test loss, test acc:", results)


# %%

model.summary()


# %%

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


# Assuming you have a model that outputs the class probabilities
# and you already have 'p' as the predicted output for the given slice.

# Define a function to compute the CAM
def compute_cam(model, image, target_class):
    # Use the correct layer name from your model
    last_conv_layer = model.get_layer("conv3d_22")  # Replace "conv2d_22" with "conv3d_22"

    # Create a gradient model
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image)
        loss = predictions[:, target_class]

    # Compute the gradient of the loss w.r.t. the convolutional layer output
    grads = tape.gradient(loss, conv_output)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))  # Adjust for 3D convolutions

    # Multiply each channel in the feature map array by the corresponding gradient
    conv_output = conv_output.numpy()[0]  # Remove batch dimension
    pooled_grads = pooled_grads.numpy()

    cam = np.zeros(conv_output.shape[:3])  # Create a blank CAM array
    for i in range(len(pooled_grads)):
        cam += pooled_grads[i] * conv_output[:, :, :, i]  # Sum weighted feature maps

    # Normalize the CAM
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image.shape[2], image.shape[1]))  # Resize to match input image dimensions
    cam = cam / np.max(cam) if np.max(cam) != 0 else cam  # Avoid division by zero

    return cam

# Preprocess the input image for CAM computation
image = np.expand_dims(p, axis=0)  # Add batch dimension, shape: (1, 128, 128, 128, 4)
image = np.expand_dims(image[:, :, :, :, eval_class], axis=-1)  # Select class, shape: (1, 128, 128, 128, 1)
image = np.repeat(image, 2, axis=-1)  # Repeat to match model's expected input shape, shape: (1, 128, 128, 128, 2)

# Compute the CAM for the selected class
# Select a specific slice from the CAM for visualization
cam_slice = cam[i + VOLUME_START_AT, :, :]  # Select the same slice as in ground truth

# Visualize the ground truth and CAM
plt.figure()
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(resized_gt, cmap="gray")
axarr[0].title.set_text('Ground truth')
axarr[1].imshow(cam_slice, cmap="jet")  # Use 'jet' colormap for CAM visualization
axarr[1].title.set_text(f'Class Activation Map (Class: {SEGMENT_CLASSES[eval_class]})')
plt.show()




# %%

def compute_cam(model, image, target_class):
    # Use the correct last convolutional layer
    last_conv_layer = model.get_layer("conv3d_22")
    grad_model = tf.keras.models.Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image)  # Forward pass
        loss = predictions[:, target_class]  # Focus on the target class

    # Compute gradients of the loss with respect to the convolutional output
    grads = tape.gradient(loss, conv_output)

    # Pool gradients spatially (average over depth, height, and width)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))

    conv_output = conv_output.numpy()[0]  # Remove batch dimension
    pooled_grads = pooled_grads.numpy()

    # Compute the CAM
    cam = np.zeros(conv_output.shape[:3])  # Create an empty CAM
    for i in range(len(pooled_grads)):
        cam += pooled_grads[i] * conv_output[:, :, :, i]

    # Normalize the CAM
    cam = np.maximum(cam, 0)  # Ensure all values are non-negative
    cam = cam / np.max(cam) if np.max(cam) != 0 else cam  # Normalize to [0, 1]

    # Resize the CAM to match the brain image slice dimensions
    cam_slice = cam[image.shape[1] // 2]  # Select a central slice
    cam_slice = cv2.resize(cam_slice, (brain_image_slice.shape[1], brain_image_slice.shape[0]))

    return cam_slice

# Preprocess the input image
image = np.expand_dims(p, axis=0)  # Add batch dimension, shape: (1, depth, height, width, channels)
image = np.expand_dims(image[:, :, :, :, eval_class], axis=-1)  # Select the target class, shape: (1, depth, height, width, 1)
image = np.repeat(image, 2, axis=-1)  # Repeat to match the model's input channels, shape: (1, depth, height, width, 2)


# Compute the CAM
cam_slice = compute_cam(model, image, eval_class)

# Visualize the CAM
plt.figure(figsize=(10, 5))
f, axarr = plt.subplots(1, 2, figsize=(12, 6))

# Display the brain image slice
axarr[0].imshow(brain_image_slice, cmap="gray")
axarr[0].title.set_text('Lát cắt hình ảnh não (T1)')

# Overlay the CAM on the brain image slice
axarr[1].imshow(brain_image_slice, cmap="gray", alpha=0.7)  # Base image with transparency
axarr[1].imshow(cam_slice, cmap="jet", alpha=0.5)  # CAM with transparency
axarr[1].title.set_text(f'Bản đồ kích hoạt lớp (Lớp: {SEGMENT_CLASSES[eval_class]})')

plt.show()


# %%

import os
import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure

# Đường dẫn và tải dữ liệu
case = test_ids[3][-3:]
path = f"/mnt/SSD2/archive/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
gt = nib.load(os.path.join(path, f'BraTS20_Training_{case}_seg.nii')).get_fdata()
p = predictByPath(path, case)

# Squeeze batch dimension from predictions
p = np.squeeze(p)  # Shape becomes (128, 128, 128, 4)

# Lấy các lớp từ kết quả dự đoán
core = p[:, :, :, 1]
edema = p[:, :, :, 2]
enhancing = p[:, :, :, 3]

# Chọn slice và class cần đánh giá
i = 60  # slice at
eval_class = 2  # 0: 'NOT tumor', 1: 'ENHANCING', 2: 'CORE', 3: 'WHOLE'

# Chỉ sử dụng một lớp (core) trong ground truth để so sánh
gt[gt != eval_class] = 0  # Đánh dấu các lớp khác là '0'

# Resize ground truth để phù hợp với kích thước hiển thị
resized_gt = cv2.resize(gt[:, :, i + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

# Kiểm tra giá trị của dự đoán và ground truth
print(np.min(enhancing), np.max(enhancing))
print(np.min(resized_gt), np.max(resized_gt))

# Áp dụng ngưỡng cho lớp dự đoán để làm nổi bật vùng có máu (enhancing)
enhancing_slice = enhancing[i + VOLUME_START_AT, :, :]  # Correct indexing
threshold = 0.3  # Thử ngưỡng thấp hơn
enhancing_slice[enhancing_slice < threshold] = 0

# Điều chỉnh độ tương phản cho enhancing_slice
enhanced_enhancing = exposure.rescale_intensity(enhancing_slice, in_range=(0, 1), out_range=(0, 255))

# Hiển thị hình ảnh
plt.figure()
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(resized_gt, cmap="gray")
axarr[0].title.set_text('Ground Truth')
axarr[1].imshow(enhanced_enhancing, cmap="hot")  # Sử dụng cmap "hot" để thấy rõ hơn
axarr[1].title.set_text(f'Predicted Class: {SEGMENT_CLASSES[eval_class]}')
plt.show()
