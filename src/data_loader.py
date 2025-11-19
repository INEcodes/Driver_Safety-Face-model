"""Data loading utilities for driver inattention dataset."""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator




def download_kaggle_dataset(url):
# Use opendatasets to download the dataset in Colab
import opendatasets as od
od.download(url)




def get_image_generators(data_path, img_height=224, img_width=224, batch_size=32, val_split=0.2):
datagen = ImageDataGenerator(
rescale=1./255,
validation_split=val_split,
rotation_range=20,
width_shift_range=0.2,
height_shift_range=0.2,
horizontal_flip=True
)


train_gen = datagen.flow_from_directory(
os.path.join(data_path, 'train'),
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode='categorical',
subset='training'
)


val_gen = datagen.flow_from_directory(
os.path.join(data_path, 'train'),
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode='categorical',
subset='validation'
)


test_gen = datagen.flow_from_directory(
os.path.join(data_path, 'test'),
target_size=(img_height, img_width),
batch_size=batch_size,
class_mode='categorical'
)


return train_gen, val_gen, test_gen
