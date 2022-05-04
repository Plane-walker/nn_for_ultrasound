import os
import h5py
import numpy as np
import pydicom
import cv2
import tensorflow as tf
import tensorflow_io as tfio
from sklearn.model_selection import train_test_split


class Generator:
    def __init__(self, file, name):
        self.file = file
        self.name = name

    def __call__(self):
        with h5py.File(self.file, 'r') as hf:
            for image in hf[self.name]:
                yield image


def rgb2gray(rgb):
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return gray


def normalization(image):
    std = np.std(image)
    mean = np.mean(image)
    image_normalized = (image - mean) / std
    image_max = np.max(image_normalized)
    image_min = np.min(image_normalized)
    image_normalized = (image_normalized - image_min) / (image_max-image_min)
    return image_normalized * 255


def write_hdf5(images, labels, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("images", data=images, dtype=images.dtype)
        f.create_dataset("labels", data=labels, dtype=labels.dtype)


def load_hdf5(infile, name):
    with h5py.File(infile, "r") as f:
        return f[name][()]


def pre_import_train_data(data_type):
    images = []
    labels = []
    for label in os.listdir(f'{data_type.capitalize()} set/'):
        for image_file in os.listdir(f'{data_type.capitalize()} set/{label}'):
            image = pydicom.read_file(f'{data_type.capitalize()} set/{label}/{image_file}')
            image = np.array(image.pixel_array)

            image = image[113:739,114:983]
            image[14:47,253:292] = 0#把"P"去掉
            image[5:348,774:] = 0#修一修右上
            image[590:622,768:815] = 0#修掉右下的"14"
            image[524:574,799:807] = 0#修掉右下的两个点
            image[366:375,800:807] = 0#修掉右侧中间部位的两个点
            image[343:367,785:791] = 0#精修
            image = np.array(255 * (image / 255) ** 0.5, dtype='uint8')#伽马调整，选择参数0.5          

            if len(image.shape) == 3:
                image = rgb2gray(image)
            image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            # image = normalization(image)
            image = np.expand_dims(image, axis=2)
            image = np.concatenate((image, image, image), axis=-1)
            images.append(image)
            labels.append(label.replace('.', '').encode('utf-8'))
    mapping_to_numbers = {b'123': 0, b'1234': 1, b'4': 2, b'5678': 3, b'58': 4, b'67': 5}
    labels_int = np.zeros((len(labels)))
    for index, raw_label in enumerate(labels):
        labels_int[index] = mapping_to_numbers[raw_label]
    labels_one_hot = tf.one_hot(labels_int, 6)
    x_train, x_test, y_train, y_test = train_test_split(np.array(images, dtype=np.uint8), np.array(labels_one_hot), test_size=0.1)
    write_hdf5(x_train, y_train, f'train.hdf5')
    write_hdf5(x_test, y_test, f'val.hdf5')


def pre_import_test_data(data_type):
    images = []
    labels = []
    for label in os.listdir(f'{data_type.capitalize()} set/'):
        for image_file in os.listdir(f'{data_type.capitalize()} set/{label}'):
            image = pydicom.read_file(f'{data_type.capitalize()} set/{label}/{image_file}')
            image = np.array(image.pixel_array)

            image = image[200:820,155:964]
            image[10:38,214:250] = 0#把"M"去掉
            image = np.array(255 * (image / 255) ** 0.5, dtype='uint8')#伽马调整，选择参数0.5          

            if len(image.shape) == 3:
                image = rgb2gray(image)
            image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            # image = normalization(image)
            image = np.expand_dims(image, axis=2)
            image = np.concatenate((image, image, image), axis=-1)
            images.append(image)
            labels.append(label.replace('.', '').encode('utf-8'))
    mapping_to_numbers = {b'123': 0, b'1234': 1, b'4': 2, b'5678': 3, b'58': 4, b'67': 5}
    labels_int = np.zeros((len(labels)))
    for index, raw_label in enumerate(labels):
        labels_int[index] = mapping_to_numbers[raw_label]
    labels_one_hot = tf.one_hot(labels_int, 6)
    write_hdf5(np.array(images, dtype=np.uint8), np.array(labels_one_hot), f'test.hdf5')


def import_data(data_type):
    images = tfio.IODataset.from_hdf5(f'{data_type}.hdf5', dataset='/images')
    labels = tfio.IODataset.from_hdf5(f'{data_type}.hdf5', dataset='/labels')
    data = tf.data.Dataset.zip((images, labels)).batch(32, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    return data


if __name__ == '__main__':
    pre_import_train_data('train')
    pre_import_test_data('test')
