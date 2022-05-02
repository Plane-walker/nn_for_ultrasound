import os
import h5py
import numpy as np
import pydicom
import cv2


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
    return image_normalized


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("dataset", data=arr, dtype=arr.dtype)


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:
        return f["dataset"][()]


def pre_import_data(data_type):
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
            image = normalization(image)
            images.append(np.reshape(image, (256, 256, 1)))
            labels.append(label.replace('.', '').encode('utf-8'))
    write_hdf5(np.array(images, dtype=np.float32), f'images_{data_type}.hdf5')
    write_hdf5(np.array(labels), f'labels_{data_type}.hdf5')


def import_data(data_type):
    images = load_hdf5(f'images_{data_type}.hdf5')
    labels = load_hdf5(f'labels_{data_type}.hdf5')
    return images, labels


if __name__ == '__main__':
    pre_import_data('train')
    pre_import_data('test')
