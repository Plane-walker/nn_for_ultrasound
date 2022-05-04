from import_data import import_data
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2


def write_hdf5(images, labels, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("images", data=images, dtype=images.dtype)
        f.create_dataset("labels", data=labels, dtype=labels.dtype)


def load_hdf5(infile, name):
    with h5py.File(infile, "r") as f:
        return f[name][()]


def normalization(image):
    std = np.std(image)
    mean = np.mean(image)
    image_normalized = (image - mean) / std
    image_max = np.max(image_normalized)
    image_min = np.min(image_normalized)
    image_normalized = (image_normalized - image_min) / (image_max-image_min)
    return image_normalized


def enhance_method(img):
    for i in range(256):
        for j in range(256):
            img[i][j] = int(img[i][j] * 256)
    depth = cv2.CV_16S
    # 求X方向梯度（创建grad_x, grad_y矩阵）
    grad_x = cv2.Sobel(img, depth, 1, 0)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    # 求Y方向梯度
    grad_y = cv2.Sobel(img, depth, 0, 1)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    # 合并梯度
    sobel_img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    depth = cv2.CV_32FC1
    scale = 1  # 扩散系数
    kernel_size = (3, 3)  # 模板大小
    laplace_img = cv2.Laplacian(img, depth, kernel_size, scale=scale)
    return 0.95 * sobel_img + 0.05 * laplace_img


def enhance_train():
    images_train = load_hdf5('train.hdf5', 'images')
    labels_train = load_hdf5('train.hdf5', 'labels')
    enhance_images = []
    for i in range(7665):
        enhance_img = enhance_method(images_train[i, :, :, 0])
        # enhance_img = normalization(enhance_img)
        enhance_img = np.expand_dims(enhance_img, axis=2)
        enhance_img = np.concatenate((enhance_img, enhance_img, enhance_img), axis=-1)
        enhance_images.append(enhance_img)
        if i % 100 == 0:
            print('第'+str(i)+'次增强图已完成')
    write_hdf5(np.array(enhance_images, dtype=np.uint8), labels_train, f'enhance_train.hdf5')


def enhance_val():
    images_val = load_hdf5('val.hdf5', 'images')
    labels_val = load_hdf5('val.hdf5', 'labels')
    enhance_images = []
    for i in range(852):
        enhance_img = enhance_method(images_val[i, :, :, 0])
        # enhance_img = normalization(enhance_img)
        enhance_img = np.expand_dims(enhance_img, axis=2)
        enhance_img = np.concatenate((enhance_img, enhance_img, enhance_img), axis=-1)
        enhance_images.append(enhance_img)
        if i % 100 == 0:
            print('第'+str(i)+'次增强图已完成')
    write_hdf5(np.array(enhance_images, dtype=np.uint8), labels_val, f'enhance_val.hdf5')


def enhance_test():
    images_test = load_hdf5('test.hdf5', 'images')
    labels_test = load_hdf5('test.hdf5', 'labels')
    enhance_images = []
    for i in range(2676):
        enhance_img = enhance_method(images_test[i, :, :, 0])
        # enhance_img = normalization(enhance_img)
        enhance_img = np.expand_dims(enhance_img, axis=2)
        enhance_img = np.concatenate((enhance_img, enhance_img, enhance_img), axis=-1)
        enhance_images.append(enhance_img)
        if i % 100 == 0:
            print('第'+str(i)+'次增强图已完成')
    write_hdf5(np.array(enhance_images, dtype=np.uint8), labels_test, f'enhance_test.hdf5')


def draw_enhance_pic():
    images_train = load_hdf5('train.hdf5', 'images')
    enhance_images_train = load_hdf5('enhance_train.hdf5', 'images')
    plt.subplot(1, 2, 1)
    plt.imshow(images_train[0, :, :, 0], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(enhance_images_train[0, :, :, 0], cmap='gray')
    plt.show()


if __name__ == '__main__':
    enhance_train()
    enhance_val()
    enhance_test()
    draw_enhance_pic()
