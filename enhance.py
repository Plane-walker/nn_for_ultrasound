from import_data import import_data
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2



def load_hdf5(infile):
    with h5py.File(infile, "r") as f:
        return f["dataset"][()]


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("dataset", data=arr, dtype=arr.dtype)

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
    enhance_images = []
    images_train, labels_train = import_data('train')
    print(images_train.shape)
    print(labels_train.shape)
    for i in range(8517):
        enhance_img = enhance_method(images_train[i, :, :, 0])
        enhance_img = normalization(enhance_img)
        enhance_images.append(np.reshape(enhance_img, (256, 256, 1)))
        print('第'+str(i)+'次增强图已完成')
    write_hdf5(np.array(enhance_images, dtype=np.float32), f'images_enhance_train.hdf5')


def enhance_test():
    images_test, labels_test = import_data('test')
    print(images_test.shape)
    print(labels_test.shape)
    enhance_images = []
    print(images_test.shape)
    print(labels_test.shape)
    for i in range(2676):
        enhance_img = enhance_method(images_test[i, :, :, 0])
        enhance_img = normalization(enhance_img)
        enhance_images.append(np.reshape(enhance_img, (256, 256, 1)))
        print('第'+str(i)+'次增强图已完成')
    write_hdf5(np.array(enhance_images, dtype=np.float32), f'images_enhance_test.hdf5')


def draw_enhance_pic():
    images_train = load_hdf5(f'images_enhance_train.hdf5')
    images_test = load_hdf5(f'images_enhance_test.hdf5')
    print(images_train.shape)
    print(images_test.shape)
    plt.subplot(1, 3, 1)
    plt.imshow(images_train[0, :, :, 0], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(images_train[1, :, :, 0], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(images_train[2, :, :, 0] , cmap='gray')
    plt.show()

if __name__ == '__main__':
    enhance_train()
    enhance_test()
    draw_enhance_pic()
