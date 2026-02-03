import numpy as np


def max_pool(picture, pool_size=2, stride=2):
    high, wide = picture.shape
    max_i = (high - pool_size) // stride + 1
    max_j = (wide - pool_size) // stride + 1
    output = np.zeros((max_i, max_j))
    for i in range(max_i):
        for j in range(max_j):
            window = picture[i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]
            output[i, j] = np.max(window)
    return output


def mean_pool(picture, pool_size=2, stride=2):
    high, wide = picture.shape
    mean_i = (high - pool_size) // stride + 1
    mean_j = (wide - pool_size) // stride + 1
    output = np.zeros((mean_i, mean_j))
    for i in range(mean_i):
        for j in range(mean_j):
            window = picture[i * stride:i * stride + pool_size, j * stride:j * stride + pool_size]
            output[i, j] = np.mean(window)
    return output

if __name__ == '__main__':
    feature_map = np.array([
        [1, 4, 2, 8],
        [3, 9, 5, 7],
        [2, 6, 4, 1],
        [5, 3, 7, 2]
    ])
    print(f'输入图像为:\n{feature_map}')
    # print(max_pool(feature_map, pool_size=2, stride=2))
    output_maxpool = max_pool(feature_map)
    print(f'最大池化后输出图片为:\n{output_maxpool}')
    output_meanpool = mean_pool(feature_map)
    print(f'平均池化后输出图片为:\n{output_meanpool}')
