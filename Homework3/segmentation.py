import numpy as np
import random
import math
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
from skimage import color

### Clustering Methods for 1-D points
def kmeans(features, k, num_iters=500):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """
    print(features.shape)
    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        old_assignments = assignments.copy()
        diff = features[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances_sq = np.sum(diff**2, axis=2)
        assignments = np.argmin(distances_sq, axis=1) 
        
        if np.all(assignments == old_assignments):
            break
        ### END YOUR CODE
        for  j in range(k):
            points_in_cluster = features[assignments == j]
            if points_in_cluster.size > 0:
                centers[j] = np.mean(points_in_cluster, axis=0)
            else:
                centers[j] = features[np.random.choice(N)]
    return assignments

### Clustering Methods for colorful image
def kmeans_color(image, k, num_iters=500):
    """ Apply K-Means clustering to a color image.

    Args:
        image - Input color image with shape (H, W, 3).
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Cluster assignments for each pixel, reshaped to (H, W).
    """
    # Step 1: Get image dimensions
    H, W, _ = image.shape
    
    # Step 2: Flatten the image to (N, 3), where N = H * W
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # Step 3: Apply K-Means clustering
    assignments = kmeans(pixels, k, num_iters=num_iters)
    
    # Step 4: Reshape assignments back to (H, W)
    assignments = assignments.reshape(H, W)
    
    return assignments





#找每个点最后会收敛到的地方（peak）
def findpeak(data, idx, r):
    t = 0.01
    shift = np.array([1])
    data_point = data[:, idx]
    dataT = data.T
    data_pointT = data_point.T
    data_pointT = data_pointT.reshape(1, 3)

    # Runs until the shift is smaller than the set threshold
    while shift.all() > t:
        # 计算当前点和所有点之间的距离
        # 并筛选出在半径r内的点，计算mean vector（这里是最简单的均值，也可尝试高斯加权）
        # 用新的center（peak）更新当前点，直到满足要求跳出循环
        ### YOUR CODE HERE
        pass
        ### END YOUR CODE

    return data_pointT.T


# Mean shift algorithm
# 可以改写代码，鼓励自己的想法，但请保证输入输出与notebook一致
def meanshift(data, r):
    labels = np.zeros(len(data.T))
    peaks = [] #聚集的类中心
    label_no = 1 #当前label
    labels[0] = label_no

    # findpeak is called for the first index out of the loop
    peak = findpeak(data, 0, r)
    peakT = np.concatenate(peak, axis=0).T
    peaks.append(peakT)

    # Every data point is iterated through
    for idx in range(0, len(data.T)):
        # 遍历数据，寻找当前点的peak
        # 并实时关注当前peak是否会收敛到一个新的聚类（和已有peaks比较）
        # 若是，更新label_no，peaks，labels，继续
        # 若不是，当前点就属于已有类，继续
        ### YOUR CODE HERE
        pass
        ### END YOUR CODE
    #print(set(labels))
    return labels, np.array(peaks).T


# image segmentation
def segmIm(img, r):
    # Image gets reshaped to a 2D array
    img_reshaped = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # We will work now with CIELAB images
    imglab = color.rgb2lab(img_reshaped)
    # segmented_image is declared
    segmented_image = np.zeros((img_reshaped.shape[0], img_reshaped.shape[1]))


    labels, peaks = meanshift(imglab.T, r)
    # Labels are reshaped to only one column for easier handling
    labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

    # We iterate through every possible peak and its corresponding label
    for label in range(0, peaks.shape[1]):
        # Obtain indices for the current label in labels array
        inds = np.where(labels_reshaped == label + 1)[0]

        # The segmented image gets indexed peaks for the corresponding label
        corresponding_peak = peaks[:, label]
        segmented_image[inds, :] = corresponding_peak
    # The segmented image gets reshaped and turn back into RGB for display
    segmented_image = np.reshape(segmented_image, (img.shape[0], img.shape[1], 3))

    res_img=color.lab2rgb(segmented_image)
    res_img=color.rgb2gray(res_img)
    return res_img
def segmIm_parallel(img, r):
    from joblib import Parallel, delayed
    """
    并行化版本的图像分割函数。
    """
    # 将图像重塑为二维数组，方便处理
    img_reshaped = np.reshape(img, (img.shape[0] * img.shape[1], 3))

    # 将图像从RGB颜色空间转换到CIELAB颜色空间
    imglab = color.rgb2lab(img_reshaped)

    # 初始化分割后的图像
    segmented_image = np.zeros((img_reshaped.shape[0], img_reshaped.shape[1]))

    # 使用均值漂移算法进行聚类，得到标签和聚类中心
    labels, peaks = meanshift(imglab.T, r)

    # 将标签重塑为一列，方便后续处理
    labels_reshaped = np.reshape(labels, (labels.shape[0], 1))

    # 使用 Joblib 的 Parallel 和 delayed 实现并行化处理每个标签
    # n_jobs=-1 表示使用所有可用的 CPU 核心
    segmented_image = Parallel(n_jobs=-1)(
        # 对每个标签并行处理，避免显式循环
        delayed(lambda label: (
            # 找到当前标签对应的像素索引
            inds := np.where(labels_reshaped == label + 1)[0],
            # 将对应聚类中心赋值给分割图像的相应位置
            segmented_image.__setitem__((inds, slice(None)), peaks[:, label])
        )[-1])  # 返回修改后的 segmented_image
        for label in range(peaks.shape[1])  # 遍历所有标签
    )

    # 将并行处理的结果合并（因为 Parallel 返回的是一个列表）
    segmented_image = np.sum(segmented_image, axis=0)

    # 将分割图像重塑回原始图像的尺寸
    segmented_image = np.reshape(segmented_image, (img.shape[0], img.shape[1], 3))

    # 将分割图像从 CIELAB 转换回 RGB，再转换为灰度图像以便显示
    res_img = color.lab2rgb(segmented_image)
    res_img = color.rgb2gray(res_img)

    return res_img

### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """
    
    if mask_gt.shape != mask.shape:
        raise ValueError("mask_gt and mask must have the same shape.")
    
    correct_pixels = (mask_gt == mask).sum()  # 逻辑比较后求和
    

    total_pixels = mask_gt.size  # 总像素数是数组的大小
    
    accuracy = correct_pixels / total_pixels
    
    return accuracy

