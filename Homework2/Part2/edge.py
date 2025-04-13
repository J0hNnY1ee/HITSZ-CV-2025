import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')
    kernel_flipped = np.flip(kernel)
    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            # Get the region of interest
            region = padded[i:i+Hk, j:j+Wk]
            # Perform element-wise multiplication and sum the result
            out[i, j] = np.sum(region * kernel_flipped)
    ### END YOUR CODE

    return out

import numpy as np

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """
    i, j = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2
    x = i - center
    y = j - center
    distance_sq = x**2 + y**2
    kernel = (1 / (2 * np.pi * sigma**2)) * np.exp(-distance_sq / (2 * sigma**2))
    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel_x = np.array([[0.5, 0, -0.5]])
    out =  conv(img, kernel_x)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    kernel_y = np.array([[0.5], [0], [-0.5]])
    out =  conv(img, kernel_y)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Ix = partial_x(img)
    Iy = partial_y(img)
    G = np.sqrt(Ix**2 + Iy**2)
    theta = np.arctan2(Iy, Ix) * (180 / np.pi)
    theta[theta < 0] += 360
    ### END YOUR CODE

    return G, theta



def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W) in degrees.

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # 将梯度方向四舍五入到最近的45度，并处理为0°/45°/90°/135°
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 180).astype(np.int32)  # 修正：使用180度周期

    for i in range(H):
        for j in range(W):
            angle = theta[i, j]

            # 根据不同方向选择比较的邻域像素
            if angle == 0:
                # 水平方向，比较左右
                neighbors = [(i, j-1), (i, j+1)]
            elif angle == 45:
                # 主对角线方向（左上-右下），比较对角线像素
                neighbors = [(i-1, j-1), (i+1, j+1)]
            elif angle == 90:
                # 垂直方向，比较上下
                neighbors = [(i-1, j), (i+1, j)]
            elif angle == 135:
                # 副对角线方向（右上-左下），比较对角线像素
                neighbors = [(i-1, j+1), (i+1, j-1)]
            else:
                # 异常处理（理论上不会触发）
                neighbors = []

            # 边界检查与比较
            if len(neighbors) != 2:
                out[i, j] = 0
                continue

            (n1_i, n1_j) = neighbors[0]
            (n2_i, n2_j) = neighbors[1]

            # 超出边界的像素视为0
            g1 = G[n1_i, n1_j] if (0 <= n1_i < H and 0 <= n1_j < W) else 0.0
            g2 = G[n2_i, n2_j] if (0 <= n2_i < H and 0 <= n2_j < W) else 0.0

            # 仅保留局部最大值
            if G[i, j] >= g1 and G[i, j] >= g2:
                out[i, j] = G[i, j]
            else:
                out[i, j] = 0

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    strong_edges = img > high
    weak_edges = (img <= high) & (img > low)
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    # 定义8邻域方向
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]

    # 初始化队列，包含所有强边缘的坐标
    queue = list(indices)

    while queue:
        i, j = queue.pop(0)  # 取出队列中的第一个元素
        for dy, dx in directions:
            ni, nj = i + dy, j + dx
            # 检查坐标有效性及是否为未处理的弱边缘
            if 0 <= ni < H and 0 <= nj < W and weak_edges[ni, nj]:
                edges[ni, nj] = True       # 标记为边缘
                weak_edges[ni, nj] = False # 标记为已处理
                queue.append((ni, nj))     # 加入队列以继续搜索
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    # 1. Gaussian filter
    kernel = gaussian_kernel(kernel_size, sigma)
    img = conv(img, kernel)
    # 2. Gradient
    G, theta = gradient(img)
    # 3. Non-maximum suppression
    img = non_maximum_suppression(G, theta)
    # 4. Double thresholding
    strong_edges, weak_edges = double_thresholding(img, high, low)
    # 5. Edge tracking by hysteresis
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge
