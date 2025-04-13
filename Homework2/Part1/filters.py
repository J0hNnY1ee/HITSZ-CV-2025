import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    flipped_kernel = np.flip(kernel, axis=(0, 1))
    pad_H = Hk //2
    pad_W = Wk //2
    padded_image = np.pad(image, ((pad_H, pad_H), (pad_W, pad_W)), mode='constant') # 我们需要补充几行空白
    out = np.zeros((Hi, Wi))
    for  i in range(Hi):
        for j in range(Wi):
            summation = 0.0
            for m in range(Hk):
                for n in range(Wk):
                    summation += padded_image[i + m, j + n] * flipped_kernel[m, n]
            out[i, j] = summation
    return out



def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    out = np.zeros((H + 2*pad_height, W + 2*pad_width))
    # 将原图像复制到画布中心
    out[pad_height:H+pad_height, pad_width:W+pad_width] = image
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape

    out = np.zeros((Hi, Wi))
    pad_height = Hk // 2
    pad_width = Wk // 2
    padded_image = zero_pad(image, pad_height, pad_width)
    flipped_kernel = np.flip(kernel)
    for i in range(Hi):
        for j in range(Wi):
            # 计算卷积
            out[i, j] = np.sum(padded_image[i:i+Hk, j:j+Wk] * flipped_kernel)
    

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    flipped_g = np.flip(g)
    out = conv_fast(f, flipped_g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g_mean = np.mean(g)
    g_zero_mean = g - g_mean
    out = cross_correlation(f, g_zero_mean)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    pad_height = Hg // 2
    pad_width = Wg // 2

    # 零填充图像
    padded_f = zero_pad(f, pad_height, pad_width)
    
    # 标准化模板
    mean_g = np.mean(g)
    std_g = np.std(g)
    if std_g < 1e-8:  # 模板无变化时直接返回0
        return np.zeros_like(f)
    g_normalized = (g - mean_g) / std_g

    out = np.zeros((Hf, Wf))

    # 遍历每个像素位置
    for i in range(Hf):
        for j in range(Wf):
            # 提取对应子图像
            patch = padded_f[i:i+Hg, j:j+Wg]
            
            # 标准化子图像
            mean_patch = np.mean(patch)
            std_patch = np.std(patch)
            if std_patch < 1e-8:  # 子图像无变化时响应为0
                out[i,j] = 0
                continue
            patch_normalized = (patch - mean_patch) / std_patch
            
            # 计算归一化互相关
            out[i,j] = np.sum(patch_normalized * g_normalized)
    
    return out
