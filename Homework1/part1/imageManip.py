import math

import numpy as np
from PIL import Image
from skimage import color, io


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)

    ### END YOUR CODE

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255.0
    return out


def crop_image(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index we want to include in our cropped image.
        start_col (int): The starting column index we want to include in our cropped image.
        num_rows (int): Number of rows in our desired cropped image.
        num_cols (int): Number of columns in our desired cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """

    out = None

    ### YOUR CODE HERE
    if image is None:
        return None
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是形状为 (height, width, 3) 的 NumPy 数组")
    if start_row < 0 or start_col < 0 or num_rows <= 0 or num_cols <= 0:
        raise ValueError("裁剪参数必须为非负整数且 num_rows 和 num_cols 必须大于 0")
    end_row = start_row + num_rows
    end_col = start_col + num_cols
    if end_row > image.shape[0] or end_col > image.shape[1]:
        raise ValueError("裁剪区域超出图像范围")
    out = image[start_row:end_row, start_col:end_col, :]
    ### END YOUR CODE

    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### YOUR CODE HERE
    # 检查输入图像是否为空
    if image is None:
        return None

    # 检查输入图像的形状是否正确
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是形状为 (height, width, 3) 的 NumPy 数组")
    out = 0.5 * (image**2)
    ### END YOUR CODE

    return out


def resize_image(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create the resized output image
    output_image = np.zeros(shape=(output_rows, output_cols, 3))

    # 2. Populate the `output_image` array using values from `input_image`
    #    > This should require two nested for loops!

    ### YOUR CODE HERE
    for i in range(output_rows):
        for j in range(output_cols):
            input_row = int(i * input_rows / output_rows)
            input_col = int(j * input_cols / output_cols)
            input_row = max(0, min(input_row, input_rows - 1))
            input_col = max(0, min(input_col, input_cols - 1))
            output_image[i, j, :] = input_image[input_row, input_col, :]

    ### END YOUR CODE

    # 3. Return the output image
    return output_image


def rotate2d(point, theta):
    """Rotate a 2D coordinate by some angle theta.

    Args:
        point (np.ndarray): A 1D NumPy array containing two values: an x and y coordinate.
        theta (float): An theta to rotate by, in radians.

    Returns:
        np.ndarray: A 1D NumPy array containing your rotated x and y values.
    """
    assert point.shape == (2,)
    assert isinstance(theta, float)

    # Reminder: np.cos() and np.sin() will be useful here!

    ## YOUR CODE HERE
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    point_col = point.reshape(2, 1)
    rotated_point = R @ point_col
    return rotated_point.flatten()
    ### END YOUR CODE


def rotate_image(input_image, theta):
    """Rotate an image by some angle theta.

    Args:
        input_image (np.ndarray): RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        theta (float): Angle to rotate our image by, in radians.

    Returns:
        (np.ndarray): Rotated image, with the same shape as the input.
    """
    input_rows, input_cols, channels = input_image.shape
    assert channels == 3

    # 1. Create an output image with the same shape as the input
    output_image = np.zeros_like(input_image)

    ## YOUR CODE HERE
        # 计算图像中心
    # 计算图像中心
    center_row = (input_rows - 1) / 2.0
    center_col = (input_cols - 1) / 2.0

    # 遍历输出图像的每个像素
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            # 计算输出图像中当前像素的坐标（相对于中心）
            output_row = i - center_row
            output_col = j - center_col

            # 反向旋转坐标（顺时针旋转）
            input_row = output_row * np.cos(-theta) + output_col * np.sin(-theta)
            input_col = -output_row * np.sin(-theta) + output_col * np.cos(-theta)

            # 将坐标转换回原图像的索引
            input_row += center_row
            input_col += center_col

            # 检查索引是否在原图像范围内
            if 0 <= input_row < input_rows and 0 <= input_col < input_cols:
                # 使用双线性插值获取像素值
                input_row_int = int(input_row)
                input_col_int = int(input_col)
                dx = input_row - input_row_int
                dy = input_col - input_col_int

                if input_row_int + 1 < input_rows and input_col_int + 1 < input_cols:
                    # 双线性插值
                    value = (1 - dx) * (1 - dy) * input_image[input_row_int, input_col_int, :] + \
                            dx * (1 - dy) * input_image[input_row_int + 1, input_col_int, :] + \
                            (1 - dx) * dy * input_image[input_row_int, input_col_int + 1, :] + \
                            dx * dy * input_image[input_row_int + 1, input_col_int + 1, :]
                    output_image[i, j, :] = value
                else:
                    # 如果超出范围，使用最近邻插值
                    output_image[i, j, :] = input_image[input_row_int, input_col_int, :]
    ### END YOUR CODE

    # 3. Return the output image
    return output_image
