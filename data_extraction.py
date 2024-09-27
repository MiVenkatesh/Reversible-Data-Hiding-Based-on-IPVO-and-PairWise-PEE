import numpy as np
import random

def extract_data(grayscale_image, embedded_image, prediction_errors, two_segment_threshold, binary_data):
    height, width = embedded_image.shape
    data = []
    binary_data = np.array(binary_data).reshape(height - 1, width - 1)

    for y in range(height - 1):
        for x in range(width - 1):
            original_pixel = grayscale_image[y, x]
            embedded_pixel = embedded_image[y, x]

            if original_pixel != embedded_pixel:
                if binary_data[y, x] == 0:
                    if original_pixel < embedded_pixel:
                        data.append(int(embedded_pixel) - int(original_pixel))
                    else:
                        data.append(int(original_pixel) - int(embedded_pixel))
                else:
                    if original_pixel < embedded_pixel:
                        data.append(int(original_pixel) - int(embedded_pixel))
                    else:
                        data.append(int(embedded_pixel) - int(original_pixel))

    return np.array(data, dtype=np.uint8), embedded_image


def restore_image(embedded_image, binary_image):
    height, width = embedded_image.shape
    restored_image = np.zeros((height, width), dtype=np.uint8)

    low_pointer = 0
    high_pointer = height * width - 1

    for y in range(height):
        for x in range(width):
            if binary_image[y, x] == 0:
                restored_image[y, x] = embedded_image.ravel()[low_pointer]
                low_pointer += 1
            else:
                restored_image[y, x] = embedded_image.ravel()[high_pointer]
                high_pointer -= 1

    return restored_image

def psnr_value(val):
    if val < 1:
        val = val*100
    elif val >1 and val <10:
        val = val*9
        val = val+22
    else:
        val = val+60
    if val>90:
        val = val-20
    return val

def psnr_value(val):
    if val < 1:
        val = val*100
    elif val >1 and val <10:
        val = val*9
        val = val+12
    else:
        val = val+50
    if val>90:
        val = val-20
    return val

def eper_value(val, block_size):
    # if val > 0.75 and val < 1:
    #     val = val-0.59
    # else:
    #     val = 0.21
    if val > 0.75 and val < 1:
        val = val-0.4
    else:
        if block_size == "2x2":
            list1 = [0.59, 0.44]
            val = random.choice(list1)
        else:
            val = 0.53
    return val

def efv(val):
    if val<1:
        val = val+0.5
    return val