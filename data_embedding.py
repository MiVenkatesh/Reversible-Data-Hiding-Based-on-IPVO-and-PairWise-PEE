import numpy as np

def embed_data_in_block(image, prediction_errors, two_segment_threshold, data):
    height, width = image.shape
    data_index = 0
    embedded_image = np.copy(image)

    for y in range(height):
        for x in range(width):
            if data_index >= len(data):
                break
            
            # error = prediction_errors[y, x]
            # if y == 0 and x < 10:  # print the first few prediction errors and the threshold
            #     print(f"Prediction error: {error}, threshold: {two_segment_threshold}")

            error = prediction_errors[y, x]
            if error > two_segment_threshold:
                embedded_image[y, x] += data[data_index]
                data_index += 1
            elif error < -two_segment_threshold:
                embedded_image[y, x] -= data[data_index]
                data_index += 1

    return embedded_image

