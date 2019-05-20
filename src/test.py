import numpy as np
from keras.utils import to_categorical


def generate():
    batch_size = 1
    num_steps = 10
    vocabulary = 100
    current_idx = 0
    data = range(11)
    skip_step = 1
    x = np.zeros((batch_size, num_steps))
    y = np.zeros((batch_size, num_steps, vocabulary))
    while True:
        for i in range(batch_size):
            if current_idx + num_steps >= len(data):
                # reset the index back to the start of the data set
                current_idx = 0
            x[i, :] = data[current_idx:current_idx + num_steps]
            temp_y = data[current_idx + 1:current_idx + num_steps + 1]
            # convert all of temp_y into a one hot representation
            y[i, :, :] = to_categorical(temp_y, num_classes=vocabulary)
            current_idx += skip_step
        yield x, y

t = generate()
for x, y in t:
    print(x)
    print(y)
    exit()