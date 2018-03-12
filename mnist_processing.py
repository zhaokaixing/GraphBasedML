import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image

class dataset_processing:
    def decode_idx3_ubyte(self, idx3_ubyte_file, saveFlag, labels):
        with open(idx3_ubyte_file, 'rb') as f:
            buf = f.read()

        offset = 0
        magic, image_count, rows, cols = struct.unpack_from('>IIII', buf, offset)
        offset += struct.calcsize('>IIII')
        images = np.empty((image_count,rows, cols))
        image_size = rows * cols
        fmt = '>' + str(image_size) + 'B'

        for i in range(image_count):

            images[i] = np.array(struct.unpack_from(fmt, buf, offset)).reshape((rows,cols))

            if saveFlag == True:
                im = Image.fromarray(np.uint8(images[i]))
                im.save('/Users/zkx/PycharmProjects/GraphProject/trainset/' + str(labels[i]) + '_' + str(i) + '.png')

            offset += struct.calcsize(fmt)

        return images

    def decode_idx1_ubyte(self, idx1_ubyte_file):
        with open(idx1_ubyte_file, 'rb') as f:
            buf = f.read()

        offset = 0
        magic, label_num = struct.unpack_from('>II', buf, offset)
        offset += struct.calcsize('>II')
        labels = np.zeros((label_num))

        for i in range(label_num):
            labels[i] = np.array(struct.unpack_from('>B', buf, offset))
            offset += struct.calcsize('>B')

        return labels

    def MNIST2vector(self, data_array):
        length,row,col = data_array.shape
        return data_array.reshape((length, row*col))

    def generate_trainset(self, image_file, label_file):
        labels = self.decode_idx1_ubyte(label_file)
        self.decode_idx3_ubyte(image_file, True, labels)

if __name__ == "__main__":
    data_object = dataset_processing()
    data_object.generate_trainset('/Users/zkx/PycharmProjects/GraphProject/train-images-idx3-ubyte',
                                  '/Users/zkx/PycharmProjects/GraphProject/train-labels-idx1-ubyte')

