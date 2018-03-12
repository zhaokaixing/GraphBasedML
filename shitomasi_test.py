import numpy as np
np.set_printoptions(threshold=np.inf)
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from skimage import img_as_ubyte

class graph_generation:
    def convert_image(self, img):
        ret, thresh1 = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
        for i in range(len(thresh1)):
            for j in range(len(thresh1[i])):
                if thresh1[i][j] == 255:
                    thresh1[i][j] = 0
                else:
                    thresh1[i][j] = 1
        return thresh1

    def skeletonize_image(self, img_convert):
        skeleton = skeletonize(img_convert)
        return skeleton

    def construct_array(self, length_image, start_x, start_y, cv_image):
        #print(cv_image[17][6])
        pixel_array = [[0 for i in range(3)] for i in range(3)]
        #print('start_x' + str(start_x) + 'start_y' + str(start_y))
        if (start_x - 1 <= length_image) & (start_y - 1 <= length_image):
            pixel_array[0][0] = cv_image[start_x - 1][start_y - 1]
        else:
            pixel_array[0][0] = -1
        if (start_x <= length_image) & (start_y - 1 <= length_image):
            pixel_array[0][1] = cv_image[start_x][start_y - 1]
        else:
            pixel_array[0][1] = -1
        if (start_x + 1 <= length_image) & (start_y - 1 <= length_image):
            pixel_array[0][2] = cv_image[start_x + 1][start_y - 1]
        else:
            pixel_array[0][2] = -1

        if (start_x - 1 <= length_image) & (start_y <= length_image):
            pixel_array[1][0] = cv_image[start_x - 1][start_y]
        else:
            pixel_array[1][0] = -1
        if (start_x <= length_image) & (start_y <= length_image):
            pixel_array[1][1] = cv_image[start_x][start_y]
            #print('aaa  ' + str(pixel_array[1][1]))
        else:
            pixel_array[1][1] = -1
        if (start_x + 1 <= length_image) & (start_y <= length_image):
            pixel_array[1][2] = cv_image[start_x + 1][start_y]
        else:
            pixel_array[1][2] = -1

        if (start_x - 1 <= length_image) & (start_y + 1 <= length_image):
            pixel_array[2][0] = cv_image[start_x - 1][start_y + 1]
        else:
            pixel_array[2][0] = -1
        if (start_x <= length_image) & (start_y + 1 <= length_image):
            pixel_array[2][1] = cv_image[start_x][start_y + 1]
        else:
            pixel_array[2][1] = -1
        if (start_x + 1 <= length_image) & (start_y + 1 <= length_image):
            pixel_array[2][2] = cv_image[start_x + 1][start_y + 1]
        else:
            pixel_array[2][2] = -1

        #print(pixel_array)
        return pixel_array

    def search_edge(self, corners, cv_image):
        k = 0
        length_image = len(cv_image)
        while k < len(corners):
            start_x = corners[k][0][0]
            start_y = corners[k][0][1]
            print(str(start_x) + ' ' + str(start_y))
            pixel_array = self.construct_array(length_image, start_x, start_y, cv_image)

            for i in range(len(pixel_array)):
                for j in range(len(pixel_array[0])):
                    if pixel_array[i][j] == 255:
                        #print(str(i) + 'qqq' + str(j))
                        if (i != start_x) & (j != start_y):
                            new_start_x = (j - 1) + start_x
                            new_start_y = (i - 1) + start_y
                            #print("test" + str(new_start_x) + ' ' + str(new_start_y))

            k = k + 1

    def generate_graph(self):
        img = cv2.imread('666.jpg')

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        image_convert = self.convert_image(gray)


        skeleton_image = self.skeletonize_image(image_convert)
        cv_image = img_as_ubyte(skeleton_image)
        #print(cv_image[14])

        corners = cv2.goodFeaturesToTrack(cv_image,25,0.01,10)
        corners = np.int0(corners)
        print(corners)

        #print(len(cv_image))
        print(cv_image[433][720])
        print(cv_image[314][347])
        #print(cv_image[342][580])

        #self.search_edge(corners, cv_image)

        for i in corners:
            x,y = i.ravel()
            cv2.circle(cv_image,(x,y),2,255,1)


        plt.imshow(cv_image, cmap=plt.cm.gray),plt.show()

if __name__ == "__main__":
    graph_object = graph_generation()
    graph_object.generate_graph()