from skimage.morphology import skeletonize
import sknw
import numpy as np
import matplotlib.pyplot as plt
import cv2
from characteristic import Characteristic
import os

def eachFile(filepath):
    file_path_array = []
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        file_path_array.append(child)
    return file_path_array

def transfer_matrix(image):
    ret, transfer_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    for i in range(len(transfer_image)):
        for j in range(len(transfer_image[i])):
            if transfer_image[i][j] == 255:
                transfer_image[i][j] = 0
            else:
                transfer_image[i][j] = 1
    return transfer_image

def draw_edges(graph):
    node_array = []
    for (s, e) in graph.edges():
        ps = graph[s][e]['pts']
        if len(ps) < 35:
            if (graph.degree(s) == 1):
                node_array.append(s)
            if (graph.degree(e) == 1):
                node_array.append(e)

    for n in node_array:
        graph.remove_node(n)

    node_array = []
    for n in graph.nodes():
        if (graph.degree(n) == 0):
            node_array.append(n)

    for n in node_array:
        graph.remove_node(n)

    for (s, e) in graph.edges():
        ps = graph[s][e]['pts']
        plt.plot(ps[:, 1], ps[:, 0], 'green')

def draw_nodes(graph):
    node, nodes = graph.node, graph.nodes()
    for i in nodes:
        ps = np.array([node[i]['o'] for i in nodes])
        plt.plot(ps[:, 1], ps[:, 0], 'r.')

def generate_characteristic_set(graph):
    charac_object = Characteristic()
    charac_object.num_node = len(graph.nodes())
    charac_object.num_edge = len(graph.edges())

    d1 = 0
    d2 = 0
    node, nodes = graph.node, graph.nodes()
    for i in nodes:
        if (graph.degree(i) == 1):
            d1 += 1
        elif (graph.degree(i) == 2):
            d2 += 1
    charac_object.degree_1 = d1
    charac_object.degree_2 = d2

    return charac_object

def show_graph(image):
    plt.imshow(image, cmap='gray')
    plt.title('Build Graph')
    plt.show()



def generate_characteristic_array():
    result_array = []
    file_path_array = eachFile('/Users/zkx/PycharmProjects/GraphProject/Base/')
    for path_str in file_path_array:
        result_array_without_type = []
        path_split = path_str.split('/')
        if path_str != '/Users/zkx/PycharmProjects/GraphProject/Base/.DS_Store':
            img = cv2.imread(path_str, 0)
            pic = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)

            image = transfer_matrix(pic)
            skeleton = skeletonize(image).astype(np.uint16)
            graph = sknw.build_sknw(skeleton, multi=False)

            draw_edges(graph)
            draw_nodes(graph)
            charac_object = generate_characteristic_set(graph)
            result_array_without_type.append(charac_object.num_node)
            result_array_without_type.append(charac_object.num_edge)
            result_array_without_type.append(charac_object.degree_1)
            result_array_without_type.append(charac_object.degree_2)

            charac_object.type = (path_split[len(path_split)-1]).split('_')[0]
            result_array_without_type.append(int(charac_object.type))

            result_array.append(result_array_without_type)
            #show_graph(image)

    result_array.sort(key=lambda x:x[4])

    #print(result_array)
    return result_array

def generate_request_charac_array(file_path):
    img = cv2.imread(file_path, 0)
    pic = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)

    image = transfer_matrix(pic)
    skeleton = skeletonize(image).astype(np.uint16)
    graph = sknw.build_sknw(skeleton, multi=False)

    draw_edges(graph)
    draw_nodes(graph)
    charac_object = generate_characteristic_set(graph)

    req_result_array = []
    req_result_array.append(charac_object.num_node)
    req_result_array.append(charac_object.num_edge)
    req_result_array.append(charac_object.degree_1)
    req_result_array.append(charac_object.degree_2)

    #print(req_result_array)
    return req_result_array

def comparaison(result_array, req_result_array):
    result = []
    for i in range(len(result_array)):
        dist = np.sqrt(np.sum(np.square(np.array(result_array[i][0:4]) - np.array(req_result_array))))
        result.append(dist)
    print(result)

    i = 0
    result_ave = []
    while (i + 3) <= len(result):
        ave = (result[i] + result[i + 1] + result[i + 2])/3
        result_ave.append(ave)
        i = i + 3

    print(result_ave.index(min(result_ave)))

if __name__ == "__main__":
    result_array = generate_characteristic_array()
    req_res_array = generate_request_charac_array('9_1.png')
    comparaison(result_array, req_res_array)
