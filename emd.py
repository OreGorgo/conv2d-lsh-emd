

from ortools.linear_solver import pywraplp
from math import sqrt

import pulp
import numpy as np
import struct
import sys
from sklearn.metrics.pairwise import manhattan_distances


def EMD(image1, image2, w1, w2, centroids1, centroids2):

    #number of clusters in each image
    #will surely be equal
    size1 = image1.shape[0]
    size2 = image2.shape[0]

    #compute the distances between the clusters
    distances = np.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            distances[i][j] = np.linalg.norm(centroids1[i] - centroids2[j])

    #set variables for EMD calculations
    variablesList = []
    for i in range(size1):
        tempList = []
        for j in range(size2):
            tempList.append(pulp.LpVariable("f"+str(i)+" "+str(j), lowBound = 0))

        variablesList.append(tempList)

    problem = pulp.LpProblem("EMD", pulp.LpMinimize)


    # objective function
    constraint = []
    objectiveFunction = []
    for i in  range(size1):
        for j in range(size2):
            objectiveFunction.append(variablesList[i][j] * distances[i][j])

            constraint.append(variablesList[i][j])

    problem += pulp.lpSum(objectiveFunction)

    # constraints
    for i in range(size1):
        constraint1 = [variablesList[i][j] for j in range(size2)]
        problem += pulp.lpSum(constraint1) == w1[i]

    for j in range(size2):
        constraint2 = [variablesList[i][j] for i in range(size1)]
        problem += pulp.lpSum(constraint2) == w2[j]



    # solve
    problem.writeLP("EMD.lp")

    #problem.solve()
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    flow = pulp.value(problem.objective)

    # return flow / tempMin
    return flow


def create_clusters(images, cluster_size, divisions):
    cluster_dataset = []
    dataset_weights = []
    dataset_centroids = []
    # split every image in the dataset into clusters
    for image in images:
        # split every image in the dataset into clusters
        division_counter = 0
        clustered_image = []
        centroids = []
        weights = []
        row_it = 0
        cluster = []

        for i in range(divisions):  # rows of clusters
            for j in range(divisions):  # cols of clusters
                row_it = i * cluster_size
                for r in range(cluster_size):  # rows of image
                    col_it = j * cluster_size
                    for c in range(cluster_size):  # cols of image
                        # print(row_it, col_it)
                        cluster.append(int(image[row_it][col_it]))
                        if (len(cluster) == 1):
                            centroid = np.array([row_it, col_it])
                        col_it += 1
                    row_it += 1
                centroids.append(centroid)
                clustered_image.append(cluster)
                weights.append(sum(cluster))
                cluster = []

        cluster_dataset.append(np.array(clustered_image))
        dataset_weights.append(weights)
        dataset_centroids.append(centroids)

    return cluster_dataset, dataset_weights, dataset_centroids


def knn(images, query, k=10):
    results = []
    query = query.flatten()

    for i, image in enumerate(images):
        image = image.flatten()
        distance = np.linalg.norm((image - query), ord=1)
        results.append( (distance, i) )

    results.sort(key=lambda tup: tup[0])

    return results[0:k]





if __name__ == '__main__':
    train = 'train.dat'
    test = 'test.dat'
    train_labels_file = 'train_labels.dat'
    test_labels_file = 'test_labels.dat'
    output = 'out.txt'


    # read input from command line
    for it, arg in enumerate(sys.argv):
        if arg == '-d':
            train = sys.argv[it + 1]
        elif arg == '-q':
            test = sys.argv[it + 1]
        elif arg == '-l1':
            train_labels = sys.argv[it + 1]
        elif arg == '-l2':
            test_labels = sys.argv[it + 1]
        elif arg == '-o':
            output = sys.argv[it + 1]

    #read dataset images
    f = open(train, "rb")

    # unpack the data and switch byte order
    magic = struct.unpack('>I', f.read(4))[0]
    size = struct.unpack('>I', f.read(4))[0]
    rows = struct.unpack('>I', f.read(4))[0]
    cols = struct.unpack('>I', f.read(4))[0]
    pixels = list(f.read())

    f.close()

    f = open(test, "rb")

    # unpack the data and switch byte order
    magic = struct.unpack('>I', f.read(4))[0]
    query_size = struct.unpack('>I', f.read(4))[0]
    rows = struct.unpack('>I', f.read(4))[0]
    cols = struct.unpack('>I', f.read(4))[0]
    query_pixels = list(f.read())

    f.close()

    # image reshaping and pixel value normalization
    images = np.array(pixels)
    #images = images / np.max(images)
    images = images.reshape(-1, rows, cols)

    query_images = np.array(query_pixels)
    #images = images / np.max(images)
    query_images = query_images.reshape(-1, rows, cols)


    d_range = size
    q_range = query_size

    #ranges for testing
    # d_range = 100
    # q_range = 10

    cluster_size = 7 #clusters will be of nxn pixels
    divisions = rows // cluster_size
    cluster_number = int(divisions**2) #total number of clusters


    #create clusters for every image
    data_clusters, data_weights, data_centroids = create_clusters(images[0:d_range], cluster_size, divisions)

    query_clusters, query_weights, query_centroids = create_clusters(query_images[0:q_range], cluster_size, divisions)

    emd_results = []

    for q in range(q_range):
        tempResult = []
        for d in range(d_range):

            emdDistance = EMD(data_clusters[d], query_clusters[q], data_weights[d], query_weights[q], data_centroids[d], query_centroids[q])
            tempResult.append((emdDistance, d))
            #print( str(emdDistance) )

        tempResult.sort(key=lambda tup: tup[0])
        emd_results.append(tempResult[0:10])

    knn_results = []

    #brute force knn
    for q in range(q_range):
        knn_results.append( knn(images[0:d_range], query_images[q]) )



    #read the labels for both images and queries

    # read data labels
    f = open(train_labels_file, "rb")
    magic = struct.unpack('>I', f.read(4))[0]
    size = struct.unpack('>I', f.read(4))[0]
    labels = list(f.read())

    f.close()
    train_labels = np.array(labels)

    # read query labels
    f = open(test_labels_file, "rb")
    magic = struct.unpack('>I', f.read(4))[0]
    size = struct.unpack('>I', f.read(4))[0]
    labels = list(f.read())

    f.close()
    test_labels = np.array(labels)

    emd_acc_list = []
    knn_acc_list = []
    result_size = len(knn_results[0])

    for i in range(q_range):
        query_label = test_labels[i]
        knn_acc = 0
        emd_acc = 0

        for dist, index in knn_results[i]:
            knn_acc += query_label == train_labels[index]
        knn_acc /= result_size

        for dist, index in emd_results[i]:
            emd_acc += query_label == train_labels[index]
        emd_acc /= result_size

        emd_acc_list.append(emd_acc)
        knn_acc_list.append(knn_acc)

    emd_acc = sum(emd_acc_list) / q_range
    knn_acc = sum(knn_acc_list) / q_range

    print("Average Correct Search Results EMD: ", emd_acc)
    print("Average Correct Search Results MANHATTAN: ", knn_acc)


