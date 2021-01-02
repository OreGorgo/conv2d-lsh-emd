

from ortools.linear_solver import pywraplp
from math import sqrt

import pulp
import numpy as np
import struct
import sys


def EMD(feature1, feature2, w1, w2):

    rows = feature1.shape[0]
    cols = feature2.shape[0]

    distances = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            distances[i][j] = np.linalg.norm(feature1[i] - feature2[j])

    # Set variables for EMD calculations
    variablesList = []
    for i in range(rows):
        tempList = []
        for j in range(cols):
            tempList.append(pulp.LpVariable("x"+str(i)+" "+str(j), lowBound = 0))

        variablesList.append(tempList)

    problem = pulp.LpProblem("EMD", pulp.LpMinimize)

    # objective function
    constraint = []
    objectiveFunction = []
    for i in  range(rows):
        for j in range(cols):
            objectiveFunction.append(variablesList[i][j] * distances[i][j])

            constraint.append(variablesList[i][j])

    problem += pulp.lpSum(objectiveFunction)


    tempMin = min(sum(w1), sum(w2))
    problem += pulp.lpSum(constraint) == tempMin

    # constraints
    for i in range(rows):
        constraint1 = [variablesList[i][j] for j in range(cols)]
        problem += pulp.lpSum(constraint1) <= w1[i]

    for j in range(cols):
        constraint2 = [variablesList[i][j] for i in range(rows)]
        problem += pulp.lpSum(constraint2) <= w2[j]



    # solve
    problem.writeLP("EMD.lp")
    problem.solve()
    #problem.solve(pulp.GLPK(msg=0))

    flow = pulp.value(problem.objective)


    return flow / tempMin





if __name__ == '__main__':
    train = 'train.dat'
    test = 'test.dat'
    train_labels = 'train_labels.dat'
    test_labels = 'test_labels.dat'
    output = 'out.txt'

    new_dim = 10

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
    # images = images / np.max(images)
    query_images = query_images.reshape(-1, rows, cols)


    cluster_size = 4 #clusters will be of nxn pixels
    divisions = rows // cluster_size
    cluster_number = int(divisions**2) #total number of clusters

    cluster_dataset = []
    #split every image in the dataset into clusters
    for image in images:
        # split every image in the dataset into clusters
        division_counter = 0
        clustered_image = []
        col_it = 0
        row_it = 0
        cluster = []

        for i in range(divisions):  # rows of clusters
            col_it = 0
            for j in range(divisions):  # cols of clusters
                row_it = i * cluster_size
                for r in range(cluster_size):  # rows of image
                    col_it = j * cluster_size
                    for c in range(cluster_size):  # cols of image
                        # print(row_it, col_it)
                        cluster.append(image[row_it][col_it])
                        col_it += 1
                    row_it += 1

                clustered_image.append(cluster)
                cluster = []

        cluster_dataset.append(clustered_image)



    # xwrizoume se clusters
    # ypologismos centroid (artios -> panw aristera, perittos -> kentro)

    #  [ ([2,5,7,8,...],index of centroid, signature(sum of all picels) ) ]

    # distance metaksy clusters




    #---------------
    feature1 = np.array([[100, 40, 22], [211, 20, 2], [32, 190, 150], [2, 100, 100]])
    feature2 = np.array([[0, 0, 0], [50, 100, 80], [255, 255, 255]])

    w1 = [0.4, 0.3, 0.2, 0.1]
    w2 = [0.5, 0.3, 0.2]

    emdDistance = EMD(feature1, feature2, w1, w2)
    print( str(emdDistance) )


