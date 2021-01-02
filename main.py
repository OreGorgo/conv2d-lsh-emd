# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1

import numpy as np
import struct
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
from math import sqrt

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D


def read_hyperparameters():
    #reads all the hyperparameters for one experiment

    filters_number = []

    layers = int(input("Give number of Convolution layers : "))

    print("Give number of convolutional filters every layer.One integer per line for every layer.")
    for i in range(layers):
        filters_number.append(int(input()))

    print("Give filter size.Input format : dim1 dim2 ")
    dim1, dim2 = map(int, input().split())
    filterSize = (dim1, dim2)

    epochs = int(input("Give number of epochs for training : "))

    batch_size = int(input("Give batch size : "))

    return layers, filters_number, filterSize, epochs, batch_size  # might need to change filterSize to filters_size


def encoder(image, filtersPerLayer, kernel_size, new_dim):
    conv = image
    counter = 0

    # encoder
    #Add conv layers in a loop
    for layer_size in filtersPerLayer:
        if counter < 2:
            conv = tf.keras.layers.Conv2D(filters=layer_size, kernel_size=kernel_size, activation='relu',
                                          padding='same', strides=2)(conv)  # conv layer with relu activation
        else:
            conv = tf.keras.layers.Conv2D(filters=layer_size, kernel_size=kernel_size, activation='relu',
                                          padding='valid', strides=2)(conv)  # conv layer with relu activation
        conv = tf.keras.layers.BatchNormalization()(conv)  # batch normalization
        counter += 1

        print(conv.shape)

    conv = tf.keras.layers.Flatten()(conv)

    print(conv.shape)
    flat = conv.shape[-1]

    conv = tf.keras.layers.Dense(new_dim, activation='relu')(conv)
    #conv = tf.keras.layers.Embedding(2, 10, input_length=784)(conv)


    return conv,flat

def decoder(conv, filtersPerLayer, kernel_size, flat):
    counter = len(filtersPerLayer)
    # decoder

    dim = int(sqrt(flat / filtersPerLayer[-1]))

    print(dim)

    print(flat)
    print(filtersPerLayer[-1])

    conv = tf.keras.layers.Dense(flat, activation='relu')(conv)
    conv = tf.keras.layers.Reshape((dim, dim, filtersPerLayer[-1]))(conv)

    filtersPerLayer[-1] = 1
    filtersPerLayer.sort()

    activation  = 'relu'

    #Add conv layers in a loop
    for layer_size in reversed(filtersPerLayer):
        counter -= 1

        if counter >= 2:
            conv = tf.keras.layers.Conv2DTranspose(filters=layer_size, kernel_size=kernel_size, activation=activation,
                                                   padding='valid', strides=2)(conv)  # conv layer with relu activation
        else:
            if layer_size == 1:
                activation = 'sigmoid'
                conv = tf.keras.layers.Conv2DTranspose(filters=layer_size, kernel_size=kernel_size, activation=activation,
                                                   padding='same', strides=2)(conv)  # conv layer with relu activation
            else:
                conv = tf.keras.layers.Conv2DTranspose(filters=layer_size, kernel_size=kernel_size, activation=activation,
                                                       padding='same', strides=2)(conv)  # conv layer with relu activation
                conv = tf.keras.layers.BatchNormalization()(conv)  # batch normalization

    #image = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='sigmoid', padding='valid')(conv) #output layer

    return conv


def plot_error(autoencoder,  autoencoder_train, epochs, hyper_list, losses):
    #plot loss vs epochs for both train and validation sets
    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'ro', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    #plot erros in training for every hyperparameter
    for i in range(len(hyper_list[0])):
        toPlot = []
        for tup in hyper_list:
            toPlot.append(tup[i])
        plt.figure()
        plt.plot(toPlot, losses, 'b', label='Validation loss')
        plt.plot(toPlot, losses, 'ro', label='Validation loss')
        if i==0:
            plt.title('Convolution layers')
        elif i==1:
            plt.title('Filters at last convolution layer')
        elif i==2:
            plt.title('Kernel dimension')
        elif i==3:
            plt.title('Number of training epochs')
        elif i==4:
            plt.title('Batch size')
        plt.legend()
        plt.show()



if __name__ == '__main__':
    train_in = 'train.dat'
    test_in = 'test.dat'
    train_out = 'new_train.dat'
    test_out = 'new_test.dat'

    new_dim = 10

    # read input from command line
    for it, arg in enumerate(sys.argv):
        if arg == '-d':
            train_in = sys.argv[it + 1]
        elif arg == '-q':
            test_in = sys.argv[it + 1]
        elif arg == '-od':
            train_out = sys.argv[it + 1]
        elif arg == '-oq':
            test_out = sys.argv[it + 1]
        elif arg == '-dim':
            new_dim = int(sys.argv[it + 1])

    f = open(train_in, "rb")

    #unpack the data and switch byte order
    magic = struct.unpack('>I', f.read(4))[0]
    size = struct.unpack('>I', f.read(4))[0]
    rows = struct.unpack('>I', f.read(4))[0]
    cols = struct.unpack('>I', f.read(4))[0]
    pixels = list(f.read())

    f.close()

    #image reshaping and pixel value normalization
    images = np.array(pixels)
    images = images/np.max(images)
    images = images.reshape(-1, 28, 28, 1)

    print(images.shape)

    action = 1
    hyper_list = []
    losses = []

    while action < 3:

        conv_layers, filtersPerLayer, kernel_size, epochs, batch_size = read_hyperparameters()
        hyper_list.append((conv_layers, filtersPerLayer[-1], kernel_size[0], epochs, batch_size))

        input_img = tf.keras.Input(shape=(28, 28, 1))

        # build the autoencoder model

        conv,flat = encoder(input_img, filtersPerLayer, kernel_size, new_dim)

        autoencoder = tf.keras.Model(input_img, decoder(encoder(input_img, filtersPerLayer, kernel_size, new_dim)[0], filtersPerLayer, kernel_size, flat))
        autoencoder.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.RMSprop())
        #autoencoder.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())

        print(autoencoder.summary())

        train_X, valid_X, train_ground, valid_ground  = train_test_split(images, images, test_size=0.2, random_state=13)


        autoencoder_train = autoencoder.fit(train_X, train_X, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_X))
        #autoencoder_train contains info useful for plotting error

        losses.append(autoencoder_train.history['val_loss'][-1])


        print('Training complete. Choose an action.')
        print('1 - Repeat the training with different hyperparameters')
        print('2 - See plots relevant to this training procedure')
        print('3 - Save the current trained model')

        action = int(input())

        if action >= 3:
            break

        elif action == 2:
            plot_error(autoencoder, autoencoder_train, epochs, hyper_list, losses)

        print('Choose an action.')
        print('1,2 - Repeat the training with different hyperparameters')
        print('3 - Save the current trained model')

        action = int(input())

    if(conv_layers <3):
        encoder_length = (len(autoencoder.layers) // 2) + 1
    else:
        encoder_length = (len(autoencoder.layers) // 2) + 2
    # create a new model from the encoder part of autoencoder
    model = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[encoder_length - 1].output)

    print(model.summary())

    #use the encoder model on every image
    pred = model.predict(images, verbose=1, use_multiprocessing=True)
    pred = pred.flatten()

    print(np.max(pred))

    pred = pred / np.max(pred)


    lower, upper = 0, 65535  #normalize from 0 to (2^16)-1

    norm = [int(upper * x) for x in pred]


    magicnum = struct.pack('>I', 1312)
    size = struct.pack('>I', size)
    rows = struct.pack('>I', 1)
    cols = struct.pack('>I', new_dim)    #new dimension !!!!


    f = open(train_out, "wb")

    f.write(magicnum)
    f.write(size)
    f.write(rows)
    f.write(cols)

    for item in norm:
        f.write(struct.pack('>H', item))  # array of new image values

    f.close()




    #read the query file and create the new one
    f = open(test_in, "rb")

    # unpack the data and switch byte order
    magic = struct.unpack('>I', f.read(4))[0]
    size = struct.unpack('>I', f.read(4))[0]
    rows = struct.unpack('>I', f.read(4))[0]
    cols = struct.unpack('>I', f.read(4))[0]
    pixels = list(f.read())

    f.close()

    # image reshaping and pixel value normalization
    images = np.array(pixels)
    images = images / np.max(images)
    images = images.reshape(-1, 28, 28, 1)

    # use the encoder model on every image
    pred = model.predict(images, verbose=1, use_multiprocessing=True)
    pred = pred.flatten()
    pred = pred / np.max(pred)

    lower, upper = 0, 65535  # normalize from 0 to (2^16)-1
    norm = [int(upper * x) for x in pred]

    #create the contents of the new file
    magicnum = struct.pack('>I', 6666)
    size = struct.pack('>I', size)
    rows = struct.pack('>I', 1)
    cols = struct.pack('>I', new_dim)

    f = open(test_out, "wb")

    f.write(magicnum)
    f.write(size)
    f.write(rows)
    f.write(cols)

    for item in norm:
        f.write(struct.pack('>H', item))  # array of new image values

    f.close()
