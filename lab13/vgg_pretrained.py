from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from matplotlib import pyplot as plt
import h5py
import cv2
import numpy as np
import os


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def main():
    image_name = 'cat.jpg'
    # read image
    img = cv2.imread("data/{}".format(image_name))
    # convert bgr to rgb
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])

    # plot image
    plt.figure(figsize=(10, 10))
    plt.xticks([]), plt.yticks([])
    plt.imshow(rgb_img)
    plt.show()

    # resize image to network imput shape
    im = cv2.resize(img, (224, 224)).astype(np.int32)

    plt.figure(figsize=(10, 10))
    plt.imshow(im)
    plt.show()

    # substract mean pixel values from each color array
    mean_pixel = im.mean(axis=0).mean(axis=0)
    for c in range(3):
        im[:, :, c] = im[:, :, c] - mean_pixel[c]

    # transform image to feed it to the network
    im = im.transpose((1, 0, 2))
    im = np.expand_dims(im, axis=0)

    ############################################################################
    ######################   DOWNLOAD WEIGHTS   ################################
    ############################################################################
    # download pretrained weights from github page:
    # https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    # and put it in data folder
    # OR just uncomment line below if you have Mac/Ubuntu
    #
    # os.system("wget --load-cookies /tmp/cookies.txt \"https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5\" -O data/vgg16_weights.h5 && rm -rf /tmp/cookies.txt")

    # pretrained model initialization
    model = VGG_16('./data/vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    # make prediction on test image
    out = model.predict(im)

    with open('./data/class_names.txt', 'r') as f:
        x = f.readlines()

    answer_str = ' '.join(x[np.argmax(out)].split(' ')[1:])
    print('Model prediction is: ', answer_str)


if __name__ == '__main__':
    main()
