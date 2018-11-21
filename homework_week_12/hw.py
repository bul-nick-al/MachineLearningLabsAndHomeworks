# import itertools
#
# import keras
# import tensorflow as tf
# from keras import Sequential
# from keras.backend import image_data_format
# from keras.datasets import mnist
# # load mnist dataset
# import matplotlib.pyplot as plt
# from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
#
#
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# # ## Normalize Data
# # X_train = tf.keras.utils.normalize(X_train, axis=1)
# # # scales data between 0 and 1
# # X_test = tf.keras.utils.normalize(X_test, axis=1)
# # # scales data between 0 and 1
#
# import pickle
#
#
# img_rows, img_cols = 28, 28
# if image_data_format() == 'channels_first':
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
# #more reshaping
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
# print('X_train shape:', X_train.shape) #X_train shape: (60000, 28, 28, 1)
#
# num_category = 10
# y_train = keras.utils.to_categorical(y_train, num_category)
# y_test = keras.utils.to_categorical(y_test, num_category)
#
# def create_model(num_nodes=(32, 20), num_nodes_dense=123, activations=('tanh', 'tanh', 'tanh'),
#                  loss_and_opt=(keras.losses.categorical_crossentropy, 'sgd')):
#     # model building
#     model = Sequential()
#     model.add(Conv2D(num_nodes[0], kernel_size=(3, 3),
#                      input_shape=(28, 28, 1),
#                      activation=activations[0]))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(num_nodes[1], (3, 3), activation=activations[1]))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Flatten())
#     model.add(Dense(num_nodes_dense, activation=activations[2]))
#     model.add(Dense(num_category, activation='softmax'))
#
#     model.compile(loss=loss_and_opt[0],
#                   optimizer=loss_and_opt[1],
#                   metrics=['accuracy'])
#
#     return model
#
# def get_activations():
#     functions = [keras.activations.tanh, keras.activations.relu, keras.activations.elu]
#     result = []
#     for i in functions:
#         for j in functions:
#             for k in functions:
#                 result.append((i, j, k))
#     return result
#
# def get_n_nodes():
#     result = []
#     nums = [16, 32, 64, 128]
#     for i in nums:
#         for j in nums:
#             result.append((i, j))
#     return result
#
#
# import numpy
# from sklearn.model_selection import GridSearchCV
# from keras.wrappers.scikit_learn import KerasClassifier
# seed = 7
# numpy.random.seed(seed)
#
#
# model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, verbose=0)
# param_grid = dict(num_nodes=get_n_nodes())
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(X_train, y_train)
#
#
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# binary_file = open('result_n_nodes.bin', mode='wb')
# pickle.dump(grid_result, binary_file)
# binary_file.close()
#
#
#
#
#
# model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=128, verbose=0)
# num_nodes_dense = [10, 20, 40, 60, 80, 100]
# param_grid = dict(num_nodes_dense=num_nodes_dense)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(X_train, y_train)
#
#
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# binary_file = open('result_dens.bin', mode='wb')
# pickle.dump(grid_result, binary_file)
# binary_file.close()
#
#
#
#
#
# model = KerasClassifier(build_fn=create_model, epochs=5, verbose=0)
# batch_size = [10, 20, 40, 60, 80, 100]
# param_grid = dict(batch_size=batch_size)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(X_train, y_train)
#
#
#
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))
#
# binary_file = open('result_batch_size.bin', mode='wb')
# pickle.dump(grid_result, binary_file)
# binary_file.close()


result = []
functions =['t','r','e']
for i in functions:
    for j in functions:
        for k in functions:
            print(i+j+k)
