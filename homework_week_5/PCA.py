import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


def perform_pca(df):
    """
    :param df:
    :return:
    """

    matrix = df - df.mean(axis=0)

    cov_mat = matrix.transpose().dot(matrix)

    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    sorted_eig_values = np.sort(eigenvalues)
    index_1 = np.where(eigenvalues == sorted_eig_values[len(sorted_eig_values)-1])[0][0]
    index_2 = np.where(eigenvalues == sorted_eig_values[len(sorted_eig_values)-2])[0][0]

    two_eig = eigenvectors[:, [index_1,index_2]]
    two_eig[:,1] = np.flip(two_eig[:,1], axis=0)
    result = matrix.dot(two_eig)
    print()

    # df = df - df.mean(axis=0)
    # n = (df.shape[1])
    # df = df.dot(df.transpose())/n
    # eigenvalues, eigenvectors = np.linalg.eig(df)
    # eigenvalues = eigenvalues.real
    # eigenvectors = eigenvectors.real
    #
    # x= pd.DataFrame([eigenvectors[:,0], eigenvectors[:,1]*-1])
    # x= x.transpose()
    #
    # x = matrix.dot(pd.DataFrame(eigenvectors))

    plt.plot(result[:,0], result[:,1]*-1, 'ro')
    plt.title("MSE for different K levels KNN")
    plt.xlabel('k')
    plt.ylabel('MSE')
    # plt.xscale('log')
    plt.show()
    print(max(eigenvalues))



def load_data():
    """
    loads iris data set
    :return: data frame with loaded data
    """

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y


perform_pca(load_data()[0])
