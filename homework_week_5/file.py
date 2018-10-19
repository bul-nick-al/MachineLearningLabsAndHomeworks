import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets

### 1 GENERATE DATA
iris = datasets.load_iris()
### Pay attention that "X" is a (150, 4) shape matrix
### y is a (150,) shape array
X = iris.data
y = iris.target

### 2 CENTER DATA
X_mean = X.mean(axis=0)
X_centered = X - X_mean
X_centered = X_centered.T
### 3 PROJECT DATA
### at first you need to get covariance matrix
### Pay attention that cov_mat should be a (4, 4) shape matrix
cov_mat = X_centered.dot(X_centered.T)
### next step you need to find eigenvalues and eigenvectors of covariance matrix
eig_values, eig_vectors = np.linalg.eig(cov_mat)
eig_values, eig_vectors = eig_values.real, eig_vectors.real
### find out which eigenvectors you should choose based on eigenvalues
sorted_eig_values = np.sort(eig_values)
index_1 = np.where(eig_values == sorted_eig_values[len(sorted_eig_values) - 1])[0][0]
index_2 = np.where(eig_values == sorted_eig_values[len(sorted_eig_values) - 2])[0][0]
print(f"this is our 2D subspace:\n {eig_vectors[:, [index_1,index_2]]}")
### now we can project our data to this 2D subspace
### project original data on chosen eigenvectors
chosen_eig = eig_vectors[:, [index_1, index_2]]

# since if "µ" is an eigenvector, "-µ" is an eigenvector as well and we got a vector opposite to the
# one that sklearn will calculate, out plots will be mirrored along the y axis. To make the plots
# look the same I will just multiply the vector I got by -1, so that it will e the same as the one
# found by sklearn
chosen_eig[:, 1] = chosen_eig[:, 1] * -1
chosen_eig = chosen_eig.T
projected_data = chosen_eig.dot(X_centered)
projected_data = projected_data.T

### now you are able to visualize projected data
### you should get excactly the same picture as in the last lab slide
plt.title("Two most significant PCAs, from scratch")
plt.plot(projected_data[y == 0, 0], projected_data[y == 0, 1], 'bo', label='Setosa')
plt.plot(projected_data[y == 1, 0], projected_data[y == 1, 1], 'go', label='Versicolour')
plt.plot(projected_data[y == 2, 0], projected_data[y == 2, 1], 'ro', label='Virginica')
plt.legend(loc=0)
plt.show()

### 4 RESTORE DATA
### we have a "projected_data" which shape is (2,150)
### and we have a 2D subspace "eig_vectors[:, [index_1, index_2]]" which shape is (4,2)
### how to recieve a restored data with shape (4,150)?
projected_data = projected_data.T
restored_data = chosen_eig.T.dot(projected_data)
############################################
### CONGRATS YOU ARE DONE WITH THE FIRST PART ###
############################################



### 1 GENERATE DATA
### already is done

### 2 CENTER DATA
### already is done

### 3 PROJECT DATA
### "n_components" show how many dimensions should we project our data on 
pca = decomposition.PCA(n_components=2)
### class method "fit" for our centered data
pca.fit(X_centered.T)
### make a projection
X_pca = pca.transform(X_centered.T)
### now we can plot our data and compare with what should we get
plt.title("Two most significant PCAs, sklearn")
plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')
plt.legend(loc=0)
plt.show()
