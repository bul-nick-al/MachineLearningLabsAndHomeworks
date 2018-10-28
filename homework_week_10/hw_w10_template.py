import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib import markers
import itertools


# finds indices of closest clusters to be merged on next iteration
# clusters_matrix - clusters matrix
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# returns cluster indices and distance between them
def find_clusters_to_merge(clusters_matrix, distance_col, cluster_col):
    # finding the smallest distance
    min_distance = min(clusters_matrix[:, distance_col])
    # getting the ids of the clusters with the minimal distance
    c1_id = np.where(clusters_matrix[:, distance_col] == min_distance)[0][0]
    c2_id = clusters_matrix[c1_id, cluster_col]
    return c1_id, c2_id, min_distance


# performs merge of clusters with indices c1_index, c2_index
# updates single-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, good implementation doesn't need it in this method
def single_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix):

    # casting to ints because of some implementation specifics
    c1_index = int(c1_index)
    c2_index = int(c2_index)

    # giving the new merged cluster the id of the first cluster to be merged (c1_index)
    X_matrix[np.where(X_matrix[:, 2] == c2_index), 2] = c1_index

    # c1 and c2 are merged now, so we set the set their mutual distance to infinity (for convenience of calculations)
    clusters_matrix[c1_index, c2_index] = np.inf
    clusters_matrix[c2_index, c1_index] = np.inf

    # finding the closest distances from the new cluster to all other clusters. In fact, the smallest distance to each
    # cluster is just the smallest distance among the distances from c1 and c2 to each cluster.
    closest_distances = np.minimum(clusters_matrix[:, c1_index], clusters_matrix[:, c2_index])

    # putting the new distances to the clusters table
    clusters_matrix[c1_index, :distance_col] = closest_distances
    clusters_matrix[:, c1_index] = closest_distances

    # after mering the index of the second cluster is not used anymore, so we set the corresponding distances to inf
    clusters_matrix[c2_index, :distance_col] = np.inf
    clusters_matrix[:, c2_index] = np.inf

    # recomputing the closest clusters ans the distances to them
    clusters_matrix[:, cluster_col] = np.argmin(clusters_matrix[:, :distance_col-1], axis=1)
    clusters_matrix[:, distance_col] = np.amin(clusters_matrix[:, :distance_col-1], axis=1)

# performs merge of clusters with indices c1_index, c2_index
# updates complete-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, good implementation doesn't need it in this method
def complete_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix):

    # casting to ints because of some implementation specifics
    c1_index = int(c1_index)
    c2_index = int(c2_index)

    # giving the new merged cluster the id of the first cluster to be merged (c1_index)
    X_matrix[np.where(X_matrix[:, 2] == c2_index), 2] = c1_index

    # c1 and c2 are merged now, so we set the set their mutual distance to infinity (for convenience of calculations)
    clusters_matrix[c1_index, int(c2_index)] = np.inf
    clusters_matrix[c2_index, c1_index] = np.inf

    # finding the furthest distances from the new cluster to all other clusters. In fact, the greatest distance to each
    # cluster is just the greatest distance among the distances from c1 and c2 to each cluster.
    furthest_distances = np.maximum(clusters_matrix[:, c1_index], clusters_matrix[:, c2_index])

    # putting the new distances to the clusters table
    clusters_matrix[c1_index, :distance_col] = furthest_distances
    clusters_matrix[:, c1_index] = furthest_distances

    # after mering the index of the second cluster is not used anymore, so we set the corresponding distances to inf
    clusters_matrix[c2_index, :distance_col] = np.inf
    clusters_matrix[:, c2_index] = np.inf

    # recomputing the closest clusters ans the distances to them
    clusters_matrix[:, cluster_col] = np.argmin(clusters_matrix[:, :distance_col - 1], axis=1)
    clusters_matrix[:, distance_col] = np.amin(clusters_matrix[:, :distance_col - 1], axis=1)


# performs merge of clusters with indices c1_index, c2_index
# updates average-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, use it for this method
def average_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix):
    # casting to ints because of some implementation specifics
    c1_index = int(c1_index)
    c2_index = int(c2_index)

    # calculating the number of the samples for both classes. We will need them for calc. the average distance
    c1_amount = len(np.where(X_matrix[:, 2] == c1_index)[0])
    c2_amount = len(np.where(X_matrix[:, 2] == c2_index)[0])
    # giving the new merged cluster the id of the first cluster to be merged (c1_index)
    X_matrix[np.where(X_matrix[:, 2] == c2_index), 2] = c1_index

    # c1 and c2 are merged now, so we set the set their mutual distance to infinity (for convenience of calculations)
    clusters_matrix[c1_index, int(c2_index)] = np.inf
    clusters_matrix[c2_index, c1_index] = np.inf

    # finding the average distances from the new cluster to all other clusters. In fact, the average distance to each
    # cluster can be computed by the formula described below
    avg_distances = \
        (clusters_matrix[:, c1_index]*c1_amount + clusters_matrix[:, c2_index]*c2_amount)/(c1_amount+c2_amount)

    # putting the new distances to the clusters table
    clusters_matrix[c1_index, :distance_col] = avg_distances
    clusters_matrix[:, c1_index] = avg_distances

    # after mering the index of the second cluster is not used anymore, so we set the corresponding distances to inf
    clusters_matrix[c2_index, :distance_col] = np.inf
    clusters_matrix[:, c2_index] = np.inf

    # recomputing the closest clusters ans the distances to them
    clusters_matrix[:, cluster_col] = np.argmin(clusters_matrix[:, :distance_col - 1], axis=1)
    clusters_matrix[:, distance_col] = np.amin(clusters_matrix[:, :distance_col - 1], axis=1)


# the function which performs bottom-up (agglomerative) clustering
# merge_func - one of the three merge functions above, each with different linkage function
# X_matrix - data itself
# threshold - maximum merge distance, we stop merging if we reached it. if None, merge until there only is one cluster
def bottom_up_clustering(merge_func, X_matrix, distances_matrix, threshold=None):
    num_points = X_matrix.shape[0]

    # take dataset, add and initialize column for cluster membership
    X_data = np.c_[X_matrix, np.arange(0, num_points, 1)]

    # create clusters matrix, initially consisting of all points and pairwise distances
    # with last columns being distance to closest cluster and id of that cluster
    clusters = np.c_[distances_matrix, np.zeros((num_points, 2))]

    # ids of added columns - column with minimal distances, column with closest cluster ids
    dist_col_id = num_points
    clust_col_id = num_points + 1

    # calculate closest clusters and corresponding distances for each cluster
    clusters[:, clust_col_id] = np.argmin(clusters[:, :num_points], axis=1)
    clusters[:, dist_col_id] = np.amin(clusters[:, :num_points], axis=1)

    # array for keeping distances between clusters that we are merging
    merge_distances = np.zeros(num_points - 1)
    # main loop. at each step we are identifying and merging two closest clusters (wrt linkage function)
    for i in range(0, num_points - 1):
        c1_id, c2_id, distance = find_clusters_to_merge(clusters, dist_col_id, clust_col_id)
        # if threshold is set, we don't merge any further if we reached the desired max distance for merging
        if threshold is not None and distance > threshold:
            break
        merge_distances[i] = distance
        merge_func(c1_id, c2_id, X_data, clusters, dist_col_id, clust_col_id, distances_matrix)
        # uncomment when testing
        print("Merging clusters #", c1_id, c2_id)
        # if i%30 == 0:
        #     for k, (marker, color) in zip(range(num_points), itertools.product(markers, colormap)):
        #         plt.scatter(X_data[X_data[:, 2] == k, 0], X_data[X_data[:, 2] == k, 1], color=color, marker=marker, label=k)
        #     plt.show()

    # todo use the plot below to find the optimal threshold to stop merging clusters
    plt.plot(np.arange(0, num_points - 1, 1), merge_distances[:num_points - 1])
    plt.title("Merge distances over iterations")
    plt.xlabel("Iteration #")
    plt.ylabel("Distance")
    plt.show()

    for k, (marker, color) in zip(range(num_points), itertools.product(markers, colormap)):
        plt.scatter(X_data[X_data[:, 2] == k, 0], X_data[X_data[:, 2] == k, 1], color=color, marker=marker)
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()


# importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# creating and populating matrix for storing pairwise distances
# diagonal elements are filled with np.inf to ease further processing
distances = squareform(pdist(X, metric='euclidean'))
np.fill_diagonal(distances, np.inf)

# seting up colors and marker types to use for plotting
markers = markers.MarkerStyle.markers
colormap = plt.cm.Dark2.colors

# performing bottom-up clustering with three different linkage functions
# todo set your own thresholds for each method.
# todo find thresholds by looking at plot titled "Merge distances over iterations" when threshold is set to None
bottom_up_clustering(single_link_merge, X, distances, threshold=10)
bottom_up_clustering(complete_link_merge, X, distances, threshold=65)
bottom_up_clustering(average_link_merge, X, distances, threshold=30)
