from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils
import tsne
import os
import pandas as pd



def main():
    # Loading dataset
    X_train = utils.load_HDR_data()

    # Extracting data
    country_names = X_train['country_names']
    indicator_names = X_train['indicator_names']
    indicator_descriptions = X_train['indicator_descriptions']
    X_train = X_train['X']

    # Prepating scaler
    scaler = preprocessing.StandardScaler().fit(X_train)
    # X_scaled = preprocessing.scale(X_train) - used to test if yielding the same value

    # Data normalisation
    X_norm = scaler.transform(X_train)

    # Preparing data for 2D visualisation using t-SNE
    X_bidim = tsne.tsne(X = X_norm, perplexity = 33)

    sse = []    # Inertia for calcualting final graph
    for number_cl in range(2, 11):
        ci = []    # Closest instance names
        print(number_cl)    # Number of clusters used [2-10]
        kmeans = KMeans(n_clusters=number_cl).fit(X_norm)    # K-Means algorithm runs with number_cl clusters on normalised data X_norm
        sse.append([number_cl, kmeans.inertia_])
        X = kmeans.predict(X_norm)    # KMeans prediction on normalised data
        cl_instance, cl_indice = utils.find_closest_instances_to_kmeans(X_norm, kmeans)    # finding the closest country to the cluster centre

        for i in cl_indice:
            ci.append(country_names[i])    # Append labels of the closest countries

        # added parameters for beter plots
        utils.show_annotated_clustering(X_bidim, X, country_names, number_cl, ci)



if __name__ == '__main__':
    # Check for data folder
    if not (os.path.exists('./data')):
        os.makedirs('./data')
    main()
