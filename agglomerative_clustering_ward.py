from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import sys


def agglomerative_clustering(data, num):
    path = './plots/agglomerative clustering (ward)/agglomerative'
    agglomerative = AgglomerativeClustering(n_clusters=6)
    agglomerative.fit(datos)
    scatter=plt.scatter(datos['x'], datos['y'], c=agglomerative.labels_, cmap='rainbow')
    labels=[0,1,2,3,4]
    plt.legend(handles=scatter.legend_elements()[0],labels=labels)
    plt.title("Agglomerative")
    plt.show()


def agglomerative_clustering_distance(data, num):
    labels = [1,2,3,4,5]
    distance = [0.25, 0.50, 0.75, 1.0, 1.5]
    for x in distance:
        path = './plots/agglomerative clustering (ward)/agglomerative'
        a = AgglomerativeClustering(linkage = 'ward', n_clusters = None, distance_threshold=x).fit(data)
        scatter = plt.scatter(data[:,0], data[:,0], c=a.labels_, cmap='Accent')
        plt.legend(handles=scatter.legend_elements()[0], labels = labels)
        plt.title("agglomerative clustering ward")
        path += '_distance_' + str(x) + '_dataset_' + str(num) + '.png'
        plt.savefig(path)


def main(path, n):
#recibe de consola el path del dataset 
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    agglomerative_clustering(data, n)
    agglomerative_clustering_distance(data, n)


if __name__ == "__main__":
    path = sys.argv[1]
    n = sys.argv[-1]
    main(path, n)
