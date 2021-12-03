from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import sys


def K_means(data, num):
    labels = [1,2,3,4,5]
    for x in range(1, 6):
        path = './resultados/k-means/k-means'
        kmeans = KMeans(n_clusters=x).fit(data)
        scatter = plt.scatter(data[:,0], data[:,1], c=kmeans.labels_, cmap='Accent')
        plt.legend(handles=scatter.legend_elements()[0], labels = labels)
        plt.title("k-means")
        path += '_' + str(x) + '_datos_' + str(num) + '.png'
        plt.savefig(path)


def main(path, n):
#recibe de consola el path del dataset 
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    K_means(data, n)


if __name__ == "__main__":
    path = sys.argv[1]
    n = sys.argv[-1]
    main(path, n)