from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import sys


def dbscan(data, num):
    labels = [1,2,3,4,5]
    min_samples = [5,10,15]
    for m in min_samples:
        path = './resultados/dbscan/dbscan'
        dbscan = DBSCAN(eps=0.25, min_samples=m).fit(data)
        scatter = plt.scatter(data[:,0], data[:,1], c=dbscan.labels_, cmap='Accent')
        plt.legend(handles=scatter.legend_elements()[0], labels = labels)
        plt.title("dbscan")
        path += '_eps_0.25_min_samples_' + str(m) + '_dataset_' + str(num) + '.png'
        plt.savefig(path)
    for m in min_samples:
        path = './resultados/dbscan/dbscan'
        dbscan = DBSCAN(eps=0.35, min_samples=m).fit(data)
        scatter = plt.scatter(data[:,0], data[:,1], c=dbscan.labels_, cmap='Accent')
        plt.legend(handles=scatter.legend_elements()[0], labels = labels)
        plt.title("dbscan")
        path += '_eps_0.35_min_samples_' + str(m) + '_dataset_' + str(num) + '.png'
        plt.savefig(path)
    for m in min_samples:
        path = './resultados/dbscan/dbscan'
        dbscan = DBSCAN(eps=0.5, min_samples=m).fit(data)
        scatter = plt.scatter(data[:,0], data[:,1], c=dbscan.labels_, cmap='Accent')
        plt.legend(handles=scatter.legend_elements()[0], labels = labels)
        plt.title("dbscan")
        path += '_eps_0.5_min_samples_' + str(m) + '_dataset_' + str(num) + '.png'
        plt.savefig(path)

def main(path, n):
#recibe de consola el path del dataset 
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    dbscan(data, n)

if __name__ == "__main__":
    path = sys.argv[1]
    n = sys.argv[-1]
    main(path, n)
