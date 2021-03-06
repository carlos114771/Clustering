import sys
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time

def Cambiobinario():
    # cambiando el dataset a binario los de dos opciones
    # dataTrain
    dataTrain['animada'].replace(['si', 'no'], [1, 0], inplace=True)
    dataTrain['basada_libro'].replace(['si', 'no'], [1, 0], inplace=True)
    dataTrain['desenlace_feliz'].replace(['si', 'no'], [1, 0], inplace=True)
    dataTrain['saga'].replace(['si', 'no'], [1, 0], inplace=True)
    dataTrain['origen'].replace(['real', 'ficticia'], [1, 0], inplace=True)
    dataTrain['trama'].replace(['simple', 'compleja'], [1, 0], inplace=True)
    # dataTest
    dataTest['animada'].replace(['si', 'no'], [1, 0], inplace=True)
    dataTest['basada_libro'].replace(['si', 'no'], [1, 0], inplace=True)
    dataTest['desenlace_feliz'].replace(['si', 'no'], [1, 0], inplace=True)
    dataTest['saga'].replace(['si', 'no'], [1, 0], inplace=True)
    dataTest['origen'].replace(['real', 'ficticia'], [1, 0], inplace=True)
    dataTest['trama'].replace(['simple', 'compleja'], [1, 0], inplace=True)

    # cambiando el dataset a binario los de tres opciones
    # dataTrain
    dataTrain['clasificacion'].replace(['A', 'B', 'C'], [1, 10, 11], inplace=True)
    dataTrain['duracion'].replace(['30-80', '80-120', '120+'], [1, 10, 11], inplace=True)
    dataTrain['narracion'].replace(['lineal', 'mosaico', 'circulo'], [1, 10, 11], inplace=True)
    dataTrain['tiempo'].replace(['contemporaneo', 'futuro', 'pasado'], [1, 10, 11], inplace=True)
    # dataTest
    dataTest['clasificacion'].replace(['A', 'B', 'C'], [1, 10, 11], inplace=True)
    dataTest['duracion'].replace(['30-80', '80-120', '120+'], [1, 10, 11], inplace=True)
    dataTest['narracion'].replace(['lineal', 'mosaico', 'circulo'], [1, 10, 11], inplace=True)
    dataTest['tiempo'].replace(['contemporaneo', 'futuro', 'pasado'], [1, 10, 11], inplace=True)

def main():
    Cambiobinario()
    #toma los datos del csv que no tengan el target('class')
    x_train = dataTrain.drop(columns=['class'])
    x_test = dataTest.drop(columns=['class'])
    #toma los datos del csv que si tengan el target('class')
    y_train = dataTrain['class'].values
    y_test = dataTest['class'].values

    #un for para probar todos los valores de k=[1,3,5,7,9,11,13,15]
    for i in range(1,16,2):
        #se crea el knn y se empiza a tomar el tiempo
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors = i)

        #se meten los datos para hacer el training
        knn.fit(x_train, y_train)

        #se saca una prediccion para luego hacer los tests
        prediction = knn.predict(x_test)
        end = time.time()

        #imprime los valores pedidos mandadoles el y_test donde esta el target=('class')
        print("K=",i)
        print("Accuracy total:", metrics.accuracy_score(y_test, prediction))
        print("Recall:", metrics.recall_score(y_test, prediction,average='macro'))
        print("Precision:", metrics.precision_score(y_test, prediction,average='macro'))
        print("F1-Score:", metrics.f1_score(y_test, prediction, average='macro'))
        print("Tiempo:", end-start, "\n")



if __name__ == "__main__":
    # capturando el csv
    dataTrain = pd.read_csv(sys.argv[1])
    dataTest = pd.read_csv(sys.argv[2])
    main()
