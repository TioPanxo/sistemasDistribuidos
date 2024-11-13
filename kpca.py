# Kernel-PCA by use Gaussian function

import numpy as np
import utility as ut

# Gaussian Kernel
def kernel_gauss(data,sigma):
    #tamaño de la matriz
    N = data.shape[0]
    
    # Calcular la norma cuadrada de cada punto (usamos broadcasting)
    # Norma cuadrada de cada fila
    norms = np.sum(data**2, axis=1).reshape(-1, 1)
    
    # Calcular la matriz de distancias cuadradas utilizando broadcasting
    # Norma cuadrada de la diferencia
    dist_sq = norms + norms.T - 2 * np.dot(data, data.T)
    
    # Calcular el kernel usando la matriz de distancias cuadradas
    K = np.exp(-dist_sq / (2 * sigma**2))

    return K


#Kernel-PCA
def kpca_gauss():
    
    return()


def load_data():
    csv_path_file = "Idx_variable.csv"
    y = np.loadtxt(csv_path_file, delimiter=',', dtype=int)

    csv_path_file = "DataIG.csv"
    x = np.loadtxt(csv_path_file, delimiter=',', dtype=float)

    return(x,y)


# Beginning ...
def main():
    ut.routing() ##Eliminar antes de enviar
    config_params = ut.config()
    sigma = config_params[4]
    top_k = config_params[5]


    X,Y = load_data()

    # Paso 1: Calcular la Matriz del Kernel:
    kernel = kernel_gauss(X,sigma)

    # Paso 2: Centrar en media la  Matriz del Kernel:


    # Paso 3: Calcular los vectores y valores propios del Kernel:
    # Paso 4: Ordenar los  valores propios y vectores  propios en orden descendente:
    # Paso 5: Proyectar la data original sobre los Top-k componentes principales:
		

if __name__ == '__main__':   
	 main()

