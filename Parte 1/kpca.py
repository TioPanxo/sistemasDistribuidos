# Kernel-PCA by use Gaussian function

import numpy as np
import utility as ut
import pandas as pd

# Gaussian Kernel con centrado en la media de la matriz
def kernel_gauss(data,sigma):
    # Calcular la norma cuadrada de cada punto (usamos broadcasting)
    # Norma cuadrada de cada fila
    norms = np.sum(data**2, axis=1).reshape(-1, 1)
    
    # Calcular la matriz de distancias cuadradas utilizando broadcasting
    # Norma cuadrada de la diferencia
    dist_sq = norms + norms.T - 2 * np.dot(data, data.T)
    
    # Calcular el kernel usando la matriz de distancias cuadradas
    K = np.exp(-dist_sq / (2 * sigma**2))
    return K


def center_kernel_matrix(K):
    # Centrar la matriz de kernel para que tenga media cero.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K_centered


#Kernel-PCA
def kpca_gauss(X,sigma,top_k):
    # Paso 1: Calcular la Matriz del Kernel:
    # Paso 2: Centrar en media la  Matriz del Kernel:
    Kernel = center_kernel_matrix(kernel_gauss(X,sigma))

    # Paso 3: Calcular los vectores y valores propios del Kernel:
    eigenvalues, eigenvectors = np.linalg.eigh(Kernel)
    
    # Paso 4: Ordenar los  valores propios y vectores  propios en orden descendente:
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Paso 5: Proyectar la data original sobre los Top-k componentes principales:
    # Proyectar los datos sobre los primeros k vectores propios
    top_k_eigenvectors = sorted_eigenvectors[:, :top_k]
    X_kpca = Kernel @ top_k_eigenvectors

    return(X_kpca)


def load_data(N):
    csv_path_file = "DataIG.csv"
    x = np.loadtxt(csv_path_file, delimiter=',', dtype=float)
    x = x[:N]

    np.savetxt('Data.csv', x, delimiter=',')

    return(x)


# Beginning ...
def main():
    config_params = ut.config()
    sigma = config_params[4]
    top_k = int(config_params[5])


    X = load_data(3000)

    # Apply Kernel PCA
    X_kpca = kpca_gauss(X, sigma, top_k)
    
    # Save the new data to DataKpca.csv
    np.savetxt('DataKpca.csv', X_kpca, delimiter=',')
    
		

if __name__ == '__main__':   
	 main()

