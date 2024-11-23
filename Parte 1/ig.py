# Information Gain

import numpy as np
import utility as ut
import pandas as pd

# Normalised by use sigmoidal
def norm_data_sigmoidal(data):
    epsilon = 1e-10
    media = np.mean(data)
    desviacion_estandar = np.std(data)
    u = (data-media)/(desviacion_estandar+epsilon)
    norm_data = 1/(1+np.exp(-u))
    return norm_data

# Dispersion entropy
def entropy_disp(data, m, tau, c):   
    #paso 1 Normalizar el conjunto de datos entre 0 y 1
    norm_data = norm_data_sigmoidal(data)

    #paso 2 Crear vectores-embeddingdesde el vector de datos X
    N = len(norm_data)
    M = int(N - (m-1)*tau)
    X_i = np.array([norm_data[i:i + m * tau:tau] for i in range(M)])

    #paso 3 Mapear cada vector-embedding X_i en c-símbolos
    Y_i = np.round(c * X_i + 0.5).astype(int)

    #Paso 4: Convertir el vector de símbolos Y_i en un número correspondiente a un patrón  (símbolo)
    patterns = np.array([1 + np.dot(symbol - 1, c ** np.arange(m)[::-1]) for symbol in Y_i]).astype(int)

    #Paso 5: Contar la frecuencia de ocurrencia del k-patrón
    counts = np.bincount(patterns)[1:]

    #Paso 6: Calcular la probabilidad de cada patrón de dispersión
    p = counts / len(patterns)
    p = p[p > 0]  # Eliminar probabilidades cero para evitar log(0)

    #Paso 7: Calcular la Entropía de Dispersión
    Hy = -np.sum(p * np.log2(p))

    return Hy



#Information gain
def inform_gain(X, Y, config_params):
    #obtenemos los parametros necesarios de config params
    m = int(config_params[0])
    tau = int(config_params[1])
    c = int(config_params[2])
    top_k = int(config_params[3])
    
    #paso 1 Calcular la entropia de dispersion de Y (clases)
    Hy = entropy_disp(Y, m, tau, c)

    # Paso 2: Calcular la Entropía Condicional de Y dado x:
    N = len(X)
    B = int(np.sqrt(N))
    
    # Paso 3: Calcular la Ganancia de Información
    IG_vector = []

    for j in range(X.shape[1]):
        #crear los bins para las x categorias
        bin_edges = np.linspace(np.min(X[:, j]), np.max(X[:, j]), B + 1)
        X_bin = np.digitize(X[:, j], bin_edges) - 1
        
        # Calcular H(Y|x) para la variable X[:, j]
        Hyx = 0
        for i in range(B):
            # Seleccionar muestras que caen en el bin 'b'
            b_idx = (X_bin == i)
            Y_bin = Y[b_idx]
            if len(Y_bin) > 0:
                d_ji = len(Y_bin)
                # Calcular DE de Y en este bin (entropía de dispersión condicional)
                bin_entropy = entropy_disp(Y_bin, m, tau, c)
                Hyx += (d_ji / N) * bin_entropy


        # Paso 3: Calcular la Ganancia de Información para cada bin
        IG_j = Hy - Hyx
        IG_vector.append([j, IG_j])



    # Paso 4: Ordenar en orden decreciente la IG
    IG_vector = sorted(IG_vector, key=lambda x: x[1], reverse=True)

    #Paso 5: Seleccionar las  top-K variables relevantes para crear la nueva base de datos
    selected_index = [feature + 1 for feature,_ in IG_vector[:top_k]]

    selected_features =  X[:, [idx - 1 for idx in selected_index]]

    # Guardar todos los índices ordenados en Idx_variable.csv
    np.savetxt('Idx_variable.csv', selected_index, delimiter=',', fmt='%d')
    # Guardar la base de datos con solo las variables más relevantes en DataIG.csv
    np.savetxt('DataIG.csv', selected_features, delimiter=',', fmt='%f')

    
    

    return()

# Load dataClass 
def load_data():   
    csv_path_file = "DataClass.csv"
    data = np.loadtxt(csv_path_file, delimiter=",", dtype = float)
    return data

# Beginning ...
def main():
    #cargar parametros de configuracion
    config_params = ut.config()

    #cargar data desde dataclass.csv
    data = load_data()

    #obtener las caracteristicas y las etiquetas
    X = data[:, :-1]  # Características (variables explicativas) todas las demas columnas
    Y = data[:, -1]   # Etiquetas (clases) ultima columna 1,2,3

    #ejecutamos la funcion de Ganancia de informacion
    inform_gain(X, Y, config_params)
    
       
if __name__ == '__main__':   
	 main()

