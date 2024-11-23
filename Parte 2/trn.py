# Extreme Deep Learning
import numpy      as np
import pandas as pd
import utility    as ut

def normalize(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    diff = X_max - X_min
    diff[diff == 0] = 1  # Evitar divisiones por cero
    X_normalized = (X - X_min) / diff
    return X_normalized


def mse(y_true, y_pred):
    """
    Calcula el error cuadrático medio (MSE).
    Parámetros:
    - y_true: Matriz de etiquetas reales (numpy array).
    - y_pred: Matriz de etiquetas predichas (numpy array).
    
    Retorno:
    - Error cuadrático medio (float).
    """
    return np.mean((y_true - y_pred) ** 2)



def train_edl(data_file, config_file):
    """
    Entrena un modelo Extreme Deep Learning (EDL).
    """
    # 1. Cargar los datos
    data = pd.read_csv(data_file, header=None)
    X = data.iloc[:, :-2].values  # Características (todas menos las dos últimas columnas)
    Y = data.iloc[:, -2:].values  # Etiquetas binarias (Class_Normal, Class_Attack)

    # Normalizar las características
    X = normalize(X)

    # 2. Cargar la configuración
    config = pd.read_csv(config_file, header=None).squeeze()
    hidden_layer_1 = int(config[0])  # Nodos capa oculta 1
    hidden_layer_2 = int(config[1])  # Nodos capa oculta 2
    penalty = float(config[2])       # Penalización pseudo-inversa
    runs = int(config[3])            # Número de corridas

    # 3. Entrenamiento SAE-ELM
    best_mse = float('inf')
    best_weights = None
    costs = []

    for run in range(runs):
        # Inicializar pesos aleatorios para las capas ocultas
        w1 = np.random.randn(X.shape[1], hidden_layer_1)
        b1 = np.random.randn(hidden_layer_1)

        # Primera capa oculta
        H1 = np.tanh(np.dot(X, w1) + b1)

        # Segunda capa oculta
        w2 = np.random.randn(hidden_layer_1, hidden_layer_2)
        b2 = np.random.randn(hidden_layer_2)
        H2 = np.tanh(np.dot(H1, w2) + b2)

        # Pseudo-inversa ajustada con penalización
        H2_pinv = np.linalg.pinv(H2.T @ H2 + penalty * np.eye(H2.shape[1])) @ H2.T
        w3 = np.dot(H2_pinv, Y)

        # Predicciones y MSE
        Y_pred = np.dot(H2, w3)
        current_mse = mse(Y, Y_pred)
        costs.append(current_mse)

        # Guardar los mejores pesos
        if current_mse < best_mse:
            best_mse = current_mse
            best_weights = (w1, w2, w3)

    # 4. Retornar los pesos y costos
    return best_weights, costs

# Beginning ...
def main():
    
    # Archivos de entrada
    data_file = 'DataTrain.csv'
    config_file = 'config_sae.csv'

    # Entrenar el modelo
    W, Cost = train_edl(data_file, config_file)

    # Guardar resultados
    pd.DataFrame(Cost).to_csv('costo.csv', index=False, header=False)
    np.savez('pesos_sae_elm.npz', w1=W[0], w2=W[1], w3=W[2])

    print("Entrenamiento completado. Costos y pesos guardados.")
    weights = np.load('pesos_sae_elm.npz')
    print("Pesos w1:", weights['w1'].shape)
    print("Pesos w2:", weights['w2'].shape)
    print("Pesos w3:", weights['w3'].shape)


if __name__ == '__main__':
    main()

