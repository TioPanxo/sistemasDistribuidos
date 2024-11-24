# Extreme Deep Learning
from json import tool
import numpy      as np
import pandas as pd
import utility    as ut

# --- Función para inicializar pesos ---
def initialize_weights(input_size, output_size):
    r = np.sqrt(6 / (input_size + output_size))
    return np.random.rand(output_size, input_size) * 2 * r - r

# --- Función para normalizar datos ---
def normalize(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    diff = X_max - X_min
    diff[diff == 0] = 1  # Evitar divisiones por cero
    return (X - X_min) / diff

# --- Función para calcular el MSE ---
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# --- Función para pseudo-inversa ---
def pseudo_inverse(H, Y, penalty=1e-3):
    H_pinv = np.linalg.pinv(H.T @ H + penalty * np.eye(H.shape[1])) @ H.T
    return H_pinv @ Y

# --- Función para entrenar un ELM ---
def train_elm(X, hidden_nodes):
    input_nodes = X.shape[1]
    W1 = initialize_weights(input_nodes, hidden_nodes)  # Pesos de la capa oculta
    B1 = np.zeros((1, hidden_nodes))            # Sesgo de la capa oculta
    
    
    # Forward pass (Encoder)
    H = np.tanh(np.dot(X, W1.T) + B1)  # Activaciones de la capa oculta

    # Ajustar el Decoder (Pseudo-inversa)
    W2 = pseudo_inverse(H, X)  # Reconstrucción H -> X


    # Reconstrucción
    X_reconstructed = np.dot(H, W2)
    loss = mse(X, X_reconstructed)

    print(f"Reconstrucción ELM completada. MSE: {loss:.6f}")
    return W1, B1, W2

# --- Función para entrenar el SAE usando pesos Decoder-ELM ---
def train_sae_decoder_elm(X, config_sae):
    """
    Entrena un SAE usando pesos de Decoder-ELM.
    """
    hidden_layer_1 = int(config_sae[0])  # Nodos de la primera capa
    hidden_layer_2 = int(config_sae[1])  # Nodos de la segunda capa

    # Entrenamiento del primer ELM
    print("Entrenando ELM 1...")
    W1, B1, W2_decoder = train_elm(X, hidden_layer_1)  # ELM para la primera capa

    # Nueva entrada para el siguiente ELM
    H1 = np.tanh(np.dot(X, W1.T) + B1)  # Salida codificada de la primera capa

    # Entrenamiento del segundo ELM
    print("Entrenando ELM 2...")
    W2, B2, W3_decoder = train_elm(H1, hidden_layer_2)  # ELM para la segunda capa

    # Construcción del SAE final
    print("Construyendo el SAE final...")
    sae_weights = {
        "W1": W1, "B1": B1,  # Primera capa del SAE
        "W2": W2, "B2": B2   # Segunda capa del SAE
    }
    

    return sae_weights, H1

# --- Función para entrenamiento del clasificador Softmax ---
def train_softmax(H, Y, max_iter, batch_size, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Entrena un clasificador Softmax usando MiniBatch-mAdam.
    """
    input_size = H.shape[1]
    output_size = Y.shape[1]

    # Inicialización de pesos y bias
    W = initialize_weights(input_size, output_size)
    B = np.zeros((1, output_size))

    # Parámetros de mAdam
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    m_B = np.zeros_like(B)
    v_B = np.zeros_like(B)

    costs = []

    for t in range(1, max_iter + 1):
        # Mezclar datos
        indices = np.random.permutation(H.shape[0])
        H = H[indices]
        Y = Y[indices]

        # Dividir en minibatches
        for batch_start in range(0, H.shape[0], batch_size):
            H_batch = H[batch_start:batch_start + batch_size]
            Y_batch = Y[batch_start:batch_start + batch_size]

            # Forward pass (Softmax)
            Z = np.dot(H_batch, W.T) + B
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Estabilidad numérica
            A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

            # Pérdida (Entropía Cruzada)
            loss = -np.mean(np.sum(Y_batch * np.log(A + 1e-9), axis=1))
            costs.append(loss)

            # Gradiente
            grad_Z = A - Y_batch
            grad_W = np.dot(grad_Z.T, H_batch) / batch_size
            grad_B = np.mean(grad_Z, axis=0, keepdims=True)

            # Actualización con mAdam
            m_W = beta1 * m_W + (1 - beta1) * grad_W
            v_W = beta2 * v_W + (1 - beta2) * (grad_W ** 2)
            m_B = beta1 * m_B + (1 - beta1) * grad_B
            v_B = beta2 * v_B + (1 - beta2) * (grad_B ** 2)

            m_W_hat = m_W / (1 - beta1 ** t)
            v_W_hat = v_W / (1 - beta2 ** t)
            m_B_hat = m_B / (1 - beta1 ** t)
            v_B_hat = v_B / (1 - beta2 ** t)

            W -= lr * m_W_hat / (np.sqrt(v_W_hat) + epsilon)
            B -= lr * m_B_hat / (np.sqrt(v_B_hat) + epsilon)

        # Imprimir pérdida periódicamente
        #if t % 10 == 0 or t == 1:
        #    print(f"Iteracion {t}/{max_iter}, Perdida: {loss:.6f}")

    print(f"Entrenamiento completado. ultima perdida: {loss:.6f}")
    return W, B, costs

# --- Main Script ---
if __name__ == "__main__":
    # Cargar datos
    data = pd.read_csv("DataTrain.csv", header=None)
    X = data.iloc[:, :-2].values
    Y = data.iloc[:, -2:].values

    # Normalizar entradas
    X = normalize(X)
    #print("Primeras filas de los datos normalizados:")
    #print(X[:5])
    
    
    # Cargar configuraciones
    config_sae = pd.read_csv("config_sae.csv", header=None).squeeze()
    config_softmax = pd.read_csv("config_softmax.csv", header=None).squeeze()

    # Entrenar SAE
    sae_weights, H1 = train_sae_decoder_elm(X, config_sae)

    # Usar H1 como entrada para el clasificador
    max_iter = int(config_softmax[0])
    batch_size = int(config_softmax[1])
    lr = float(config_softmax[2])

    # Entrenar Softmax
    softmax_weights, softmax_biases, costs = train_softmax(H1, Y, max_iter, batch_size, lr)

    # Guardar resultados
    pd.DataFrame(costs).to_csv("costo.csv", index=False, header=False)
    np.savez("pesos_sae_softmax.npz", sae_weights=sae_weights, softmax_weights=softmax_weights, softmax_biases=softmax_biases)

    print("Entrenamiento completado.")

    # --- Análisis del Softmax (Añade Aquí) ---
    # Mostrar costos finales
    #print("Costos de las últimas iteraciones:")
    #print(costs[-10:])  # Últimas 10 pérdidas

    # Estadísticas de los pesos del Softmax
    #weights_summary = pd.DataFrame(softmax_weights).describe()
    #print("Estadísticas de los Pesos Finales del Softmax:")
    #print(weights_summary)

    # Gráfica de evolución de los costos
    #import matplotlib.pyplot as plt

    #plt.plot(costs)
    #plt.xlabel("Iteraciones")
    #plt.ylabel("Pérdida (Loss)")
    #plt.title("Evolución de la Pérdida durante el Entrenamiento Softmax")
    #plt.grid()
    #plt.show()

