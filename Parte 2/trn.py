# Extreme Deep Learning
from json import tool
import numpy      as np
import pandas as pd
import utility    as ut

# --- Función para inicializar pesos ---




# --- Función para pseudo-inversa ---
def pseudo_inverse(H, Y, penalty):
    H_pinv = np.linalg.pinv(H.T @ H + penalty * np.eye(H.shape[1])) @ H.T
    return H_pinv @ Y

def initialize_weights(L,d):
    r = np.sqrt(6 / (L + d))
    return np.random.rand(L, d) * 2 * r - r

# --- Función para entrenar un ELM ---
def train_elm(X, L, runs, penalty):

    d = X.shape[1]
    best_mse = float('inf')
    best_W1, best_B1, best_W2 = None, None, None  # Variables para almacenar los mejores pesos
    for run in range(runs):
        # Inicialización de pesos
        W1 = initialize_weights(L, d)
        B1 = np.zeros((L, 1))

        # Forward pass (Encoder)
        H = np.tanh(np.dot(W1,X.T) + B1)  # Activaciones de la capa oculta

        #print(H)
        # Ajustar el Decoder (Pseudo-inversa)
        W2 = pseudo_inverse(H.T, Y, penalty)  # Reconstrucción H -> X
        
        # Reconstrucción

        Y_pred = np.dot(H.T, W2)
        mse = ut.mse(Y, Y_pred)

        print(f"Run {run + 1}/{runs} completado. MSE: {mse:.6f}")

        # Guardar los pesos si este run produce un mejor MSE
        if mse < best_mse:
            best_mse = mse
            best_W1, best_B1, best_W2 = W1, B1, W2

    print(f"Mejor MSE obtenido: {best_mse:.6f}")
    return best_W1, best_B1, best_W2

# --- Función para entrenar el SAE usando pesos Decoder-ELM ---
def train_sae_decoder_elm(X, config_sae):
 
    # Parámetros de configuración
    hidden_layer_1 = int(config_sae[0])  # Nodos de la primera capa
    hidden_layer_2 = int(config_sae[1]) 
    penalty = int(config_sae[2]) # Nodos de la segunda capa
    runs = int(config_sae[3])           # Número de runs

    # Entrenamiento del primer ELM
    print("Entrenando ELM 1...")
    W1, B1, W1_decoder = train_elm(X, hidden_layer_1, runs, penalty)  # ELM para la primera capa

    # Nueva entrada para el siguiente ELM (Salida codificada de la primera capa)
    H1 = np.tanh(np.dot(X, W1.T) + B1.T)  # Salida codificada de la primera capa


    # Entrenamiento del segundo ELM
    print("Entrenando ELM 2...")
    W2, B2, W2_decoder = train_elm(H1, hidden_layer_2, runs, penalty)  # ELM para la segunda capa
    H2 = np.tanh(np.dot(H1, W2.T) + B2.T)
    # Construcción del SAE final
    print("Construyendo el SAE final...")
    sae_weights = {
        "W1": W1, "B1": B1,  # Primera capa del SAE
        "W2": W2, "B2": B2,  # Segunda capa del SAE
        "W1_decoder": W1_decoder,  # Decoder de la primera capa
        "W2_decoder": W2_decoder   # Decoder de la segunda capa
    }

    # Retornar pesos del SAE y salida codificada final
    return sae_weights, H1, H2

# --- Función para entrenamiento del clasificador Softmax ---
def train_softmax(H, Y, max_iter, batch_size, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Entrena un clasificador Softmax utilizando mAdam con mini-batches.
    """
    n_samples = H.shape[0]  # Número de muestras
    hidden_units = H.shape[1]  # Número de nodos de la capa oculta
    num_classes = Y.shape[1]  # Número de clases

    # Inicialización de pesos y acumuladores
    weights = np.random.randn(num_classes, hidden_units) * 0.01  # (num_classes, hidden_units)
    velocity = np.zeros_like(weights)  # Para momento (beta1)
    scale = np.zeros_like(weights)  # Para ajuste adaptativo (beta2)

    # Mezclar los datos
    indices = np.random.permutation(n_samples)
    shuffled_H = H[indices, :]
    shuffled_Y = Y[indices, :]

    # Calcular el número de batches
    num_batches = n_samples // batch_size
    loss_history = []
    update_step = 0

    for epoch in range(max_iter):
        epoch_loss = 0

        for batch in range(num_batches):
            update_step += 1
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size

            H_batch = shuffled_H[start_idx:end_idx, :]
            Y_batch = shuffled_Y[start_idx:end_idx, :]

            # Forward pass
            logits = np.dot(weights, H_batch.T)  # (num_classes, batch_size)
            activations = ut.softmax(logits)

            # Calcular la pérdida (entropía cruzada)
            batch_loss = -np.mean(np.sum(Y_batch.T * np.log(activations + 1e-12), axis=0))
            epoch_loss += batch_loss

            # Gradiente
            delta = activations - Y_batch.T
            grad_weights = np.dot(delta, H_batch) / batch_size  # (num_classes, hidden_units)

            # mAdam actualización
            velocity = beta1 * velocity + (1 - beta1) * grad_weights
            scale = beta2 * scale + (1 - beta2) * (grad_weights**2)

            velocity_corr = velocity / (1 - beta1**update_step)
            scale_corr = scale / (1 - beta2**update_step)

            weights -= learning_rate * (velocity_corr / (np.sqrt(scale_corr) + epsilon))

        # Almacenar pérdida promedio por época
        epoch_loss /= num_batches
        loss_history.append(epoch_loss)

        # Mostrar información de progreso
        if epoch % 100 == 0 or epoch == max_iter - 1:
            print(f"Iteración {epoch + 1}/{max_iter}, Pérdida: {epoch_loss:.6f}")

    print("Entrenamiento Softmax completado.")
    return weights, loss_history



# --- Main Script ---
if __name__ == "__main__":
    # Preparar datos
    # ut.prepare_data()  # Crea DataTrain.csv y DataTest.csv si no existen
    # Cargar datos
    data = pd.read_csv("DataTrain.csv", header=None)
    X = data.iloc[:, :-2].values
    Y = data.iloc[:, -2:].values

    # Normalizar X si es necesario
    # X = ut.normalize(X)

    # Cargar configuraciones
    config_sae = pd.read_csv("config_sae.csv", header=None).squeeze()
    config_softmax = pd.read_csv("config_softmax.csv", header=None).squeeze()

    # Entrenar SAE
    sae_weights, H1, H2 = train_sae_decoder_elm(X, config_sae)

    # Guardar los pesos del SAE
    pd.DataFrame(sae_weights["W1"]).to_csv("w1.csv", index=False, header=False)
    pd.DataFrame(sae_weights["W2"]).to_csv("w2.csv", index=False, header=False)
    # Nota: Si necesitas guardar sesgos, incluye líneas para B1 y B2 si no son ceros
    # pd.DataFrame(sae_weights["B1"]).to_csv("b1.csv", index=False, header=False)
    # pd.DataFrame(sae_weights["B2"]).to_csv("b2.csv", index=False, header=False)

    # Usar H1 como entrada para el clasificador
    max_iter = int(config_softmax[0])
    batch_size = int(config_softmax[1])
    lr = float(config_softmax[2])

    # Entrenar Softmax
    softmax_weights, costs = train_softmax(H2, Y, max_iter, batch_size, lr)

    # Guardar los pesos del clasificador Softmax
    pd.DataFrame(softmax_weights).to_csv("w3.csv", index=False, header=False)

    # Guardar costos de entrenamiento
    pd.DataFrame(costs).to_csv("costo.csv", index=False, header=False)

    print("Entrenamiento completado. Pesos y costos guardados.")


