import numpy as np
import utility as ut
import pandas as pd

def propagate_forward(data, weights):
    """
    Realiza la propagación hacia adelante utilizando el modelo SAE + Softmax.
    """
    # Primera capa
    layer1_output = np.tanh(np.dot(weights['W1'], data.T) + weights['B1'])
    # Segunda capa
    layer2_output = np.tanh(np.dot(weights['W2'], layer1_output) + weights['B2'])
    # Capa softmax
    logits = np.dot(weights['softmax'], layer2_output)
    outputs = ut.softmax(logits.T)
    return outputs

def evaluate_model(predictions, ground_truth):
    """
    Evalúa el modelo calculando métricas como matriz de confusión y F1-scores.
    """
    # Calcular matriz de confusión
    confusion_matrix = ut.calculate_confusion_matrix(ground_truth, predictions)
    # Calcular F1-scores
    f1_scores = calculate_f1_scores(confusion_matrix)
    # Crear diccionario de métricas
    metrics = {
        "confusion_matrix": confusion_matrix,
        "f1_scores": f1_scores  # F1-scores por clase
    }
    return metrics

def calculate_f1_scores(confusion_matrix):
    """
    Calcula F1-scores para las clases positiva y negativa.
    Devuelve un vector (1, 2) con F1-scores para cada clase.
    """
    tp, fp, fn, tn = (
        confusion_matrix[0, 0],
        confusion_matrix[0, 1],
        confusion_matrix[1, 0],
        confusion_matrix[1, 1],
    )
    # Clase positiva
    precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0
    # Clase negativa
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0
    # Retornar vector de F1-scores
    return np.array([f1_score_pos, f1_score_neg])

def main():
    # Leer datos de prueba
    test_data = np.loadtxt('dtest.csv', delimiter=',')
    gain_indices = np.loadtxt('idx_igain.csv', delimiter=',').astype(int)

    # Separar características y etiquetas
    features = test_data[:, :-1]
    labels = test_data[:, -1]
    # Transformar etiquetas a formato binario
    binary_labels = ut.transform_classes(pd.DataFrame(test_data), test_data.shape[1] - 1).iloc[:, -2:].values
    # Seleccionar características relevantes
    relevant_features = features[:, gain_indices - 1]
    # Guardar características procesadas
    np.savetxt('ProcessedTest.csv', relevant_features, delimiter=',', fmt='%f')

    # Cargar pesos entrenados
    weights = {
        'W1': np.loadtxt('w1.csv', delimiter=','),
        'B1': np.zeros((np.loadtxt('w1.csv', delimiter=',').shape[0], 1)),
        'W2': np.loadtxt('w2.csv', delimiter=','),
        'B2': np.zeros((np.loadtxt('w2.csv', delimiter=',').shape[0], 1)),
        'softmax': np.loadtxt('w3.csv', delimiter=',')
    }

    # Propagación hacia adelante
    outputs = propagate_forward(relevant_features, weights)

    # Convertir predicciones y etiquetas a clases
    predicted_classes = np.argmax(outputs, axis=1)
    true_classes = np.argmax(binary_labels, axis=1)

    # Evaluar el modelo
    evaluation = evaluate_model(predicted_classes, true_classes)

    # Guardar la matriz de confusión
    np.savetxt('ConfusionMatrix.csv', evaluation["confusion_matrix"], fmt='%d', delimiter=',')

    # Guardar F1-scores en un vector (1, 2)
    f1_scores = evaluation["f1_scores"]
    np.savetxt('FScores.csv', f1_scores.reshape(1, -1), fmt='%.4f', delimiter=',')

    print("Prueba completada. Resultados guardados en 'ConfusionMatrix.csv' y 'FScores.csv'.")

if __name__ == "__main__":
    main()
