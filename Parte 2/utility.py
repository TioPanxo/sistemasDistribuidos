# My Utility : auxiliars functions

import pandas as pd
import numpy  as np


# Función para transformar las clases numéricas a binarias
def transform_classes(df, class_column):

    #Transforma las clases numéricas:
    # Normal (1) -> (1, 0)
    #- Ataques (2) -> (0, 1)
    
    transformed_classes = df[class_column].apply(lambda x: [1, 0] if x == 1 else [0, 1])
    # Dividir las clases en dos columnas
    df['Class_Normal'] = transformed_classes.apply(lambda x: x[0])
    df['Class_Attack'] = transformed_classes.apply(lambda x: x[1])
    return df.drop(columns=[class_column])  # Eliminar la columna original de clase



# Función para seleccionar variables relevantes
def select_relevant_features(df, idx_igain_file):

    idx_igain = pd.read_csv(idx_igain_file, header=None).squeeze() - 1 # Cargar índices relevantes
    return df.iloc[:, idx_igain]  # Filtrar columnas relevantes por índices

def softmax(X):
    exps = np.exp(X - np.max(X, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)

# Preparación de los datos
def prepare_data():
    # Archivos requeridos
    train_file = 'dtrain.csv'  # Archivo original de entrenamiento
    test_file = "dtest.csv"    # Archivo original de prueba
    idx_igain_file = 'idx_igain.csv'  # Archivo de índices de ganancia de información
    train_output = "DataTrain.csv"    # Salida de datos de entrenamiento
    test_output = "DataTest.csv"      # Salida de datos de prueba

    # Cargar datos originales
    df_train = pd.read_csv(train_file, header=None)
    df_test = pd.read_csv(test_file, header=None)

    # Transformar clases numéricas a binarias
    class_column = df_train.columns[-1]  # Última columna es la clase
    df_train = transform_classes(df_train, class_column)
    df_test = transform_classes(df_test, class_column)

    # Seleccionar variables relevantes (sin eliminar las clases)
    relevant_columns_train = select_relevant_features(df_train, idx_igain_file)
    relevant_columns_test = select_relevant_features(df_test, idx_igain_file)

    # Agregar columnas de clases al final del DataFrame
    class_columns_train = df_train[['Class_Normal', 'Class_Attack']]
    class_columns_test = df_test[['Class_Normal', 'Class_Attack']]
    df_train_relevant = pd.concat([relevant_columns_train, class_columns_train], axis=1)
    df_test_relevant = pd.concat([relevant_columns_test, class_columns_test], axis=1)

    # Guardar los datos procesados
    df_train_relevant.to_csv(train_output, index=False, header=False)
    df_test_relevant.to_csv(test_output, index=False, header=False)

    print(f"Datos preparados y guardados en: {train_output} y {test_output}")


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


# --- Cálculo de la matriz de confusión ---
def calculate_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    classes = np.unique(np.concatenate((y_true, y_pred)))
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            confusion_matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    np.savetxt('ConfusionMatrix.csv', confusion_matrix, delimiter=',', fmt='%d')
    return confusion_matrix

# --- Cálculo de métricas de evaluación ---
def calculate_metrics(y_true, y_pred):
    cm = calculate_confusion_matrix(y_true, y_pred)
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": cm
    }

#


