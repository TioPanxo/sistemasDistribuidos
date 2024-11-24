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

    idx_igain = pd.read_csv(idx_igain_file, header=None).squeeze()  # Cargar índices relevantes
    return df.iloc[:, idx_igain]  # Filtrar columnas relevantes por índices



# Preparación de los datos
def prepare_data():
    
    # Archivos requeridos
    train_file = 'dtrain.csv'  # Archivo original de entrenamiento
    test_file = "dtest.csv"    # Archivo original de prueba
    idx_igain_file = 'idx_igain.csv'  # Archivo de índices de ganancia de información
    train_output = "DataTrain.csv"    # Salida de datos de entrenamiento
    test_output = "DataTest.csv"      # Salida de datos de prueba

    #Prepara los datos para entrenamiento y prueba.
    
    # Cargar datos originales
    df_train = pd.read_csv(train_file, header=None)
    df_test = pd.read_csv(test_file, header=None)

    # Transformar clases numéricas a binarias
    class_column = df_train.columns[-1]  # Última columna es la clase
    df_train = transform_classes(df_train, class_column)
    df_test = transform_classes(df_test, class_column)

    # Seleccionar variables relevantes
    df_train_relevant = select_relevant_features(df_train, idx_igain_file)
    df_test_relevant = select_relevant_features(df_test, idx_igain_file)

    # Guardar los datos procesados
    df_train_relevant.to_csv(train_output, index=False, header=False)
    df_test_relevant.to_csv(test_output, index=False, header=False)
    print(f"Datos preparados y guardados en: {train_output} y {test_output}")







# CResidual-Dispersion Entropy
# CResidual-Permutation Entropy
""" def mtx_confusion():
    ...
    return(entr) """
#

#z

#


