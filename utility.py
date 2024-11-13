import pandas as pd
import os

# Load parameters from config.csv
def config():
    config_params = []
    with open("config.csv","r") as file:
        for line in file:
            if line != "\n":
                config_params.append(float(line.strip()))
    return config_params

def routing():
    os.chdir("sistemasDistribuidos/")

#paso 1 y 2
def import_data(file):

    protocol_map = {'tcp': 1, 'udp': 2, 'icmp': 3}
    attack_map = {'normal': 1, 'neptune': 2 , 'teardrop': 2 , 'smurf': 2, 'pod': 2, 'back': 2, 'land': 2, 'apache2': 2, 'processtable': 2, 'mailbomb': 2, 'udpstorm': 2, 'ipsweep': 3 , 'portsweep': 3 , 'nmap': 3 , 'satan': 3 , 'saint': 3 , 'mscan': 3}
    # sistemasDistribuidos/data/KDDTrain.txt
    txt_path_file = f"{file}.txt"

    # Read the text file into a DataFrame (assuming it’s whitespace-separated or CSV format)
    df = pd.read_csv(txt_path_file, sep=",", header=None)

    # Reemplazar los valores en la columna 2 utilizando el diccionario de protocolos 
    df[1] = df[1].map(protocol_map)

    # Reemplazar los valores en la columna 41 utilizando el diccionario de ataques y tranformamos los elementos que no se encuentren en el diccionario en tipo normal
    df[41] = df[41].map(attack_map).fillna(1)

    # Filas que queremos mapear
    cols_to_encode = [2,3] 

    # Creamos los diccionarios para mapear las palabras a números
    word_to_int = {}
    next_int = 0

    for col_index in cols_to_encode:
        next_int = 0
        for word in df.iloc[:, col_index].unique():
            if word not in word_to_int:
                word_to_int[word] = next_int
                next_int += 1

    # Reemplazamos las columnas con los  numeros correspondientes
    for col_index in cols_to_encode:
        df.iloc[:, col_index] = df.iloc[:, col_index].map(word_to_int)

    # Eliminamos la columna 42
    df.drop(df.columns[42], axis=1, inplace=True)

    # Almacenamos la informacion en Data,csv
    open('Data.csv', 'w').close()  # Clears Data.csv
    df.to_csv('Data.csv', index=False, header=False)

    return None


#paso 3 
def genNewClass():
    # sistemasDistribuidos/data/KDDTrain.txt
    csv_path_file = "Data.csv"

    # Read the text file into a DataFrame (assuming it’s whitespace-separated or CSV format)
    df = pd.read_csv(csv_path_file, sep=",", header=None)

    # Create separate DataFrames for each class
    class1_df = df[df[41] == 1]  # Normal traffic
    class2_df = df[df[41] == 2]  # DOS attack
    class3_df = df[df[41] == 3]  # Probe attack


    # Create copies of the DataFrames before dropping
    class1_df_modified = class1_df.drop(class1_df.columns[41], axis=1)
    class2_df_modified = class2_df.drop(class2_df.columns[41], axis=1)
    class3_df_modified = class3_df.drop(class3_df.columns[41], axis=1)

    # Assign the modified DataFrames back to the original variables
    class1_df = class1_df_modified
    class2_df = class2_df_modified
    class3_df = class3_df_modified

    # guardamos los df en los diferentes archivos csv
    class1_df.to_csv('class1.csv', index=False, header=False, mode='w')
    class2_df.to_csv('class2.csv', index=False, header=False, mode='w')
    class3_df.to_csv('class3.csv', index=False, header=False, mode='w')

    return None

def genDataClass():

    csv_path_file = "Data.csv"

    # Leer el archivo CSV en un DataFrame
    df = pd.read_csv(csv_path_file, sep=",", header=None)

    # Cargar los índices de las muestras
    idx_class1 = pd.read_csv('idx_class1.csv', header=None).values.squeeze() + 1
    idx_class2 = pd.read_csv('idx_class2.csv', header=None).values.squeeze() + 1
    idx_class3 = pd.read_csv('idx_class3.csv', header=None).values.squeeze() + 1

    # Filtrar el DataFrame utilizando `.iloc` con la lista de índices
    idx_class1_df = df.iloc[idx_class1]
    idx_class2_df = df.iloc[idx_class2]
    idx_class3_df = df.iloc[idx_class3]

    # Concatenar los DataFrames
    data_class_df = pd.concat([idx_class1_df, idx_class2_df], ignore_index=True)
    data_class_df = pd.concat([data_class_df, idx_class3_df], ignore_index=True)

    # Guardar el DataFrame resultante en un archivo CSV
    data_class_df.to_csv('DataClass.csv', index=False, header=False, mode='w')

    return None




    