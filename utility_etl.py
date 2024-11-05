import pandas as pd

def import_data(file):

    protocol_map = {'tcp': 1, 'udp': 2, 'icmp': 3}
    attack_map = {'normal': 1, 'neptune': 2 , 'teardrop': 2 , 'smurf': 2, 'pod': 2, 'back': 2, 'land': 2, 'apache2': 2, 'processtable': 2, 'mailbomb': 2, 'udpstorm': 2, 'ipsweep': 3 , 'portsweep': 3 , 'nmap': 3 , 'satan': 3 , 'saint': 3 , 'mscan': 3}

    txt_path_file = f"{file}.txt"

    # Read the text file into a DataFrame (assuming it’s whitespace-separated or CSV format)
    df = pd.read_csv(txt_path_file, sep=",", header=None)

    # Replace values in column 2 using the protocol map
    df[1] = df[1].map(protocol_map)

    # Replace values in column 42 using the attack map, defaulting to 1 for any unmapped value
    df[41] = df[41].map(attack_map)
    df = df[df[41].notna()]  # Keep only rows with valid attack classes

    # Drop columns 3 and 4 (using 0-based indexing, they are columns 2 and 3)
    df.drop(columns=[2, 3], inplace=True)

    # Create separate DataFrames for each class
    class1_df = df[df[41] == 1]  # Normal traffic
    class2_df = df[df[41] == 2]  # DOS attack
    class3_df = df[df[41] == 3]  # Probe attack

    # Save the filtered DataFrames to CSV files
    open('class1.csv', 'w').close()  # Clears class1.csv
    class1_df.to_csv('class1.csv', index=False, header=False)
        
    open('class2.csv', 'w').close()  # Clears class2.csv
    class2_df.to_csv('class2.csv', index=False, header=False)
        
    open('class3.csv', 'w').close()  # Clears class3.csv
    class3_df.to_csv('class3.csv', index=False, header=False)
    # Save the modified DataFrame to a new CSV file
    df.to_csv('Data.csv', index=False, header=False)

    return None






