#-------------------------------------------------------
# Load Data from File: KDDTrain.txt
#--------------------------------------------------------

import numpy   as np
import utility_etl  as ut

# Load parameters from config.csv
def config():
    config_params = []
    with open("config.csv","r") as file:
        for line in file:
            if line != "\n":
                config_params.append(float(line.strip()))
    return config_params


# Beginning ...
def main():
    config()
    ut.import_data("KDDTrain")
   
      
if __name__ == '__main__':   
	main()

