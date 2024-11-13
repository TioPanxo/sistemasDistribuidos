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


    