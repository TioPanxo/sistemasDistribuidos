#-------------------------------------------------------
# Load Data from File: KDDTrain.txt
#--------------------------------------------------------

import numpy   as np
import utility_etl  as ut
import os
import pandas as pd


# Beginning ...
def main():
    ut.routing()
    ut.config()
    ut.import_data("KDDTrain")
    ut.genNewClass()
    ut.genDataClass()
      
if __name__ == '__main__':   
	main()

