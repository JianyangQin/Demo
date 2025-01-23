import numpy as np
import os

def load_predefined_matrix(city, predefined_matrix):
    if city == 'nyc_mb':
        path = "data/Metro+Bus/"+predefined_matrix
    elif city == 'nyc_mt':
        path = "data/Metro+Taxi/"+predefined_matrix
    elif city == 'nyc_bt':
        path = "data/Bus+Taxi/"+predefined_matrix
        
    predefined_mat = np.load(path)
    
    return predefined_mat