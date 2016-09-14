from __future__ import division
import numpy as np
from vtk_rw import read_vtk
from sklearn import linear_model
import scipy.stats as stats
import pickle
import itertools
import pandas as pd
from graphs import graph_from_mesh
import gdist
from joblib import Parallel, delayed

'''
Make summary csv of model selection
'''

def load_pickle(pkl_file):
    pkl_in = open(pkl_file, 'r')
    pkl_dict = pickle.load(pkl_in)
    pkl_in.close()
    return pkl_dict


models = range(1048575)

model_file = '/scr/ilz3/myelinconnect/new_groupavg/model/linear_combination/t1avg/smooth_1.5/model_comparison/model_%s.pkl'
df = pd.DataFrame(columns=["maps", 
                           "Pearson's r", 
                           "R squared", 
                           "BIC", 
                           "Residual SD"], 
                  index=models)

for m in models: 
    print m
    
    model_dict = load_pickle(model_file%str(m))
    df["maps"][m] = model_dict["maps"] 
    df["Pearson's r"][m] = model_dict["corr"] 
    df["R squared"][m] = model_dict["rsquared"] 
    df["BIC"][m] = model_dict["bic"] 
    df["Residual SD"][m] = model_dict["res"] 
    
df.to_csv('/scr/ilz3/myelinconnect/new_groupavg/model/linear_combination/t1avg/smooth_1.5/model_comparison_20maps.csv')





    

