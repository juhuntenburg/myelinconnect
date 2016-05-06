import numpy as np
import itertools
from sklearn import linear_model
import pandas as pd


## function to calculate BIC
def BIC(residuals, n_free):
    n = residuals.shape[0]
    chisquare = np.sum(residuals**2)
    bic = np.log(chisquare/n) + np.log(n) * n_free
    return bic


# list all possible combinations of maps
combinations = []
for i in range(len(maps)):
    element = [list(x) for x in itertools.combinations(range(10), i+1)]
    combinations.extend(element)
    

df = pd.DataFrame(columns=["Maps", "Pearson's r", "R squared", "BIC"], 
                  index=range(len(combinations)))


for c in range(len(combinations)):
    
    maps=combinations[c]
    
    clf = linear_model.LinearRegression()
    clf.fit(masked_embed[:,maps], masked_t1)

    modelled_fit = clf.predict(masked_embed[:,maps])
    residuals = masked_t1 - clf.predict(masked_embed[:,maps])
    
    df["Maps"][c] = tuple(maps)
    df["Pearson's r"][c] = stats.pearsonr(modelled_fit, masked_t1)
    df["R squared"][c] = clf.score(masked_embed[:,maps], masked_t1)
    df["BIC"][c] = BIC(residuals, len(maps))
    

    
    
