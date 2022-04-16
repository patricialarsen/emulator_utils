"""
lhc.py
======
Latin hypercube functions

"""

import pyDOE
import numpy as np

def create_lhc(AllPara, num_evals, output_dir):
    """
    Create a latin hypercube of the parameter range

    Parameters
    ----------
    AllPara: ndarray(ndarray(float))
       parameter options 
    num_evals: int
       number of points to evaluate at
    output_dir: str
       path to put outputs 

    Returns
    -------
    params: ndarray(ndarray(float))
       selected choices of parameters

    """
    num_params = AllPara.shape[0]
    lhd = pyDOE.lhs(num_params, samples=num_evals, criterion=None) # c cm corr m
    idx = (lhd * num_evals).astype(int)

    AllCombinations = np.zeros((num_evals, AllPara.shape[0]))

    for i in range(AllPara.shape[0]):
        AllCombinations[:, i] = AllPara[i][idx[:, i]]
    params = AllCombinations

    np.savetxt(output_dir + '/lhc_'+str(num_evals)+'_'+str(num_params)+'.txt', AllCombinations)
    return params

