# @kxyang 2024.7.5
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

join = os.path.join
def eval_acc(gtp, predp):
    gt = np.array(pd.read_csv(gtp, sep='\t'))[:,1:].flatten()
    pred = np.array(pd.read_csv(predp, sep='\t'))[:,1:].flatten()
    return accuracy_score(gt, pred)

if __name__ == "__main__":
    pass