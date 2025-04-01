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
    basedir = ''
    clip_length = '256'
    # print(eval_acc(
    #   join(basedir, 'gt_78_1-256', f"video78_{clip_length}-phase.txt"),
    #   join(basedir, 'pred_78_1-256', f"video78_{clip_length}-phase.txt")
    # ))
    ID = 43
    print(eval_acc(
      join(basedir, 'gt', f"video{ID}-phase.txt"),
      join(basedir, 'predv2_maxea_woh', f"video{ID}-phase.txt")
    ))