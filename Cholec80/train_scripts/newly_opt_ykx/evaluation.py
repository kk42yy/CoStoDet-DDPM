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
    predp = "/memory/yangkaixiang/SurgicalSceneUnderstanding/pitfalls_bn/output/predictions/phase/20240820-1719_ablation_nolong_cuhk4040Split_lstm_convnextv2_lr1e-05_bs1_seq64_e2e/predv2_best"
    oripredp = "/memory/yangkaixiang/SurgicalSceneUnderstanding/pitfalls_bn/output/predictions/phase/20240820-1719_ablation_nolong_cuhk4040Split_lstm_convnextv2_lr1e-05_bs1_seq64_e2e/predv2_onlymaxr"
    gtp = "/memory/yangkaixiang/SurgicalSceneUnderstanding/pitfalls_bn/output/predictions/phase/20240723-1608_LongShortV3_maxeav1_cuhk4040Split_lstm_convnextv2_lr1e-05_bs1_seq64_e2e/gt"
    for ID in range(41, 42):
      ori = eval_acc(
        f"{gtp}/video{ID}-phase.txt",
        f"{oripredp}/video{ID}-phase.txt"
      )
      
      new = eval_acc(
        f"{gtp}/video{ID}-phase.txt",
        f"{predp}/video{ID}-phase.txt"
      )
      
      print(ID, ori < new, f"{ori:.3f}", f"{new:.3f}", f"{new-ori:.3f}")