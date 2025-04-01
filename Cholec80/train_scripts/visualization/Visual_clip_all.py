import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

basep = "/memory/yangkaixiang/SurgicalSceneUnderstanding/pitfalls_bn/output/predictions/phase/20240723-1608_LongShortV3_maxeav1_cuhk4040Split_lstm_convnextv2_lr1e-05_bs1_seq64_e2e/predv2_maxca_index"

arr = [0] * 100

for ID in range(41,81):
    pred_pt_p = f"{basep}/video{ID}-index.txt"

    with open(pred_pt_p, 'r+') as f:
        pred_arr = [idx-int(i.split('\t')[1].split('\n')[0]) for idx, i in enumerate(f.readlines()[1:])]
    L = len(pred_arr)
    cnt = Counter(pred_arr)
    # arr += [0] * (max(cnt.keys()) + 1 - len(arr))
    for k, v in cnt.items():
        arr[int(k/L*100)] += v
        # arr[k] += v
    # print(arr)
    break
print(arr)
plt.plot(arr)
plt.savefig(f"{basep}/clip41.png")