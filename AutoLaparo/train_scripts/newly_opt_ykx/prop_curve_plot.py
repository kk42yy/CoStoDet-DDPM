# @kxyang 2024.7.5

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

join = os.path.join

basedir = ""
save_dir = join(basedir, 'best_clip_length')
os.makedirs(save_dir, exist_ok=True)

for ID in [78]:
    prop_dir = join(basedir, 'pred_78_1-256', f"case_{ID}")
    save_dir = join(save_dir, 'frame_prop_figure')
    os.makedirs(save_dir, exist_ok=True)

    all_prop = [] # from clip 1 ~ 256
    for i in range(1, 257):
        prop_name = f"video{ID}_{i}-prop.txt"
        cur_prop = pd.read_csv(join(prop_dir, prop_name), sep='\t')
        cur_prop = np.array(cur_prop)[:,1:]
        all_prop.append(cur_prop)
        
    all_prop = np.stack(all_prop) # [256, Length, 7prop+1GT]

    refine = []
    best_clip_length = []

    x = list(range(1,257))
    for frame_idx in tqdm(range(all_prop.shape[1])):
        GT = int(all_prop[0][frame_idx][-1])
        cur_frame = np.array([all_prop[i][frame_idx][GT] for i in range(256)])
        
        if cur_frame.max() >= 0.5:
            refine.append(GT)
        else:
            refine.append((all_prop[-1][frame_idx][:-1]).argmax())
        best_clip_length.append(cur_frame.argmax()+1)
        
        plt.plot(x, cur_frame)
        plt.title(f'Frame_{frame_idx}')
        
        plt.savefig(f'{save_dir}/framd_{frame_idx}.png', bbox_inches='tight')
        plt.close()

    refine = pd.DataFrame(refine,columns=['Phase'])
    refine.to_csv(join(save_dir,'bestrefine__on256-phase.txt'), index=True,index_label='Frame',sep='\t')

    best_clip_length = pd.DataFrame(best_clip_length,columns=['Length'])
    best_clip_length.to_csv(join(save_dir,f'bestclip_length_{ID}.txt'), index=True,index_label='Frame',sep='\t')

    plt.plot(list(range(len(best_clip_length))), best_clip_length)
    plt.title(f'Best_ClipLength')

    plt.savefig(f'{save_dir}/Best_ClipLength_{ID}.png', bbox_inches='tight')
    plt.close()