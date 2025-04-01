import numpy as np
import matplotlib.pyplot as plt
import os
import util_eval as util
from options_eval import parser

opts = parser.parse_args()

horizon = 5
sample_path = '.../pred'
# sample_path = basedir + '/pred_DDIM16_eMAE_114514_CNN/'
basedir = os.path.split(sample_path)[0]
# savepath = basedir + '/visual_DDIM4_eMAE/'
savepath = sample_path.replace('/pred_', '/visual_')
os.makedirs(savepath, exist_ok=True)

for f in sorted(os.listdir(sample_path)):

	if 'video' in f:
	
		path_pred_reg = os.path.join(sample_path,f)
		path_gt_reg = basedir + '/gt' + f"/{f}"

		with open(path_pred_reg, 'r+') as ff:
			pred_arr = [list(map(float, i.strip().split('\t')[1:])) for i in ff.readlines()[1:]]
		prediction_reg = np.array(pred_arr)
		prediction_regs = prediction_reg.swapaxes(0,1)
		
		with open(path_gt_reg, 'r+') as ff:
			gt_arr = [list(map(float, i.strip().split('\t')[1:])) for i in ff.readlines()[1:]]
		target_reg = np.array(gt_arr)
		target_regs = target_reg.swapaxes(0,1)
		
		for task in ['Tool', 'Phase']:

			if task == 'Tool':
				prediction_reg, target_reg = prediction_regs[:5, :], target_regs[:5, :]
			else:
				prediction_reg, target_reg = prediction_regs[5:, :], target_regs[5:, :]

			plt.rcParams["figure.figsize"] = (40, 20)
			fig, axes = plt.subplots(target_reg.shape[0])
			# fig.suptitle(f)
			for i, (ax, pred, gt) in enumerate(zip(axes, prediction_reg, target_reg)):
				x = np.arange(gt.shape[0], dtype=np.float32)

				mean = pred

				ax.fill_between(x, np.zeros_like(x), np.full_like(x, horizon), where=(gt > 0.1*horizon) & (gt < horizon),
								facecolor='gray', alpha=.2)
				ax.fill_between(x, np.zeros_like(x), np.full_like(x, horizon), where=(gt > 0) & (gt < 0.1*horizon),
								facecolor='red', alpha=.2)
				ax.plot(x, gt, c='black', label='Ground truth')
				ax.plot(x, mean, c=util.colors[i], label='Prediction')

				ax.set_xlabel('Time [sec.]', size=15)
				ax.set_ylabel('{}\n[min.]'.format(util.instruments[i] if task == 'Tool' else util.phases[i]), size=15)

				ax.set_yticks([0, horizon / 2, horizon])
				ax.set_yticklabels([0, horizon / 2, '>' + str(horizon)], size=12)
				ax.set_ylim(-.5, horizon + .5)

				ax.legend(fontsize=15)

			print('saving', savepath + f.split('.')[0] + f'_{task}.png')
			plt.savefig(savepath + '/' + f.split('.')[0] + f'_{task}.png', bbox_inches='tight', dpi=300, pad_inches=0.0)
			plt.close()
