import numpy as np
import os
from options_eval import parser
import util_eval as util

def load_samples():

	prediction_reg, target_reg = [], []

	for f in sorted(os.listdir(sample_path)):
		if 'video' in f:

			path_pred_reg = os.path.join(sample_path,f)
			path_gt_reg = basedir + '/gt' + f"/{f}"

			with open(path_pred_reg, 'r+') as ff:
				pred_arr = [list(map(float, i.strip().split('\t')[1:])) for i in ff.readlines()[1:]]
			p_reg = np.array(pred_arr)
			p_reg = p_reg.swapaxes(0,1)
			
			with open(path_gt_reg, 'r+') as ff:
				gt_arr = [list(map(float, i.strip().split('\t')[1:])) for i in ff.readlines()[1:]]
			t_reg = np.array(gt_arr)
			t_reg = t_reg.swapaxes(0,1)

			prediction_reg.append(p_reg)
			target_reg.append(t_reg)

	prediction_reg = np.concatenate(prediction_reg,axis=1)
	target_reg = np.concatenate(target_reg,axis=1)

	return prediction_reg, target_reg

def print_scores(instruments,scores,header):
    
    with open(saving_file, 'a+') as f:
        print('\n{}'.format(header))
        f.write('\n{}'.format(header))
        
        for i,s in zip(instruments,scores):
            num_spaces = max([1, len(header) - (len(i) + 5)])
            print('{}:{}{:.3f}'.format(i, ' '*num_spaces, s))
            f.write('\n{}:{}{:.3f}'.format(i, ' '*num_spaces, s))
        
        print('-'*len(header))
        f.write('\n'+'-'*len(header))
        
        num_spaces = max([1, len(header) - 9])
        print('Mean:{}{:.3f}\n'.format(' '*num_spaces, np.mean(scores)))
        f.write('\nMean:{}{:.3f}\n'.format(' '*num_spaces, np.mean(scores)))

horizon = 5
basedir = ''
sample_path = basedir + '/pred'
saving_file = sample_path + '/matrics.txt'
if os.path.isfile(saving_file):
    os.remove(saving_file)

# load predictions (NUM_INSTRUMENTS x NUM_FRAMES x NUM_SAMPLES) and targets (NUM_INSTRUMENTS x NUM_FRAMES)
prediction, target = load_samples()
# use sample mean as an estimate for the predictive expectation
# now prediction and target both have shape (NUM_INSTRUMENTS x NUM_FRAMES)
# prediction = prediction.mean(axis=2)

print(prediction.shape, target.shape)
print('########################### Metrics for Tool ###############################')
wMAE = []
out_MAE = []
in_MAE = []
pMAE = []
eMAE = []
dMAE = []
mMAE = []

for y, t in zip(prediction, target):
	prediction[prediction > horizon] = horizon
	outside_horizon = (t == horizon)
	inside_horizon = (t < horizon) & (t > 0)
	anticipating = (y > horizon*.1) & (y < horizon*.9)

	distant_anticipating = (t > horizon*.9) & (t < horizon)
	e_anticipating = (t < horizon*.1) & (t > 0)
	m_anticipating = (t > horizon*.1) & (t < horizon*.9)


	wMAE_ins = np.mean([
	np.abs(y[outside_horizon]-t[outside_horizon]).mean(),
	np.abs(y[inside_horizon]-t[inside_horizon]).mean()
	])
	out_MAE_ins = np.mean([np.abs(y[outside_horizon]-t[outside_horizon]).mean()])
	in_MAE_ins = np.mean([np.abs(y[inside_horizon]-t[inside_horizon]).mean()])
	pMAE_ins = np.abs(y[anticipating]-t[anticipating]).mean()
	eMAE_ins = np.abs(y[e_anticipating]-t[e_anticipating]).mean()
	dMAE_ins = np.abs(y[distant_anticipating]-t[distant_anticipating]).mean()
	mMAE_ins = np.abs(y[m_anticipating]-t[m_anticipating]).mean()


	wMAE.append(wMAE_ins)
	out_MAE.append(out_MAE_ins)
	in_MAE.append(in_MAE_ins)
	pMAE.append(pMAE_ins)
	eMAE.append(eMAE_ins)
	dMAE.append(dMAE_ins)
	mMAE.append(mMAE_ins)

print_scores(instruments=util.instruments, scores=wMAE, header='== wMAE [min.] ===')
print_scores(instruments=util.instruments, scores=out_MAE, header='== out MAE [min.] ===')
print_scores(instruments=util.instruments, scores=in_MAE, header='== in MAE [min.] ===')
print_scores(instruments=util.instruments, scores=pMAE, header='== pMAE [min.] ===')
print_scores(instruments=util.instruments, scores=eMAE, header='== eMAE [min.] ===')
print_scores(instruments=util.instruments, scores=dMAE, header='== dMAE [min.] ===')
print_scores(instruments=util.instruments, scores=mMAE, header='== mMAE [min.] ===')