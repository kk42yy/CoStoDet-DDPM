import numpy as np
import os

instruments = [
	'Bipolar',
	'Scissors',
	'Clipper',
	'Irrigator',
	'SpecBag'
]

# phases = [
# 	'Preparation',
# 	'CalotTriangleDissection',
# 	'ClippingCutting',
# 	'GallbladderDissection',
# 	'GallbladderPackaging',
# 	'CleaningCoagulation',
# 	'GallbladderRetraction'
# ]

phases = [
	'P1',
	'P2',
	'P3',
	'P4',
	'P5',
	'P6',
	'P7'
]
colors = [
	'teal',
	'darkorange',
	'maroon',
	'forestgreen',
	'indigo',
	'red',
	'blue'
]
uncert_names = {
	'epistemic_reg': 'Epistemic uncertainty (reg.)',
	'epistemic_cls': 'Epistemic uncertainty (cls.)',
	'aleatoric_cls': 'Aleatoric uncertainty (cls.)',
	'entropy_cls': 'Entropy (cls.)'
}
classes = {
	'instrument_present': 0,
	'outside_horizon': 1,
	'inside_horizon': 2
}

epistemic_reg = lambda pred, dim_samples: pred.std(axis=dim_samples)
epistemic_cls = lambda p, dim_samples, dim_classes: np.sqrt(((p - p.mean(axis=dim_samples,keepdims=True))**2).mean(axis=(dim_classes,dim_samples)))
aleatoric_cls = lambda p, dim_samples, dim_classes: np.sqrt(np.mean(p*(1-p), axis=(dim_classes,dim_samples)))
entropy_cls = lambda p, dim_samples, dim_classes: -(p.mean(axis=dim_samples) * np.log(p.mean(axis=dim_samples))).sum(axis=dim_classes)

def load_samples(opts):

	horizon = opts.horizon

	prediction_reg, target_reg = [], []
	prediction_cls, target_cls = [], []
	
	for f in sorted(os.listdir(opts.sample_path)):
		if ('sample_epoch_{}_'.format(opts.epoch) in f) and ('pred_reg' in f):

			path_pred_reg = os.path.join(opts.sample_path,f)
			path_gt_reg = path_pred_reg.replace('pred_reg','gt_reg')
			path_pred_cls = path_pred_reg.replace('pred_reg','pred_cls')
			path_gt_cls = path_pred_reg.replace('pred_reg','gt_cls')

			y_reg = np.load(path_pred_reg)
			t_reg = np.load(path_gt_reg)
			y_cls = np.load(path_pred_cls)
			t_cls = np.load(path_gt_cls)

			prediction_reg.append(y_reg)
			target_reg.append(t_reg)
			prediction_cls.append(y_cls)
			target_cls.append(t_cls)

	prediction_reg = np.concatenate(prediction_reg,axis=1)
	target_reg = np.concatenate(target_reg,axis=1)
	prediction_cls = np.concatenate(prediction_cls,axis=1)
	target_cls = np.concatenate(target_cls,axis=1)
	
	return prediction_reg, target_reg, prediction_cls, target_cls