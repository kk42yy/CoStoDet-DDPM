import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from options_eval import parser
import util_eval as util

opts = parser.parse_args()

_, _, prediction, target = util.load_samples(opts)

uncertainties = {
	'epistemic_reg': None,
	'epistemic_cls': util.epistemic_cls(prediction, dim_samples=3, dim_classes=2),
	'aleatoric_cls': util.aleatoric_cls(prediction, dim_samples=3, dim_classes=2),
	'entropy_cls': util.entropy_cls(prediction, dim_samples=3, dim_classes=2)
}
uncertainty = uncertainties[opts.uncertainty_type]

prediction = np.argmax(prediction.mean(axis=3),axis=2)

### plot uncertainties for true positives and false positives in classification task ###

fig, axes = plt.subplots(1,prediction.shape[0])
fig.suptitle('Classification')
for i,(ax,y,t,u) in enumerate(zip(axes,prediction,target,uncertainty)):
	
	anticipating = (y == util.classes['inside_horizon'])
	tp = u[(t == y) & anticipating]
	fp = u[(t != y) & anticipating]

	ax.scatter(np.random.uniform(low=.1,high=.9,size=fp.shape)+1,fp,s=.01,c=util.colors[i])
	ax.scatter(np.random.uniform(low=.1,high=.9,size=tp.shape),tp,s=.01,c=util.colors[i])
	ax.scatter(1.5,np.median(fp),c='black')
	ax.scatter(.5,np.median(tp),c='black')
	
	ax.set_title(util.instruments[i])
	ax.set_xticks([.5,1.5])
	ax.set_xticklabels(['TP','FP'],size=9)
	if i != 0:
		ax.set_yticks([])
		ax.set_yticklabels([])
	if i == 0:
		if opts.uncertainty_type == 'epistemic_cls':
			ax.set_ylabel('$\sigma_{epistemic}$',size=15)
		elif opts.uncertainty_type == 'aleatoric_cls':
			ax.set_ylabel('$\sigma_{aleatoric}$',size=15)
		elif opts.uncertainty_type == 'entropy_cls':
			ax.set_ylabel('$\sigma_{entropy}$',size=15)
plt.show()
plt.close()

### plot uncertainty for anticipating scissors depending on presence of the clipper ###

prediction_scissors = prediction[1]
uncertainty_scissors = uncertainty[1]

anticipating_scissors = (prediction_scissors == util.classes['inside_horizon'])
clipper_present = (target[2] == util.classes['instrument_present'])

uncert_no_clipper = uncertainty_scissors[anticipating_scissors & ~clipper_present]
uncert_clipper = uncertainty_scissors[anticipating_scissors & clipper_present]

plt.title('Classification (anticipating scissors)')

plt.scatter(np.random.uniform(low=.1,high=.9,size=uncert_no_clipper.shape),uncert_no_clipper,s=.1,c=util.colors[1])
plt.scatter(np.random.uniform(low=.1,high=.9,size=uncert_clipper.shape)+1,uncert_clipper,s=.1,c=util.colors[1])

plt.scatter(0.5,np.median(uncert_no_clipper),c='black')
plt.scatter(1.5,np.median(uncert_clipper),c='black')

plt.xticks([.5,1.5],['No\nclipper\npresent', 'Clipper\npresent'])
if opts.uncertainty_type == 'epistemic_cls':
	plt.ylabel('$\sigma_{epistemic}$',size=15)
elif opts.uncertainty_type == 'aleatoric_cls':
	plt.ylabel('$\sigma_{aleatoric}$',size=15)
elif opts.uncertainty_type == 'entropy_cls':
	plt.ylabel('$\sigma_{entropy}$',size=15)

plt.show()
plt.close()