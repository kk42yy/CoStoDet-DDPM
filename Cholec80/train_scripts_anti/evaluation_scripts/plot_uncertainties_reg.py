import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from options_eval import parser
import util_eval as util

opts = parser.parse_args()

prediction, target, _, _ = util.load_samples(opts)

uncertainty = util.epistemic_reg(prediction,dim_samples=2)
prediction = prediction.mean(axis=2)
error = np.abs(prediction-target)

horizon = opts.horizon
upper_bound, lower_bound = .9*horizon, .1*horizon

### plot uncertainty as function of prediction error ###

fig, axes = plt.subplots(1,prediction.shape[0])
fig.suptitle('Regression')
y_max = uncertainty.max()

for i,(ax,e,u,y) in enumerate(zip(axes,error,uncertainty,prediction)):
	
	anticipating = (y < upper_bound) & (y > lower_bound)
	e,u = e[anticipating], u[anticipating]
	
	slope, intercept, correlation_factor, _, _ = stats.linregress(e,u)
	x = np.arange(horizon+1)
	y = slope*x + intercept

	ax.scatter(e,u,s=.003,c=util.colors[i])
	ax.plot(x,y,color='black')
	ax.text(x=.5,y=y_max-.1,s='r = {:.2f}'.format(correlation_factor))
	ax.set_ylim((0,y_max))
	ax.set_title(util.instruments[i])
	if i != 0:
		ax.set_yticks([])
	if i == 0:
		ax.set_ylabel('$\sigma_{epistemic}$',size=15)
	if i == 2:
		ax.set_xlabel('$L1$ error [min.]',size=12)

plt.show()
plt.close()

### plot uncertainty for anticipating scissors depending on presence of the clipper ###

prediction_scissors = prediction[1]
uncertainty_scissors = uncertainty[1]

anticipating_scissors = (prediction_scissors < upper_bound) & (prediction_scissors > lower_bound)
clipper_present = (target[2] == 0)

uncert_no_clipper = uncertainty_scissors[anticipating_scissors & ~clipper_present]
uncert_clipper = uncertainty_scissors[anticipating_scissors & clipper_present]

plt.title('Regression (anticipating scissors)')

plt.scatter(np.random.uniform(low=.1,high=.9,size=uncert_no_clipper.shape),uncert_no_clipper,s=.1,c=util.colors[1])
plt.scatter(np.random.uniform(low=.1,high=.9,size=uncert_clipper.shape)+1,uncert_clipper,s=.1,c=util.colors[1])

plt.scatter(0.5,np.median(uncert_no_clipper),c='black')
plt.scatter(1.5,np.median(uncert_clipper),c='black')

plt.xticks([.5,1.5],['No\nclipper\npresent', 'Clipper\npresent'])
plt.ylabel('$\sigma_{epistemic}$',size=15)

plt.show()
plt.close()

### plot precision metric pMAE as function of percentage of predictions rejected by uncertainty ###

quantiles = np.arange(1,0,-.01)
pMAE = []

for e,u,y in zip(error,uncertainty,prediction):

	anticipating = (y < upper_bound) & (y > lower_bound)
	e,u = e[anticipating], u[anticipating]

	pMAE_ins = []
	for q in quantiles:
		
		uncert_threshold = np.quantile(u,q)
		uncert_filter = u < uncert_threshold
		error_filtered = e[uncert_filter].mean()

		pMAE_ins.append(error_filtered)

	pMAE.append(np.array(pMAE_ins))

pMAE = np.stack(pMAE)

for i,prec in enumerate(pMAE):
	plt.plot(quantiles,prec,linestyle='dashed',color=util.colors[i],label='{}'.format(util.instruments[i]))
plt.plot(quantiles,pMAE.mean(axis=0),label='Mean',color='black')
plt.xlim(1,0)
plt.ylim(.1*horizon,.5*horizon)
plt.xlabel('Filtered predictions [%]',size=12)
plt.ylabel('pMAE [min.]',size=12)
plt.legend(fontsize=10)
plt.show()
plt.close()