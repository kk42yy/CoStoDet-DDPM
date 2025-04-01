import torch
from random import shuffle
from options_train import parser
from dataloader import prepare_dataset, prepare_image_features, prepare_batch
from model_anticipation import AnticipationModel
from model_phase import PhaseModel
import util_train as util
import os
import pandas as pd

opts = parser.parse_args()

# assumes <opts.resume> has form "output/checkpoints/<task>/<trial_name>/models/<checkpoint>.pth.tar"
out_folder = os.path.dirname(os.path.dirname(opts.resume)).replace('/checkpoints/','/predictions/')
gt_folder = os.path.join(out_folder,'gt')
pred_folder = os.path.join(out_folder,'pred_BNPitfalls')
os.makedirs(gt_folder,exist_ok=True)
os.makedirs(pred_folder,exist_ok=True)

if opts.task == 'anticipation':
	model = AnticipationModel(opts,train=False)
if opts.task == 'phase':
	model = PhaseModel(opts,train=False)

if opts.only_temporal:
	_,_,test_set = prepare_image_features(model.net,opts,test_mode=True)
else:
	_,test_set,_ = prepare_dataset(opts)

with torch.no_grad():

	if opts.cheat:
		model.net.train()
	else:
		model.net.eval()

	for ID,op in test_set:

		predictions = []
		labels = []

		if not opts.image_based:
			model.net.temporal_head.reset()
		
		model.metric_meter['test'].start_new_op()

		for data,target in op:

			data, target = prepare_batch(data,target)

			if opts.shuffle:
				model.net.temporal_head.reset()

			if opts.sliding_window:
				output = model.forward_sliding_window(data)
			else:
				output = model.forward(data)

			model.update_stats(
				0,
				output,
				target,
				mode='test'
			)

			if opts.task == 'phase':
				_,pred = output[-1].max(dim=2)
				predictions.append(pred.flatten())
				labels.append(target.flatten())
			
			elif opts.task == 'anticipation':
				pred = output[-1][0]
				pred *= opts.horizon
				target *= opts.horizon
				predictions.append(pred.flatten(end_dim=-2))
				labels.append(target.flatten(end_dim=-2))

		predictions = torch.cat(predictions)
		labels = torch.cat(labels)
	
		if opts.task == 'phase':
			predictions = pd.DataFrame(predictions.cpu().numpy(),columns=['Phase'])
			labels = pd.DataFrame(labels.cpu().numpy(),columns=['Phase'])
			
		elif opts.task == 'anticipation':
			predictions = pd.DataFrame(predictions.cpu().numpy(),columns=['Bipolar','Scissors','Clipper','Irrigator','SpecBag'])
			labels = pd.DataFrame(labels.cpu().numpy(),columns=['Bipolar','Scissors','Clipper','Irrigator','SpecBag'])


		predictions.to_csv(os.path.join(pred_folder,'video{}-phase.txt'.format(ID)), index=True,index_label='Frame',sep='\t')
		labels.to_csv(os.path.join(gt_folder,'video{}-phase.txt'.format(ID)), index=True,index_label='Frame',sep='\t')
		print('saved predictions/labels for video {}'.format(ID))
		
	epoch = torch.load(opts.resume)['epoch']
	model.summary(log_file=os.path.join(pred_folder, 'log.txt'), epoch=epoch)
	from visualization.Visualize import visual_main
	visual_main(out_folder)
