from regex import F
import torch
from random import shuffle
from options_train_DDPM import parser
from dataloader import prepare_dataset, prepare_image_features, prepare_batch
from model_phase_DDPM_DACAT import PhaseModel
import util_train as util
import os
import pandas as pd
from tqdm import tqdm

seed = 114514
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

opts = parser.parse_args()

# assumes <opts.resume> has form "output/checkpoints/<task>/<trial_name>/models/<checkpoint>.pth.tar"
out_folder = os.path.dirname(os.path.dirname(opts.resume)).replace('/checkpoints/','/predictions/')
gt_folder = os.path.join(out_folder,'gt')
pred_folder = os.path.join(out_folder,f'pred_CoStoDet-DDPM')
os.makedirs(gt_folder,exist_ok=True)
os.makedirs(pred_folder,exist_ok=True)

_, test_set, _ = prepare_dataset(opts)
num_iters_per_epoch = util.get_iters_per_epoch(test_set,opts)
model = PhaseModel(opts, False)

with torch.no_grad():
	policy = model.net
	if opts.use_ema:
		policy = model.ema_model

	if opts.cheat:
		policy.train()
	else:
		policy.eval()

	for mode in ['test']:

		eval_set = test_set

		for ID, op in eval_set:

			predictions = []
			labels = []

			model.metric_meter[mode].start_new_op()
			try:
				policy.obs_encoder.lstm.reset()
				policy.obs_encoder.CA_lstm.reset()
				policy.obs_encoder.long_cache_reset()
			except:
				pass

			for data, target in tqdm(op):
				# if torch.rand(1) < 0.8:
				# 	continue

				data, target = prepare_batch(data,target) # data [B,horizon,3,216,384], target [B,horizon,5]
				
				"""
				output: {
					'action': [B, Ta, 5]
							: o|o|
								a|a|a|a|a|a|a|a
					'action_pred': [B, horizon, 5]
							: o|o|
							  a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|
				}
				ours: {
					'nocheat': [B, Ta, 5]
							: o|o|o|o|o|o|o|o|o|o|o|o|o|o|o|o| 32 frame observation
															a| overlap last-frame inference
					'cheat': [B, horizon, 5]
							: o|o|o|o|o|o|o|o|o|o|o|o|o|o|o|o| 32 frame observation
							  a|a|a|a|a|a|a|a|a|a|a|a|a|a|a|a| no overlap inference
				}
				"""
				# L = opts.seq_len
				# if data.size(1) % L != 0:
				# 	t = L - data.size(1) % L
				# 	data = torch.cat([data, data[:,-1:].repeat(1,t,1,1,1)], dim=1)
				# data = data.reshape(-1, L, *data.shape[2:])
				
				# change flatten data
				output = policy.predict_action(data, data.shape[1])['action_pred']
				# output = output.reshape(1,-1,*output.shape[2:])[:,:target.size(1)]

				model.update_stats(
					0,
					output,
					target,
					mode=mode,
					phase_anti=False
				)
    
				_,pred = output[0].max(dim=-1)
				target = target[0]
				
				# pred *= opts.horizon
				# target *= opts.horizon
				predictions.append(pred.flatten())
				labels.append(target.flatten())


			predictions = torch.cat(predictions)
			labels = torch.cat(labels)
		
			if opts.task == 'phase':
				predictions = pd.DataFrame(predictions.cpu().numpy(),columns=['Phase'])
				labels = pd.DataFrame(labels.cpu().numpy(),columns=['Phase'])
				
			elif opts.task == 'anticipation':
				if opts.num_ins > 5:
					columns = [
                                # 'Frame',
                                # 'Grasper',
                                'Bipolar',
                                # 'Hook',
                                'Scissors',
                                'Clipper',
                                'Irrigator',
                                'SpecimenBag',
                                'Preparation',
                                'CalotTriangleDissection',
                                'ClippingCutting',
                                'GallbladderDissection',
                                'GallbladderPackaging',
                                'CleaningCoagulation',
                                'GallbladderRetraction'
                             ]
					predictions = pd.DataFrame(predictions.cpu().numpy(), columns=columns)
					labels = pd.DataFrame(labels.cpu().numpy(), columns=columns)
					
				else:
					predictions = pd.DataFrame(predictions.cpu().numpy(),columns=['Bipolar','Scissors','Clipper','Irrigator','SpecBag'])
					labels = pd.DataFrame(labels.cpu().numpy(),columns=['Bipolar','Scissors','Clipper','Irrigator','SpecBag'])


			predictions.to_csv(os.path.join(pred_folder,'video{}-phase.txt'.format(ID)), index=True,index_label='Frame',sep='\t')
			labels.to_csv(os.path.join(gt_folder,'video{}-phase.txt'.format(ID)), index=True,index_label='Frame',sep='\t')
			print('saved predictions/labels for video {}'.format(ID))

		epoch = torch.load(opts.resume, 'cpu')['epoch']
		model.summary(log_file=os.path.join(pred_folder, 'log.txt'), epoch=epoch)