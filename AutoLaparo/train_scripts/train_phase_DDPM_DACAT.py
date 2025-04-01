from cv2 import phase
import torch, os
from random import shuffle
from tqdm import tqdm
from options_train_DDPM import parser
from dataloader import prepare_dataset, prepare_image_features, prepare_batch
from model_phase_DDPM_DACAT import PhaseModel
import util_train as util
import pandas as pd

opts = parser.parse_args()

if not opts.random_seed:
	torch.manual_seed(7)


train_set, val_set, test_set = prepare_dataset(opts)
num_iters_per_epoch = util.get_iters_per_epoch(train_set,opts) // 2

model = PhaseModel(opts, True, num_iters_per_epoch*opts.epochs)

with open(model.log_path, "w") as log_file:

	log_file.write(f'{model}\n')
	log_file.flush()

	train_pred_save = os.path.split(model.log_path)[0]

	start_epoch = util.get_start_epoch(opts)
	
	for epoch in range(start_epoch,opts.epochs+1):

		model.reset_stats()
		model.net.train()

		if opts.CHE:
			for _,op in train_set:

				model.metric_meter['train'].start_new_op() # necessary to compute video-wise metrics
				
				with tqdm(total=num_iters_per_epoch) as pbar:
					for i, data_target in enumerate(op):
						
						if opts.Phase_joint:
							data, target, phase_target = data_target
							data, target = prepare_batch(data,target)
							_, phase_target = prepare_batch(phase_target, phase_target)

						else:
							data, target = data_target
							phase_target = None
							data, target = prepare_batch(data,target)

						loss = model.forward(data, target, phase_target)
						model.update_weights(loss)

						model.update_stats(
							loss.item(),
							mode='train',
						)

						pbar.update(1)

						if opts.shuffle and (i+1) >= num_iters_per_epoch:
							break

					# 	break
					# break
		else:
			for _,op in tqdm(train_set):

				model.metric_meter['train'].start_new_op() # necessary to compute video-wise metrics
				model.net.obs_encoder.lstm.reset()
				model.net.obs_encoder.CA_lstm.reset()
				model.net.obs_encoder.long_cache_reset()
				
				for i, (data, target) in enumerate(op):
					
					data, target = prepare_batch(data,target)
					if i == 0:
						model.net.gt_cache_reset(target)
					
					loss = model.forward(data, target)
					model.update_weights(loss)

					model.update_stats(
						loss.item(),
						mode='train'
					)

					if opts.shuffle and (i+1) >= num_iters_per_epoch:
						break

				# 	break
				# break


		with torch.no_grad():
			policy = model.net
			if opts.use_ema:
				policy = model.ema_model

			if opts.cheat:
				policy.train()
			else:
				policy.eval()

			for mode in ['val']:

				if mode == 'val':
					eval_set = tqdm(val_set)
				# elif mode == 'test':
				# 	eval_set = test_set

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

					for data, target in op:
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
						# # L = 32
						# if data.size(1) % L != 0:
						# 	t = L - data.size(1) % L
						# 	data = torch.cat([data, data[:,-1:].repeat(1,t,1,1,1)], dim=1)
						# if opts.Obs != "LSTM":
						# 	data = data.reshape(-1, L, *data.shape[2:])
						
						# change flatten data
						output = policy.predict_action(data, data.shape[1])['action_pred']
						# output = output.reshape(1,-1,*output.shape[2:])[:,:target.size(1)]

						loss = model.compute_loss(output,target)
						model.update_stats(
							loss.item(),
							output,
							target,
							mode=mode,
							phase_anti=opts.num_classes > 5
						)


						# break
					# break
					# predictions = torch.cat(predictions)
					# # labels = torch.cat(labels)
				
					# predictions = pd.DataFrame(predictions.cpu().numpy(),columns=['Bipolar','Scissors','Clipper','Irrigator','SpecBag'])
					# # labels = pd.DataFrame(labels.cpu().numpy(),columns=['Bipolar','Scissors','Clipper','Irrigator','SpecBag'])
					
					# predictions.to_csv(os.path.join(train_pred_save_epoch,'video{}-phase.txt'.format(ID)), index=True,index_label='Frame',sep='\t')
					# # labels.to_csv(os.path.join(gt_folder,'video{}-phase.txt'.format(ID)), index=True,index_label='Frame',sep='\t')
					# # print('saved predictions/labels for video {}'.format(ID))
			
			policy.train()
			model.summary(log_file,epoch)