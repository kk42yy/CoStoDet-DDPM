from json import load
import torch
from torch import nn, optim
from newly_opt_ykx.DDPM_Anti.DDPM_ForBackward import DiffusionUnetHybridImagePolicy
from newly_opt_ykx.DDPM_Anti.DDPM_EMA import EMAModel
import newly_opt_ykx.DDPM_Anti.DDPM_BNP_utils as util
import os, math, copy
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class AnticipationModel(nn.Module):

	def __init__(self,opts,train=True,num_train_step=1000,last_epoch=-1):
		super().__init__()
		self.opts = opts
		self.train = train
		self.horizon = opts.horizon

		print("horizon:", opts.seq_len)
		print("n_obs_steps:", opts.n_obs_steps)
		print("n_action_steps:", opts.n_action_steps)
		print("infer_steps:", opts.infer_steps)
  

		self.net = DiffusionUnetHybridImagePolicy(
			noise_scheduler=(DDIMScheduler if opts.DDIM else DDPMScheduler),
			action_dim=opts.num_ins,
			horizon=opts.seq_len,
			n_obs_steps=opts.n_obs_steps,
			n_action_steps=opts.n_action_steps,
			num_train_timesteps=100,
			num_inference_steps=opts.infer_steps,
			opts=opts,
			loadpretrain=False
		).cuda()
		
		self.use_ema = opts.use_ema
		if self.use_ema:
			self.ema_model = copy.deepcopy(self.net).cuda()
			self.EMA = EMAModel(
				inv_gamma=1.0,
				max_value=0.9999,
				min_value=0.0,
				power=0.75,
				update_after_step=0,
				model=self.ema_model
			)

		if opts.resume is not None:
			checkpoint = torch.load(opts.resume)
			if opts.use_ema:
				self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
			else:
				self.net.load_state_dict(checkpoint['state_dict'])
			print('loaded model weights...')

		self.metric_meter = {
			'train': util.AnticipationMetricMeter(opts.horizon, opts.num_ins),
			'val': util.AnticipationMetricMeter(opts.horizon, opts.num_ins),
			'test': util.AnticipationMetricMeter(opts.horizon, opts.num_ins)
		}

		if self.train:
			self.result_folder, self.model_folder, self.log_path = util.prepare_output_folders(opts)
			self.best_MAE = 99999
			self.best_wMAE = 99999
			self.best_inMAE = 99999
			self.best_outMAE = 99999
			self.best_eMAE = 99999
			self.best_pMAE = 99999
			self.avg_best = 99999
			self.criterion = nn.SmoothL1Loss(reduction='mean')
			self.criterion_cls = nn.CrossEntropyLoss(reduction='mean')
			
			## LSTM Obs Frozen ##
			# if self.opts.Obs == 'LSTM':
			# 	self.net.obs_encoder.requires_grad_(False)
    
			self.optimizer = optim.AdamW(self.net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, eps=1e-8,
                                betas=(0.95,0.999))
			self.lr_scheduler = get_cosine_schedule_with_warmup(
				self.optimizer, num_warmup_steps=500, num_training_steps=num_train_step, last_epoch=last_epoch
			)
			
			if opts.resume is not None:
				self.optimizer.load_state_dict(checkpoint['optimizer'])
				print('loaded optimizer settings...')
			
			self.optimizer.zero_grad()
			self.batch_accum = 1
			self.batch_num = 0
			# doesn't seem to work:
			#if opts.backbone == 'nfnet':
			#	self.optimizer = AGC(self.net.parameters(), self.optimizer, model=self.net, ignore_agc=['out_layer'])
			
			normalizer = 0
			self.net.set_normalizer(normalizer)
			if self.use_ema:
				self.ema_model.set_normalizer(normalizer)

				
	def forward(self, data, target, phase_target=None):

		loss = self.net.compute_loss(data, target, phase_target)

		return loss

	def forward_sliding_window(self,data):

		output = self.net.forward_sliding_window(data)

		output = [self.structure_prediction(out) for out in output]

		return output


	def update_weights(self,loss):

		loss = loss / self.batch_accum
		loss.backward()
		self.batch_num += 1

		if self.batch_num == self.batch_accum:
			self.optimizer.step()
			self.optimizer.zero_grad()
			self.lr_scheduler.step()
			self.batch_num = 0
			if self.use_ema:
				self.EMA.step(self.net)

	def reset_stats(self):

		self.metric_meter['train'].reset()
		self.metric_meter['val'].reset()
		self.metric_meter['test'].reset()

	def update_stats(self,loss,output=None,target=None,mode='train',phase_anti=False):
     
		if mode == 'train':
			self.metric_meter[mode].update(loss,B=self.opts.batch_size)
			return

		output = output.detach()
		target = target.detach()

		self.metric_meter[mode].update(loss,output,target,phase_joint=phase_anti)
	
	def compute_loss_single_prediction(self,output,target):

		output_reg = output
		target_reg = target

		return self.criterion(output_reg,target_reg)
	
	def compute_loss(self,output,target):

		loss = self.compute_loss_single_prediction(output,target)

		return loss

	def summary(self,log_file=None,epoch=None):

		if self.train:

			loss_train = self.metric_meter['train'].get_scores('train')
			if self.opts.num_ins > 5:
				loss_val, l1_val, inMAE_val, outMAE_val, wMAE_val, MAE_val, eMAE_val, pMAE_val, \
					phase_inMAE_val, phase_outMAE_val, phase_wMAE_val, phase_MAE_val, phase_eMAE_val, phase_pMAE_val = self.metric_meter['val'].get_scores()
			else:
				loss_val, l1_val, inMAE_val, outMAE_val, wMAE_val, MAE_val, eMAE_val, pMAE_val = self.metric_meter['val'].get_scores()
			# loss_test, l1_test, inMAE_test, outMAE_test, wMAE_test, MAE_test, eMAE_test, pMAE_test = self.metric_meter['test'].get_scores()
			
			if self.opts.num_ins > 5:
				log_message = (
					'Epoch {:3d}: '
					'Train (loss {:1.3f}) '
					'Val Tool (eMAE {:1.3f}, inMAE {:1.3f}, outMAE {:1.3f}, wMAE {:1.3f}, pMAE {:1.3f}) \n'
					'Val Phase (eMAE {:1.3f}, inMAE {:1.3f}, outMAE {:1.3f}, wMAE {:1.3f}, pMAE {:1.3f}) \n'
					# 'Test (loss {:1.3f}, L1 {:1.3f}, MAE {:1.3f}, wMAE {:1.3f})'
				).format(
					epoch,
					loss_train, 
					eMAE_val, inMAE_val, outMAE_val, wMAE_val, pMAE_val,
					phase_eMAE_val, phase_inMAE_val, phase_outMAE_val, phase_wMAE_val, phase_pMAE_val,
					# loss_test, l1_test, MAE_test, wMAE_test,
				)
			else:
				log_message = (
					'Epoch {:3d}: '
					'Train (loss {:1.3f}) '
					'Val (loss {:1.3f}, L1 {:1.3f}, MAE {:1.3f}, eMAE {:1.3f}, inMAE {:1.3f}, outMAE {:1.3f}, wMAE {:1.3f}, pMAE {:1.3f})'
					# 'Test (loss {:1.3f}, L1 {:1.3f}, MAE {:1.3f}, wMAE {:1.3f})'
				).format(
					epoch,
					loss_train, 
					loss_val, l1_val, MAE_val, eMAE_val, inMAE_val, outMAE_val, wMAE_val, pMAE_val,
					# loss_test, l1_test, MAE_test, wMAE_test,
				)

			checkpoint = {
				'epoch': epoch,
				'state_dict': self.net.state_dict(),
				'ema_state_dict': self.ema_model.state_dict() if self.use_ema else None,
				'optimizer' : self.optimizer.state_dict(),
				# 'scores': {
				# 	'l1': l1_test,
				# 	'inMAE': inMAE_test,
				# 	'outMAE': outMAE_test,
				# 	'wMAE': wMAE_test,
				# 	'MAE': MAE_test,
				# 	'eMAE': eMAE_test,
				# 	'pMAE': pMAE_test
				# }
			}
			#if self.opts.image_based:
			# if (epoch % 10) == 0:
			# 	model_file_path = os.path.join(self.model_folder,'checkpoint_{:03d}.pth.tar'.format(epoch))
			# 	torch.save(checkpoint, model_file_path)
			#else:
			model_file_path = os.path.join(self.model_folder,'checkpoint_current.pth.tar')
			torch.save(checkpoint, model_file_path)
			
			
			if wMAE_val < self.best_wMAE:
				model_file_path = os.path.join(self.model_folder,'checkpoint_best_wMAE.pth.tar')
				torch.save(checkpoint, model_file_path)
				self.best_wMAE = wMAE_val


			if eMAE_val < self.best_eMAE:
				model_file_path = os.path.join(self.model_folder,'checkpoint_best_eMAE.pth.tar')
				torch.save(checkpoint, model_file_path)
				self.best_eMAE = eMAE_val
    
			if outMAE_val < self.best_outMAE:
				model_file_path = os.path.join(self.model_folder,'checkpoint_best_outMAE.pth.tar')
				torch.save(checkpoint, model_file_path)
				self.best_outMAE = outMAE_val
			
			try:
				if eMAE_val + outMAE_val + wMAE_val + \
				   phase_eMAE_val + phase_outMAE_val + phase_wMAE_val < self.avg_best:
					model_file_path = os.path.join(self.model_folder,'checkpoint_best_avg.pth.tar')
					torch.save(checkpoint, model_file_path)
					self.avg_best = eMAE_val + outMAE_val + wMAE_val + \
									phase_eMAE_val + phase_outMAE_val + phase_wMAE_val
			
			except:
				if eMAE_val + outMAE_val + wMAE_val < self.avg_best:
					model_file_path = os.path.join(self.model_folder,'checkpoint_best_avg.pth.tar')
					torch.save(checkpoint, model_file_path)
					self.avg_best = eMAE_val + outMAE_val + wMAE_val

			print(log_message)
			log_file.write(log_message + '\n')
			log_file.flush()

		else:

			
			if self.opts.num_ins > 5:
				loss_test, l1_test, inMAE_test, outMAE_test, wMAE_test, MAE_test, eMAE_test, pMAE_test, \
					phase_inMAE_test, phase_outMAE_test, phase_wMAE_test, phase_MAE_test, phase_eMAE_test, phase_pMAE_test = self.metric_meter['test'].get_scores()
			else:
				loss_test, l1_test, inMAE_test, outMAE_test, wMAE_test, MAE_test, eMAE_test, pMAE_test = self.metric_meter['test'].get_scores()
			
			
			
			if self.opts.num_ins > 5:
				log_message = (
					'Epoch {:3d}: '
					'loss {:1.3f}, L1 {:1.3f}, \n'
					'Test Tool  (MAE {:1.3f}, eMAE {:1.3f}, inMAE {:1.3f}, outMAE {:1.3f}, wMAE {:1.3f}, pMAE {:1.3f}) \n'
					'Test Phase (MAE {:1.3f}, eMAE {:1.3f}, inMAE {:1.3f}, outMAE {:1.3f}, wMAE {:1.3f}, pMAE {:1.3f}) \n'
					# 'Test (loss {:1.3f}, L1 {:1.3f}, MAE {:1.3f}, wMAE {:1.3f})'
				).format(
					epoch,
					loss_test, l1_test, 
					MAE_test, eMAE_test, inMAE_test, outMAE_test, wMAE_test, pMAE_test,
					phase_MAE_test, phase_eMAE_test, phase_inMAE_test, phase_outMAE_test, phase_wMAE_test, phase_pMAE_test
					# loss_test, l1_test, MAE_test, wMAE_test,
				)
			else:
				log_message = (
					'Epoch {:3d}: '
					'Test (loss {:1.3f}, L1 {:1.3f}, MAE {:1.3f}, eMAE {:1.3f}, inMAE {:1.3f}, outMAE {:1.3f}, wMAE {:1.3f}, pMAE {:1.3f})'
				).format(
					epoch,
					loss_test, l1_test, MAE_test, eMAE_test, inMAE_test, outMAE_test, wMAE_test, pMAE_test
				)
    
			print(log_message)
			with open(log_file, "w+") as f:
				f.write(log_message)


	def structure_prediction(self,output):

		B = output.size(0)
		S = output.size(1)

		output_reg = output[:,:,-self.opts.num_ins:]
		output_cls = output[:,:,:-self.opts.num_ins].view(B,S,3,self.opts.num_ins)

		return (output_reg, output_cls)