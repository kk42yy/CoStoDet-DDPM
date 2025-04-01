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

class PhaseModel(nn.Module):

	def __init__(self,opts,train=True,num_train_step=1000,last_epoch=-1):
		super().__init__()
		self.opts = opts
		self.train = train
		
		print("horizon:", opts.seq_len)
		print("n_obs_steps:", opts.n_obs_steps)
		print("n_action_steps:", opts.n_action_steps)
		print("infer_steps:", opts.infer_steps)
  

		self.net = DiffusionUnetHybridImagePolicy(
			noise_scheduler=(DDIMScheduler if opts.DDIM else DDPMScheduler),
			action_dim=opts.num_classes,
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
			'train': util.PhaseMetricMeter(opts.num_classes),
			'val': util.PhaseMetricMeter(opts.num_classes),
			'test': util.PhaseMetricMeter(opts.num_classes)
		}

		if self.train:
			self.result_folder, self.model_folder, self.log_path = util.prepare_output_folders(opts)
			self.best_acc = 0
			self.best_f1 = 0
			
			weight = torch.Tensor([
				1.6411019141231247,
				0.19090963801041133,
				1.0,
				0.2502662616859295,
				1.9176363911137977,
				0.9840248158200853,
				2.174635818337618,
			]).cuda()
			self.criterion = nn.CrossEntropyLoss(reduction='mean',weight=weight)
			
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

		output_reg = output.transpose(1, 2)
		target_reg = target

		return self.criterion(output_reg,target_reg)
	
	def compute_loss(self,output,target):
		loss = self.compute_loss_single_prediction(output,target)

		return loss

	def summary(self,log_file=None,epoch=None):

		if self.train:

			loss_train = self.metric_meter['train'].get_scores('train')
			_, _, _, _, _, f1_val, ba_val, acc_val = self.metric_meter['val'].get_scores()
			# _, _, p_test, r_test, j_test, f1_test, ba_test, acc_test = self.metric_meter['test'].get_scores()

			log_message = (
				f'Epoch {epoch:3d}: '
				f'Train (loss {loss_train:1.3f}) '
				f'Val (f1 {f1_val:1.3f}, ba {ba_val:1.3f}, acc {acc_val:1.3f}) '
				# f'Test (Frame scores: p {p_test:1.3f}, r {r_test:1.3f}, j {j_test:1.3f}, f1 {f1_test:1.3f}; Video scores: ba {ba_test:1.3f}, acc {acc_test:1.3f}) '
			)

			checkpoint = {
				'epoch': epoch,
				'state_dict': self.net.state_dict(),
				'ema_state_dict': self.ema_model.state_dict() if self.use_ema else None,
				# 'optimizer' : self.optimizer.state_dict(),
				# 'predictions': self.metric_meter['test'].pred_per_vid,
				# 'targets': self.metric_meter['test'].target_per_vid,
				# 'scores': {
				# 	'acc': acc_test,
				# 	'ba': ba_test,
				# 	'f1': f1_test
				# }
			}
			if self.opts.image_based:
				model_file_path = os.path.join(self.model_folder,'checkpoint_{:03d}.pth.tar'.format(epoch))
			else:
				model_file_path = os.path.join(self.model_folder,'checkpoint_current.pth.tar')
			torch.save(checkpoint, model_file_path)

			if f1_val > self.best_f1:
				model_file_path = os.path.join(self.model_folder,'checkpoint_best_f1.pth.tar')
				torch.save(checkpoint, model_file_path)
				self.best_f1 = f1_val

			if acc_val > self.best_acc:
				model_file_path = os.path.join(self.model_folder,'checkpoint_best_acc.pth.tar')
				torch.save(checkpoint, model_file_path)
				self.best_acc = acc_val

			print(log_message)
			log_file.write(log_message + '\n')
			log_file.flush()

		else:
			loss, aver_acc, p_test, r_test, j_test, f1_test, ba_test, acc_test = self.metric_meter['test'].get_scores()

			log_message = (
				f'Epoch {epoch:3d}: \n'
				f'Test (loss {loss:1.3f}, acc frame {aver_acc:1.3f}\n'
				f'      ba   {ba_test:1.3f}, f1   {f1_test:1.3f}\n'
				f'      acc  {acc_test:1.3f}, prec {p_test:1.3f}, rec  {r_test:1.3f}, jacc {j_test:1.3f})'
			)

			print(log_message)
			with open(log_file, "w+") as f:
				f.write(log_message)