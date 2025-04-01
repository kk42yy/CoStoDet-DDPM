from ast import mod
import torch
from torch import nn, optim
from nfnets_optim import SGD_AGC
from networks import CNN, TemporalCNN
# import util_train as util
import newly_opt_ykx.DDPM_Anti.DDPM_BNP_utils as util
import os
import numpy as np

class AnticipationModel(nn.Module):

	def __init__(self,opts,train=True):
		super().__init__()
		self.opts = opts
		self.train = train
		self.horizon = opts.horizon

		#output_size = opts.num_ins
		output_size = opts.num_ins + 3*opts.num_ins
		if opts.image_based:
			self.net = CNN(output_size,opts.backbone,opts).cuda()
			for param in self.net.parameters():
				param.requires_grad = True
		else:
			self.net = TemporalCNN(output_size,opts.backbone,opts.head,opts).cuda()

		if opts.only_temporal:
			for param in self.net.cnn.parameters():
				param.requires_grad = False

		if not opts.image_based:
			if opts.cnn_weight_path != 'imagenet':
				checkpoint = torch.load(opts.cnn_weight_path)
				self.net.cnn.load_state_dict(checkpoint['state_dict'])
				print('loaded pretrained CNN weights...')
			else:
				print('loaded ImageNet weights...')

		if opts.resume is not None:
			checkpoint = torch.load(opts.resume)
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

			if opts.backbone == 'nfnet':
				self.optimizer = SGD_AGC(
					named_params=self.net.named_parameters(),
					lr=opts.lr,
					momentum=0.9,
					clipping=0.1,
					weight_decay=opts.weight_decay,
					nesterov=True,
				)
			else:
				#self.optimizer = optim.Adam(self.net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
				self.optimizer = optim.AdamW(self.net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
				#self.optimizer = optim.AdamW(
				#	[
				#		{'params': self.net.cnn.parameters(), 'lr': 1e-5},
				#		{'params': self.net.temporal_head.parameters(), 'lr': opts.lr},
				#	],
				#	weight_decay=opts.weight_decay
				#)
			
			if opts.resume is not None:
				self.optimizer.load_state_dict(checkpoint['optimizer'])
				print('loaded optimizer settings...')
			
			self.optimizer.zero_grad()
			self.batch_accum = 1
			self.batch_num = 0
			# doesn't seem to work:
			#if opts.backbone == 'nfnet':
			#	self.optimizer = AGC(self.net.parameters(), self.optimizer, model=self.net, ignore_agc=['out_layer'])

	def forward(self,data): 

		if self.opts.only_temporal:
			output = self.net.temporal_head(data)
		else:
			output = self.net(data)

		output = [self.structure_prediction(out) for out in output]

		return output

	def forward_sliding_window(self,data):

		output = self.net.forward_sliding_window(data)

		output = [self.structure_prediction(out) for out in output]

		return output
		
	def compute_loss_single_prediction(self,output,target):

		output_reg, output_cls = output # cls: B*S*3*5
		output_cls = output_cls.transpose(1,2) # B*3*S*5

		target_cls = torch.where((target < 1) & (target > 0),torch.cuda.FloatTensor([2]),target).type(torch.cuda.LongTensor)
		target_reg = target

		return (self.criterion(output_reg,target_reg) + self.opts.loss_scale*self.criterion_cls(output_cls,target_cls)) * self.opts.num_ins
		# return self.criterion(output_reg,target_reg)


	def compute_loss(self,output,target):

		loss = [self.compute_loss_single_prediction(out,target) for out in output]
		loss = sum(loss) / len(loss)

		return loss

	def update_weights(self,loss):

		loss = loss / self.batch_accum
		loss.backward()
		self.batch_num += 1

		if self.batch_num == self.batch_accum:
			self.optimizer.step()
			self.optimizer.zero_grad()
			self.batch_num = 0

	def reset_stats(self):

		self.metric_meter['train'].reset()
		self.metric_meter['val'].reset()
		self.metric_meter['test'].reset()

	def update_stats(self,loss,output,target,mode,phase_anti=False):

		output = output[-1][0].detach()
		target = target.detach()

		self.metric_meter[mode].update(loss,output,target,phase_joint=(mode != 'train'))
		
	def summary_ori(self,log_file=None,epoch=None):

		if self.train:

			loss_train, l1_train, _, _, _, _, _, _ = self.metric_meter['train'].get_scores()
			loss_val, l1_val, inMAE_val, outMAE_val, wMAE_val, MAE_val, eMAE_val, pMAE_val = self.metric_meter['val'].get_scores()
			loss_test, l1_test, inMAE_test, outMAE_test, wMAE_test, MAE_test, eMAE_test, pMAE_test = self.metric_meter['test'].get_scores()

			log_message = (
				'Epoch {:3d}: '
				'Train (loss {:1.3f}, L1 {:1.3f}) '
				'Val (loss {:1.3f}, L1 {:1.3f}, MAE {:1.3f}, eMAE {:1.3f}, inMAE {:1.3f}, outMAE {:1.3f}, wMAE {:1.3f}, pMAE {:1.3f})'
				'Test (loss {:1.3f}, L1 {:1.3f}, MAE {:1.3f}, wMAE {:1.3f})'
			).format(
				epoch,
				loss_train, l1_train,
				loss_val, l1_val, MAE_val, eMAE_val, inMAE_val, outMAE_val, wMAE_val, pMAE_val,
				loss_test, l1_test, MAE_test, wMAE_test,
			)

			checkpoint = {
				'epoch': epoch,
				'state_dict': self.net.state_dict(),
				'optimizer' : self.optimizer.state_dict(),
				'scores': {
					'l1': l1_test,
					'inMAE': inMAE_test,
					'outMAE': outMAE_test,
					'wMAE': wMAE_test,
					'MAE': MAE_test,
					'eMAE': eMAE_test,
					'pMAE': pMAE_test
				}
			}
			#if self.opts.image_based:
			# if (epoch % 10) == 0:
			# 	model_file_path = os.path.join(self.model_folder,'checkpoint_{:03d}.pth.tar'.format(epoch))
			# 	torch.save(checkpoint, model_file_path)
			#else:
			model_file_path = os.path.join(self.model_folder,'checkpoint_current.pth.tar')
			torch.save(checkpoint, model_file_path)

			if MAE_val < self.best_MAE:
				model_file_path = os.path.join(self.model_folder,'checkpoint_best_MAE.pth.tar')
				torch.save(checkpoint, model_file_path)
				self.best_MAE = MAE_val

			if wMAE_val < self.best_wMAE:
				model_file_path = os.path.join(self.model_folder,'checkpoint_best_wMAE.pth.tar')
				torch.save(checkpoint, model_file_path)
				self.best_wMAE = wMAE_val

			if inMAE_val < self.best_inMAE:
				model_file_path = os.path.join(self.model_folder,'checkpoint_best_inMAE.pth.tar')
				torch.save(checkpoint, model_file_path)
				self.best_inMAE = inMAE_val
			
			if eMAE_val < self.best_eMAE:
				model_file_path = os.path.join(self.model_folder,'checkpoint_best_eMAE.pth.tar')
				torch.save(checkpoint, model_file_path)
				self.best_eMAE = eMAE_val

			if pMAE_val < self.best_pMAE:
				model_file_path = os.path.join(self.model_folder,'checkpoint_best_pMAE.pth.tar')
				torch.save(checkpoint, model_file_path)
				self.best_pMAE = pMAE_val

			print(log_message)
			log_file.write(log_message + '\n')
			log_file.flush()

		else:

			loss_test, l1_test, inMAE_test, outMAE_test, wMAE_test, MAE_test, eMAE_test, pMAE_test = self.metric_meter['test'].get_scores()

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
				'ema_state_dict': self.ema_model.state_dict(),
				# 'ema_state_dict': self.ema_model.state_dict() if self.use_ema else None,
				# # 'optimizer' : self.optimizer.state_dict(),
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