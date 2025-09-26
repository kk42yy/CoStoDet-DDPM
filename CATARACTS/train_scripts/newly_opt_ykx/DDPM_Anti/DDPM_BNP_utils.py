import os
import torch
import datetime
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, jaccard_score, f1_score


def prepare_output_folders(opts):

	if opts.image_based:
		temp_head = 'imageBased'
	else:
		temp_head = opts.head
	if opts.freeze:
		depth = 'frozen'
	else:
		depth = 'e2e'
	if opts.shuffle:
		shuffle = '_shuffle'
	else:
		shuffle = ''
	if opts.CHE:
		che = '_CHE'
	else:
		che = '_CHT'
	
	try:
		horizon = f"_horizon{opts.horizon}"
	except:
		horizon = ''

	trial_name_full = '{}_{}{}_{}Split_{}_{}_lr{}_bs{}_seq{}_{}{}{}'.format(
		datetime.datetime.now().strftime("%Y%m%d-%H%M"),
		opts.trial_name,
		horizon,
		opts.split,
		temp_head,
		opts.backbone,
        opts.lr,
		opts.batch_size,
		opts.seq_len,
		depth,
		shuffle,
		che
	)

	if opts.only_temporal:
		trial_name_full = '{}_{}_{}Split_{}_{}_2step'.format(
			datetime.datetime.now().strftime("%Y%m%d-%H%M"),
			opts.trial_name,
			opts.split,
			temp_head,
			opts.backbone
		)

	output_folder = os.path.join(opts.output_folder,trial_name_full)
	print('Output directory: ' + output_folder)
	result_folder = os.path.join(output_folder,'results')
	script_folder = os.path.join(output_folder,'scripts')
	model_folder = os.path.join(output_folder,'models')
	log_path = os.path.join(output_folder,'log.txt')

	os.makedirs(result_folder,exist_ok=True)
	os.makedirs(script_folder,exist_ok=True)
	os.makedirs(model_folder,exist_ok=True)

	# for f in os.listdir():
	# 	if '.py' in f:
	# 		copy2(f,script_folder)

	return result_folder, model_folder, log_path

class AnticipationMetricMeter:

    def __init__(self, horizon, num_ins):

        self.eval_metric = torch.nn.L1Loss(reduction='sum')

        self.horizon = horizon
        self.num_ins = num_ins
        self.phase_joint = False
        
        self.loss = 0
        self.l1 = 0
        self.count = 0

        self.inMAE = 0
        self.inMAE_count = 0
        self.outMAE = 0
        self.outMAE_count = 0
        self.eMAE = 0
        self.eMAE_count = 0
        self.pMAE = 0
        self.pMAE_count = 0
        
        self.phase_inMAE = 0
        self.phase_inMAE_count = 1e-7
        self.phase_outMAE = 0
        self.phase_outMAE_count = 1e-7
        self.phase_eMAE = 0
        self.phase_eMAE_count = 1e-7
        self.phase_pMAE = 0
        self.phase_pMAE_count = 1e-7

    def update(self,loss,output=None,target=None,B=1,phase_joint=False):
        
        if output is None and target is None:
            self.loss += loss * B
            self.count += B
            return
        
        output, target = output.flatten(end_dim=1), target.flatten(end_dim=1)

        # ### DDPM 240924 1-target
        # output = 1. - output
        # target = 1. - target

        self.loss += loss * target.size(0)
        self.l1 += (self.eval_metric(output,target) / self.num_ins).item()
        self.count += target.size(0)

        output, target = output.clone(), target.clone()
        output *= self.horizon
        target *= self.horizon
        
        if phase_joint:
            self.phase_joint = phase_joint
            phase_output, phase_target = output[:,5:], target[:,5:]
            output, target = output[:,:5], target[:,:5]

            phase_inside_horizon = (phase_target < self.horizon) & (phase_target > 0)
            phase_outside_horizon = phase_target == self.horizon
            phase_early_horizon = (phase_target < (self.horizon*.1)) & (phase_target > 0)
            phase_anti = (phase_output > (self.horizon*.1)) & (phase_output < (self.horizon*.9))

            phase_abs_error = (phase_output-phase_target).abs()
            phase_zeros = torch.zeros(1).cuda()

            self.phase_inMAE += torch.where(phase_inside_horizon, phase_abs_error, phase_zeros).sum(dim=0).data
            self.phase_inMAE_count += phase_inside_horizon.sum(dim=0).data
            self.phase_outMAE += torch.where(phase_outside_horizon, phase_abs_error, phase_zeros).sum(dim=0).data
            self.phase_outMAE_count += phase_outside_horizon.sum(dim=0).data
            self.phase_eMAE += torch.where(phase_early_horizon, phase_abs_error, phase_zeros).sum(dim=0).data
            self.phase_eMAE_count += phase_early_horizon.sum(dim=0).data
            self.phase_pMAE += torch.where(phase_anti, phase_abs_error, phase_zeros).sum(dim=0).data
            self.phase_pMAE_count += phase_anti.sum(dim=0).data

        inside_horizon = (target < self.horizon) & (target > 0)
        outside_horizon = target == self.horizon
        early_horizon = (target < (self.horizon*.1)) & (target > 0)
        anti = (output > (self.horizon*.1)) & (output < (self.horizon*.9))

        abs_error = (output-target).abs()
        zeros = torch.zeros(1).cuda()

        self.inMAE += torch.where(inside_horizon, abs_error, zeros).sum(dim=0).data
        self.inMAE_count += inside_horizon.sum(dim=0).data
        self.outMAE += torch.where(outside_horizon, abs_error, zeros).sum(dim=0).data
        self.outMAE_count += outside_horizon.sum(dim=0).data
        self.eMAE += torch.where(early_horizon, abs_error, zeros).sum(dim=0).data
        self.eMAE_count += early_horizon.sum(dim=0).data
        self.pMAE += torch.where(anti, abs_error, zeros).sum(dim=0).data
        self.pMAE_count += anti.sum(dim=0).data

    def get_scores(self, mode='val'):

        loss = self.loss/self.count
        if mode == 'train':
            return loss
        
        l1 = self.l1/self.count

        inMAE = (self.inMAE/self.inMAE_count).mean().item()
        outMAE = (self.outMAE/self.outMAE_count).mean().item()
        wMAE = (inMAE+outMAE)/2
        MAE = ((self.inMAE + self.outMAE) / (self.inMAE_count + self.outMAE_count)).mean().item()
        eMAE = (self.eMAE/self.eMAE_count).mean().item()
        pMAE = (self.pMAE/self.pMAE_count).mean().item()

        if self.phase_joint:
            phase_inMAE = (self.phase_inMAE/self.phase_inMAE_count).mean().item()
            phase_outMAE = (self.phase_outMAE/self.phase_outMAE_count).mean().item()
            phase_wMAE = (phase_inMAE+phase_outMAE)/2
            phase_MAE = ((self.phase_inMAE + self.phase_outMAE) / (self.phase_inMAE_count + self.phase_outMAE_count)).mean().item()
            phase_eMAE = (self.phase_eMAE/self.phase_eMAE_count).mean().item()
            phase_pMAE = (self.phase_pMAE/self.phase_pMAE_count).mean().item()
            
            return loss, l1, inMAE, outMAE, wMAE, MAE, eMAE, pMAE,\
                phase_inMAE, phase_outMAE, phase_wMAE, phase_MAE, phase_eMAE, phase_pMAE

        return loss, l1, inMAE, outMAE, wMAE, MAE, eMAE, pMAE

    def reset(self):

        self.loss = 0
        self.l1 = 0
        self.count = 0

        self.inMAE = 0
        self.inMAE_count = 0
        self.outMAE = 0
        self.outMAE_count = 0
        self.eMAE = 0
        self.eMAE_count = 0
        self.pMAE = 0
        self.pMAE_count = 0
        
        self.phase_inMAE = 0
        self.phase_inMAE_count = 1e-7
        self.phase_outMAE = 0
        self.phase_outMAE_count = 1e-7
        self.phase_eMAE = 0
        self.phase_eMAE_count = 1e-7
        self.phase_pMAE = 0
        self.phase_pMAE_count = 1e-7

    def start_new_op(self):

        pass
    
class PhaseMetricMeter:

    def __init__(self, num_classes):

        self.num_classes = num_classes
        self.count = 0
        self.loss = 0
        self.reset()

    def update(self,loss,output=None,target=None,B=1,phase_joint=False):

        if output is None and target is None:
            self.loss += loss * B
            self.count += B
            return
        output, target = output.flatten(end_dim=1), target.flatten(end_dim=1)

        self.loss += loss * target.size(0)
        _, predicted = torch.max(output, dim=1)

        self.pred_per_vid[-1] = np.concatenate([self.pred_per_vid[-1], predicted.cpu().numpy()])
        self.target_per_vid[-1] = np.concatenate([self.target_per_vid[-1], target.cpu().numpy()])

    def get_scores(self, mode='val'):
        if mode == 'train':
            loss = self.loss / self.count
            return loss
        
        # mean video-wise metrics

        acc_vid = np.mean([accuracy_score(gt,pred) for gt,pred in zip(self.target_per_vid,self.pred_per_vid)])
        ba_vid = np.mean([balanced_accuracy_score(gt,pred) for gt,pred in zip(self.target_per_vid,self.pred_per_vid)])

        # frame-wise metrics

        all_predictions = np.concatenate(self.pred_per_vid)
        all_targets = np.concatenate(self.target_per_vid)

        acc_frames = accuracy_score(all_targets,all_predictions)
        p = precision_score(all_targets,all_predictions,average='macro')
        r = recall_score(all_targets,all_predictions,average='macro')
        j = jaccard_score(all_targets,all_predictions,average='macro')
        f1 = f1_score(all_targets,all_predictions,average='macro')

        loss = self.loss / len(all_targets)

        return loss, acc_frames, p, r, j, f1, ba_vid, acc_vid

    def reset(self):

        self.loss = 0
        self.count = 0

        self.pred_per_vid = []
        self.target_per_vid = []

    def start_new_op(self):

        self.pred_per_vid.append(np.array([],dtype=int))
        self.target_per_vid.append(np.array([],dtype=int))