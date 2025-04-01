import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import csv
import torch


class Cholec80Test():
	def __init__(self, image_path, ID, opts, seq_len=None, batch_size=1):
		
		#print(ID)
		self.image_path = image_path
		self.seq_len = opts.seq_len if seq_len is None else seq_len
		self.ext = 'jpg'
		self.dynamic_infer = seq_len is not None
		self.batchsize = batch_size
		self.To = opts.n_obs_steps
		self.Ta = opts.n_action_steps

		self.transform = A.Compose([
			A.SmallestMaxSize(max_size=opts.height),
			A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
			ToTensorV2(),
		])

		# load phase ground truth
		if opts.task == 'phase':
			phase_map = {
				'Preparation':0,
				'CalotTriangleDissection':1,
				'ClippingCutting':2,
				'GallbladderDissection':3,
				'GallbladderPackaging':4,
				'CleaningCoagulation':5,
				'GallbladderRetraction':6
			}
			annotation_path = os.path.join(opts.annotation_folder,'phase_annotations',"video"+ID+"-phase.txt")
			with open(annotation_path, "r") as f:
				self.target = []
				reader = csv.reader(f, delimiter='\t')
				next(reader, None)
				for count,row in enumerate(reader):
					if count % 25 == 0:
						self.target.append(phase_map[row[1]])
			self.target = torch.LongTensor(self.target)

		elif opts.task == 'anticipation':
			annotation_path = os.path.join(opts.annotation_folder,'tool_annotations',"video"+ID+"-tool.txt")
			with open(annotation_path, "r") as f:
				tool_presence = []
				reader = csv.reader(f, delimiter='\t')
				next(reader, None)
				for i,row in enumerate(reader):
					tool_presence.append([int(row[x]) for x in [2,4,5,6,7]])
				tool_presence = torch.LongTensor(tool_presence).permute(1,0)
			self.target = generate_anticipation_gt(tool_presence,opts.horizon)

		self.Length = len(self.target)
		self.idx = 0
		self.imagebank = []
		self.targetbank = []
		self.VIDEOLENGTH = len(self.target)
  
	def __next__(self):
		
		CNT = self.batchsize if self.Length - self.idx >= self.batchsize else self.Length - self.idx
		img_batch, target_batch = [], []
		for _ in range(CNT):
			img, tar = self.get_frame()
			img_batch.append(img)
			target_batch.append(tar)

		return torch.stack(img_batch), torch.stack(target_batch)

	def get_frame(self):

		img, target = self.load_frame(self.idx)

		if self.idx == 0:
			self.imagebank = [img] * self.seq_len
			self.targetbank = [target] * self.seq_len
		else:
			self.imagebank = self.imagebank[1:self.To] + [img] * (self.seq_len - self.To + 1)
			self.targetbank = self.targetbank[1:self.To] + [target] * (self.seq_len - self.To + 1)
		img_seq, target_seq = torch.stack(self.imagebank), torch.stack(self.targetbank)
		
		self.idx += 1

		return img_seq, target_seq #[seq_len,3,H,W]

	def load_frame(self,index):
		target = self.target[index]

		file_name = os.path.join(self.image_path,'{:08d}.{}'.format(index,self.ext))
		img = cv2.imread(file_name)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = self.transform(image=img)['image']

		return img, target

	def __len__(self):
		return (len(self.target) + self.batchsize - 1) // self.batchsize

# generates the ground truth signal over time for a single tool and a single operation
def generate_anticipation_gt_onetool(tool_code,horizon):
	# initialize ground truth signal
	anticipation = torch.zeros_like(tool_code).type(torch.FloatTensor)
	# default ground truth value is <horizon> minutes
	# (i.e. tool will not appear within next <horizon> minutes)
	anticipation_count = horizon
	# iterate through tool-presence signal backwards
	for i in torch.arange(len(tool_code)-1,-1,-1):
		# if tool is present, then set anticipation value to 0 minutes
		if tool_code[i]:
			anticipation_count = 0
		# else increase anticipation value with each (reverse) time step but clip at <horizon> minutes
		else:
			anticipation_count = min(horizon, anticipation_count + 1/60) # video is sampled at 1fps, so 1 step = 1/60 minutes
		anticipation[i] = anticipation_count
	# normalize ground truth signal to values between 0 and 1
	anticipation = anticipation / horizon
	return anticipation

# generates the ground truth signal over time for a single operation
def generate_anticipation_gt(tools,horizon):
	return torch.stack([generate_anticipation_gt_onetool(tool_code,horizon) for tool_code in tools]).permute(1,0)