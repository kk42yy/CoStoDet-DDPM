params_gtea = {
   "naming":"default",
   "root_data_dir":"./datasets",
   "dataset_name":"gtea",
   "split_id":1,
   "sample_rate":1,
   "temporal_aug":True,
   "encoder_params":{
    #   "use_instance_norm":False, 
      "num_classes": 5+7,
      "num_layers":10,
      "num_f_maps":64,
      "input_dim":64, # global_cond dim: CNN+LSTM+Linear, 768->512->64
      "kernel_size":5,
      "normal_dropout_rate":0.5,
      "channel_dropout_rate":0.5,
      "temporal_dropout_rate":0.5,
      "feature_layer_indices":[
         5,
         7,
         9,
      ]
   },
   "decoder_params":{
      "input_dim": 192,
      "num_classes": 5+7,
      "num_layers":8,
      "num_f_maps":24,
      "time_emb_dim":512,
      "kernel_size":5,
      "dropout_rate":0.1,
   },
   "diffusion_params":{
      "timesteps":1000,
      "sampling_timesteps":25,
      "ddim_sampling_eta":1.0,
      "snr_scale":0.5,
      "cond_types":  ['full', 'zero', 'boundary03-', 'segment=1', 'segment=1'],
     "detach_decoder": False,
   },
   "loss_weights":{
      "encoder_ce_loss":0.5,
      "encoder_mse_loss":0.1,
      "encoder_boundary_loss":0.0,
      "decoder_ce_loss":0.5,
      "decoder_mse_loss":0.1,
      "decoder_boundary_loss":0.1
   },
   "batch_size":4,
   "learning_rate":0.0005,
   "weight_decay":1e-6,
   "num_epochs":10001,
   "log_freq":100,
   "class_weighting":True,
   "set_sampling_seed":True,
   "boundary_smooth":1,
   "soft_label": 1.4,
   "log_train_results":False,
   "postprocess":{
      "type":"purge",
      "value":3
   },
}

import copy
import math
import torch
import random
import numpy as np
import time as Time
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

# Modified from DiffusionDet and pytorch-diffusion-model

########################################################################################

def get_timestep_embedding(timesteps, embedding_dim): # for diffusion model
    # timesteps: batch,
    # out:       batch, embedding_dim
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

def swish(x):
    return x * torch.sigmoid(x)

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def normalize(x, scale): # [0,1] > [-scale, scale]
    x = (x * 2 - 1.) * scale
    return x

def denormalize(x, scale): #  [-scale, scale] > [0,1]
    x = ((x / scale) + 1) / 2  
    return x

######################################################################################

# class ASDiffusionModel(nn.Module):
#     def __init__(self, encoder_params, decoder_params, diffusion_params, num_classes, device):
#         super(ASDiffusionModel, self).__init__()

class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim=5, # 5 tools reg
        local_cond_dim=None,
        global_cond_dim=None, # To * obs_feature_dim
        diffusion_step_embed_dim=128,
        down_dims=[512,1024,2048],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        To=32,
        LargeDim=32,
        encoder_params=params_gtea["encoder_params"],
        decoder_params=params_gtea["decoder_params"],
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        self.To = To
        
        num_classes = input_dim
        
        self.num_classes = num_classes

        ################################################################

        decoder_params['input_dim'] = len([i for i in encoder_params['feature_layer_indices'] if i not in [-1, -2]]) * encoder_params['num_f_maps']
        if -1 in encoder_params['feature_layer_indices']: # -1 means "video feature"
            decoder_params['input_dim'] += encoder_params['input_dim']
        if -2 in encoder_params['feature_layer_indices']: # -2 means "encoder prediction"
            decoder_params['input_dim'] += self.num_classes

        decoder_params['num_classes'] = num_classes
        encoder_params['num_classes'] = num_classes
        encoder_params.pop('use_instance_norm')

        self.encoder = EncoderModel(**encoder_params)
        self.decoder = DecoderModel(**decoder_params)

    
    def forward(self,
                sample: torch.Tensor, 
                timestep, 
                local_cond=None, global_cond=None, **kwargs):
        """
        sample: (B,T,input_dim,Previous_length=32)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,T*global_cond_dim)
        output: (B,T,input_dim)
        """
        self.To = sample.size(1)
        
        # 0. combine B and T
        B, Time, Dim = sample.size(0), sample.size(1), sample.size(2)
        sample = sample.reshape(B*Time, Dim, -1) # B*Hori, Dim, (1 or 4 or 32)
        global_cond = (global_cond.reshape(B, self.To, -1)).reshape(B*self.To, -1) # B*Hori, 64
        global_cond = global_cond[...,None].repeat(1,1,2)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(B) # [B]
        
        # 2.encoder编码视觉条件特征
        encoder_out, backbone_feats = self.encoder(global_cond, get_features=True) # backbone_feats [B*To, 64*len(indices)=192, 2]
        
        # 3.decoder用视觉条件二次编码+t+gt_noised生成noise
        event_out = self.decoder(backbone_feats, timesteps, sample.float())
        
        event_out = event_out.reshape(B, Time, Dim, -1)

        return event_out



########################################################################################
# Encoder and Decoder are adapted from ASFormer. 
# Compared to ASFormer, the main difference is that this version applies attention in a similar manner as dilated temporal convolutions.
# This difference does not change performance evidently in preliminary experiments.


class EncoderModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes, kernel_size, 
                 normal_dropout_rate, channel_dropout_rate, temporal_dropout_rate,
                 feature_layer_indices=None):
        super(EncoderModel, self).__init__()
        
        self.num_classes = num_classes
        self.feature_layer_indices = feature_layer_indices
        
        self.dropout_channel = nn.Dropout2d(p=channel_dropout_rate)
        self.dropout_temporal = nn.Dropout2d(p=temporal_dropout_rate)
        
        self.conv_in = nn.Conv1d(input_dim, num_f_maps, 1)
        self.encoder = MixedConvAttModule(num_layers, num_f_maps, kernel_size, normal_dropout_rate)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        
        # self.final_outlinear = nn.Linear(input_dim*len(feature_layer_indices), input_dim)


    def forward(self, x, get_features=True):
        if get_features:
            assert(self.feature_layer_indices is not None and len(self.feature_layer_indices) > 0)
            features = []
            if -1 in self.feature_layer_indices:
                features.append(x)
            x = self.dropout_channel(x.unsqueeze(3)).squeeze(3)
            x = self.dropout_temporal(x.unsqueeze(3).transpose(1, 2)).squeeze(3).transpose(1, 2)
            x, feature = self.encoder(self.conv_in(x), feature_layer_indices=self.feature_layer_indices)
            if feature is not None:
                features.append(feature)
            out = self.conv_out(x)
            if -2 in self.feature_layer_indices:
                features.append(F.softmax(out, 1))
            # return out, self.final_outlinear(torch.cat(features, 1).mean(dim=-1))
            return out, torch.cat(features, 1)
        else:
            x = self.dropout_channel(x.unsqueeze(3)).squeeze(3)
            x = self.dropout_temporal(x.unsqueeze(3).transpose(1, 2)).squeeze(3).transpose(1, 2)
            out = self.conv_out(self.encoder(self.conv_in(x), feature_layer_indices=None))
            return out



class DecoderModel(nn.Module):
    def __init__(self, input_dim, num_classes,
        num_layers, num_f_maps, time_emb_dim, kernel_size, dropout_rate):
        
        super(DecoderModel, self).__init__()

        self.time_emb_dim = time_emb_dim

        self.time_in = nn.ModuleList([
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.Linear(time_emb_dim, time_emb_dim)
        ])

        self.conv_in = nn.Conv1d(num_classes, num_f_maps, 1)
        self.module = MixedConvAttModuleV2(num_layers, num_f_maps, input_dim, kernel_size, dropout_rate, time_emb_dim)
        self.conv_out =  nn.Conv1d(num_f_maps, num_classes, 1)
        
        self.change1 = nn.Linear(32, 2)
        self.change2 = nn.Linear(2, 32)

    def forward(self, x, t, event):

        time_emb = get_timestep_embedding(t, self.time_emb_dim)
        time_emb = self.time_in[0](time_emb)
        time_emb = swish(time_emb)
        time_emb = self.time_in[1](time_emb)
        
        # time_emb: [B, time_emb_dim]
        time_emb = time_emb.unsqueeze(1).repeat(1,x.shape[0]//time_emb.shape[0],1).reshape(-1,self.time_emb_dim)

        fra = self.change1(self.conv_in(event)) # [B*To,num_f_maps=24, 1]
        fra = self.module(fra, x, time_emb)
        event_out = self.change2(self.conv_out(fra))

        return event_out


class MixedConvAttModuleV2(nn.Module): # for decoder
    def __init__(self, num_layers, num_f_maps, input_dim_cross, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConvAttModuleV2, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConvAttentionLayerV2(num_f_maps, input_dim_cross, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  #2 ** i
    
    def forward(self, x, x_cross, time_emb=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:,:,None]

        for layer in self.layers:
            x = layer(x, x_cross)

        return x


class MixedConvAttentionLayerV2(nn.Module):
    
    def __init__(self, d_model, d_cross, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayerV2, self).__init__()
        
        self.d_model = d_model
        self.d_cross = d_cross
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation 
        
        assert(self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)

        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None


    def get_attn_indices(self, l, device):
            
        attn_indices = []
                
        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)  
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)  
            # 3  3  3 ...                             (k=3, //1)          
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)
                        
            indices = [i + self.padding for i in range(s,e,step)]

            attn_indices.append(indices)
        
        attn_indices = np.array(attn_indices)
            
        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)
        
        
    def attention(self, x, x_cross):
        
        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)
                                
        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2],:], (-1,))
        
        x_q = self.att_linear_q(torch.cat([x, x_cross], 1))
        x_k = self.att_linear_k(torch.cat([x, x_cross], 1))
        x_v = self.att_linear_v(x)

        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)  
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0), 
            2, flat_indicies)  
                        
        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1])) 
        
        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)
        
        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2],:] >= self.padding,
            self.attn_indices[:x.shape[2],:] < att.shape[1] + self.padding
        ) # 1 keep, 0 mask
        
        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2)
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)
        
        return r
    
                
    def forward(self, x, x_cross):
        
        x_drop = self.dropout(x)
        x_cross_drop = self.dropout(x_cross)

        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop, x_cross_drop)
                
        out = self.ffn_block(self.norm(out1 + out2))

        return x + out


class MixedConvAttModule(nn.Module): # for encoder
    def __init__(self, num_layers, num_f_maps, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConvAttModule, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConvAttentionLayer(num_f_maps, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  #2 ** i
    
    def forward(self, x, time_emb=None, feature_layer_indices=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:,:,None]

        if feature_layer_indices is None:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            out = []
            for l_id, layer in enumerate(self.layers):
                x = layer(x)
                if l_id in feature_layer_indices:
                    out.append(x)
            
            if len(out) > 0:
                out = torch.cat(out, 1)
            else:
                out = None

            return x, out
    

class MixedConvAttentionLayer(nn.Module):
    
    def __init__(self, d_model, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayer, self).__init__()
        
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation 
        
        assert(self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)
        
        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None


    def get_attn_indices(self, l, device):
            
        attn_indices = []
                
        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)  
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)  
            # 3  3  3 ...                             (k=3, //1)          
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)
                        
            indices = [i + self.padding for i in range(s,e,step)]

            attn_indices.append(indices)
        
        attn_indices = np.array(attn_indices)
            
        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)
        
        
    def attention(self, x):
        
        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)
                                
        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2],:], (-1,))
        
        x_q = self.att_linear_q(x)
        x_k = self.att_linear_k(x)
        x_v = self.att_linear_v(x)
                
        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)  
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0), 
            2, flat_indicies)  
                        
        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1])) 
        
        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)
        
        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2],:] >= self.padding,
            self.attn_indices[:x.shape[2],:] < att.shape[1] + self.padding
        ) # 1 keep, 0 mask
        
        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2) 
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)
        
        return r
    
                
    def forward(self, x):
        
        x_drop = self.dropout(x)
        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop)
        out = self.ffn_block(self.norm(out1 + out2))

        return x + out
    
if __name__ == "__main__":
    # enc = EncoderModel(**params_gtea["encoder_params"])
    # global_cond = torch.randn(4*32,64)
    # global_cond = global_cond[...,None].repeat(1,1,2)
    # print(enc(global_cond)[-1].shape)
    
    dec = DecoderModel(**params_gtea["decoder_params"])
    backbone_feats = torch.randn(2*64, 192, 2)
    times = torch.randint(0,10,(2,))
    sample = torch.randn(2*64, 12, 32)
    print(dec(backbone_feats, times, sample).shape)