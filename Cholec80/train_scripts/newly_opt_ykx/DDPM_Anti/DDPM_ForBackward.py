from curses import newwin
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from scipy.ndimage import gaussian_filter1d
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from .DDMP_Norm import LinearNormalizer
from .DDPM_ConditionUNet import ConditionalUnet1D
from .DDPM_ConditionUNet_DivideTime import ConditionalUnet1D as LSTMConditionalUnet1D
from .DDPM_ObsEncoder import ObsEncoder, ObsEncoder_LSTM
from .DDPM_utils import LowdimMaskGenerator, replace_submodules


class DiffusionUnetHybridImagePolicy(nn.Module):
    def __init__(self, 
            action_dim=5+7, # 5, 5 tools + 7 phase
            noise_scheduler=DDPMScheduler,
            horizon=16, # 16, input 16 frames
            n_action_steps=8, # predict 8 frames
            n_obs_steps=2, # use 2 for predict
            num_train_timesteps=100,
            num_inference_steps=100,
            prediction_type='epsilon',
            obs_as_global_cond=True,
            diffusion_step_embed_dim=128,
            down_dims=(512,1024,2048),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=True,
            opts=None,
            loadpretrain=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
                
        # LSTM Obs
        if opts.Obs == 'LSTM':
            obs_encoder = ObsEncoder_LSTM(
                pretrain=True,
                To=n_obs_steps,
                lstm_size=512,
                out_channels=64,
                flatten=False,
                opts=opts
            )
        else:
            # ori obs
            obs_encoder = ObsEncoder(
                pretrain=True,
                # input_shape=(512,7,12), # resnet18
                input_shape=(768,6,12),
                num_kp=32,
                temperature=1.0,
                noise_std=0.0,
                out_channels=64,
                flatten=True
            )
        
        self.CNN_output_loss = opts.CNN_output_loss
        if self.CNN_output_loss:
            self.ConvNeXt_LSTM_outlinear = nn.Linear(512, 7)
            weight = torch.Tensor([
                1.6411019141231247,
                0.19090963801041133,
                1.0,
                0.2502662616859295,
                1.9176363911137977,
                0.9840248158200853,
                2.174635818337618,
            ]).cuda()
            self.criterion_phase = nn.CrossEntropyLoss(reduction='mean',weight=weight)
        else:
            self.ConvNeXt_LSTM_outlinear = nn.Identity()
        
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps
        
        if opts.Obs == "LSTM":
            model = LSTMConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
                To=n_obs_steps,
            )
        else:
            model = ConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale,
            )

        self.obs_encoder = obs_encoder
        self.model = model
        
        if noise_scheduler == DDPMScheduler:
            self.noise_scheduler = noise_scheduler(
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                beta_start=0.0001,
                clip_sample=True,
                num_train_timesteps=num_train_timesteps,
                prediction_type="epsilon",
                variance_type="fixed_small",
            )
        elif noise_scheduler == DDIMScheduler:
            self.noise_scheduler = noise_scheduler(
                num_train_timesteps=num_train_timesteps,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon" # or sample
            )
            
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim, # 2
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps, # 2
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.opts = opts

        if num_inference_steps is None:
            num_inference_steps = 1000
        self.num_inference_steps = num_inference_steps
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        
        ### 2024.10.25 UNIFORM ###
        # range_width = 1 * (12 ** 0.5)
        # a = -range_width / 2
        # b = range_width / 2
        # trajectory = torch.empty(
        #                     size=condition_data.shape, 
        #                     dtype=condition_data.dtype,
        #                     device=condition_data.device,
        #                     # generator=generator
        #                 ).uniform_(a, b)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, nobs, To_use_for_lstm=None) -> Dict[str, torch.Tensor]:
        """
        data, target from batch
        result: must include "action" key
        """
        # normalize input
        # nobs = self.normalizer.normalize(obs_dict)
        # value = next(iter(nobs.values()))
        B, To = nobs.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps if To_use_for_lstm is None else To_use_for_lstm
        
        # build input
        device = nobs.device
        dtype = nobs.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = nobs[:,:To,...].reshape(-1,*nobs.shape[2:])
            nobs_features, lstm_features = self.obs_encoder(this_nobs) # 20241012 for LSTM output
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            
            if self.opts.Obs == 'LSTM':
                #### 241104 last 4 frame ####
                cond_data = torch.zeros(size=(B, To, Da, 32), device=device, dtype=dtype)
            else:
                cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
                
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = nobs[:,:To,...].reshape(-1,*nobs.shape[2:])
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True
        # # run sampling
        # nsample = self.conditional_sample(
        #     cond_data, 
        #     cond_mask,
        #     local_cond=local_cond,
        #     global_cond=global_cond,
        #     **self.kwargs)
        
        # # unnormalize prediction
        # action_pred = nsample[...,:Da,-1] #### 241104 ####
        # # action_pred = self.normalizer['action'].unnormalize(naction_pred)
        
        # Combine with CNN
        if self.CNN_output_loss:

            outputp = self.ConvNeXt_LSTM_outlinear(lstm_features[0])
            outputp = outputp.reshape(B, To, -1)
            # outputp = outputp.transpose(1, 2) # b, 7, horizon
            action_pred = outputp


        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        # self.normalizer.load_state_dict(normalizer.state_dict())
        pass

    def compute_loss(self, nobs, nactions, phase_nactions=None):
        # normalize input
        # nobs = self.normalizer.normalize(nobs)
        # nactions = self.normalizer['action'].normalize(nactions)
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        
        # 将PhaseGT转换为One-hot，并加噪
        
        trajectory = nactions
        trajectory = F.one_hot(nactions.long(), num_classes=self.opts.num_classes).permute(0, 2, 1)
        soft_event_gt = torch.clone(trajectory).float().cpu().numpy()
        for i in range(soft_event_gt.shape[1]):
            soft_event_gt[0,i] = gaussian_filter1d(soft_event_gt[0,i], 1.0)
        trajectory = torch.from_numpy(soft_event_gt).to(nactions.device)
        trajectory = trajectory.permute(0, 2, 1) # b, seq_len, 7
        
        if not self.opts.CHE:
            self.gt_cache = torch.cat([self.gt_cache, trajectory], dim=1)
        
        
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            # this_nobs = nobs[:,:self.n_obs_steps,...].reshape(-1,*nobs.shape[2:]) # [B*To,3,96,96]
            this_nobs = nobs[:,:,...].reshape(-1,*nobs.shape[2:]) # [B*To,3,96,96]
            nobs_features, lstm_features = self.obs_encoder(this_nobs) # [B*To,64]
            
            ### LSTM 241003 ###########
            if self.opts.CHE and self.opts.Obs == 'LSTM':
                self.obs_encoder.lstm.reset()

            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1) # [B,66*To]
        else:
            # reshape B, T, ... to B*T
            this_nobs = nobs.reshape(-1, *nobs.shape[2:])
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

                
        """
        ## 20241104 将每一帧及其前三帧进行联合预测, trajectory：[B, T32, 12, 4]
        
        trajectory_cur = trajectory.clone() # [b, T, 12]
        trajectory_cur_minus1 = trajectory.clone()
        trajectory_cur_minus2 = trajectory.clone()
        trajectory_cur_minus3 = trajectory.clone()
        
        trajectory_cur_minus1[:,1:,:] = trajectory[:,:-1,:]
        
        trajectory_cur_minus2[:,0:2,:] = trajectory_cur_minus2[:,0:1,:]
        trajectory_cur_minus2[:,2:,:] = trajectory[:,:-2,:]
        
        trajectory_cur_minus3[:,0:3,:] = trajectory_cur_minus3[:,0:1,:]
        trajectory_cur_minus3[:,3:,:] = trajectory[:,:-3,:]
        
        trajectory = torch.cat([
            trajectory_cur_minus3[..., None],
            trajectory_cur_minus2[..., None],
            trajectory_cur_minus1[..., None],
            trajectory_cur[..., None]
        ], dim=-1)
        del trajectory_cur, trajectory_cur_minus1, trajectory_cur_minus2, trajectory_cur_minus3    
        """
        
        ## 20241105 将每一帧及其前31帧进行联合预测, trajectory：[B, T32, 12, 32]
        ## 因此，我们需(1)提前设置horizon = 64，但是nbos_step设为32，并且(2)需要修改obs只使用最后32帧的obs特征，但是必须从前64开始LSTM
        ## (3)调整ground truth
        
        if self.opts.CHE:
            ## 调整ground truth（trajectory）
            new_trajectory = torch.zeros((trajectory.shape[0], 32, trajectory.shape[2], 32), device=trajectory.device)
            for i in range(1, 33):
                new_trajectory[:,:,:,i-1] = trajectory[:, i:i+32, :]
            trajectory = new_trajectory.clone()
            del new_trajectory
            
            ## 调整obs
            global_cond = nobs_features.reshape(batch_size, -1, 64)[:, -32:, :]
            global_cond = global_cond.reshape(batch_size, -1) # [B,64*To]
        else:# CHT 不需要更新前32，只需要从gt_cache里抽取gt，第一组的前32帧可能有重复
            ## 调整ground truth（trajectory）
            new_trajectory = torch.zeros((trajectory.shape[0], trajectory.shape[1], trajectory.shape[2], 32), device=trajectory.device)
            L = self.gt_cache.shape[1]
            SEQLen = trajectory.shape[1]
            for idxi, idxl in enumerate(range(L-32, L)):
                new_trajectory[:,:,:,idxi] = self.gt_cache[:, idxl-SEQLen:idxl, :]
            trajectory = new_trajectory.clone()
            del new_trajectory

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long() # [B] \in [0,100)
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        
        # # generate impainting mask
        # condition_mask = self.mask_generator(trajectory.shape, trajectory.device) # [B, T, 2]
        
        # # compute loss mask
        # loss_mask = ~condition_mask

        # # apply conditioning
        # noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        # loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        if self.CNN_output_loss:
            outputp = self.ConvNeXt_LSTM_outlinear(lstm_features[0])
            outputp = outputp.reshape(batch_size, horizon, -1)
            outputp = outputp.transpose(1, 2) # b, 7, horizon
            loss += self.criterion_phase(outputp, nactions)
        
        return loss

    def gt_cache_reset(self, target): # only for CHT
        self.gt_cache = target[0].clone()
        self.gt_cache = torch.zeros(1, 32,).fill_(self.gt_cache[0]).to(self.gt_cache.device)
        self.gt_cache = F.one_hot(self.gt_cache.long(), num_classes=self.opts.num_classes).permute(0, 2, 1)
            
        soft_event_gt = torch.clone(self.gt_cache).float().cpu().numpy()
        for i in range(soft_event_gt.shape[1]):
            soft_event_gt[0,i] = gaussian_filter1d(soft_event_gt[0,i], 1.0)
        self.gt_cache = torch.from_numpy(soft_event_gt).to(target.device)
        self.gt_cache = self.gt_cache.permute(0, 2, 1) # b, seq_len, 7
        
if __name__ == "__main__":
    net = DiffusionUnetHybridImagePolicy()
    x = torch.randn(2,16,3,216,384)
    y = torch.randn(2,16,5)
    print(net.compute_loss(x,y))
    # for k, v in net.predict_action(x).items():
    #     print(k, v.shape)