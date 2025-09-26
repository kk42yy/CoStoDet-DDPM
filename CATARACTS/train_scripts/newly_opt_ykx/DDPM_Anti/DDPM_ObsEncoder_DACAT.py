import sys
sys.path.append('.')
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import convnext

class CrossAttention(nn.Module):
    def __init__(self, emb_dim=768, in_channels=512, att_dropout=0.0, aropout=0.0):
        super(CrossAttention, self).__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5
 
        # self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)
 
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)
 
        # self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x, context, pad_mask=None):
        '''
        :param x: [batch_size, T=256, C]
        :param context: [batch_szie, T=L, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        # b, c, h, w = x.shape
 
        # x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        # x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]
 
        Q = self.Wq(x)  # [batch_size, T=256, emb_dim=768]
        K = self.Wk(context)  # [batch_size, seq_len=L, emb_dim]
        V = self.Wv(context)
 
        # [batch_size, h*w, seq_len]
        att_weights = torch.einsum('bid,bjd -> bij', Q, K)
        att_weights = att_weights * self.scale
 
        if pad_mask is not None:
            # [batch_size, h*w, seq_len]
            att_weights = att_weights.masked_fill(pad_mask, -1e9)
 
        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bij, bjd -> bid', att_weights, V)   # [batch_size, h*w, emb_dim]
 
        # out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
        # out = self.proj_out(out)   # [batch_size, c, h, w]
 
        # print(out.shape)
 
        return out #, att_weights
    
class Identity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x=None, y=None):
        return x

class ConvNeXt(nn.Module):
    def __init__(self, backbone='convnext', pretrain=False, freeze=False) -> None:
        super().__init__()
        
        if backbone == 'convnext':
            self.featureNet = convnext.convnext_tiny(pretrained=pretrain)
        elif backbone == 'convnextv2':
            self.featureNet = convnext.convnextv2_tiny(pretrained=pretrain)
        self.featureNet.head = nn.Identity()
        self.feature_size = 768
        if freeze:
            for i in [0,1,2]:
                for param in self.featureNet.downsample_layers[i].parameters():
                    param.requires_grad = False
                for param in self.featureNet.stages[i].parameters():
                    param.requires_grad = False
                    
    def forward(self, x):
        for i in range(4):
            x = self.featureNet.downsample_layers[i](x)
            x = self.featureNet.stages[i](x)

        x = x.permute(0,2,3,1)
        x = self.featureNet.norm(x) # global average pooling, (N, C, H, W) -> (N, C)
        x = x.permute(0,3,1,2)
        # x = self.featureNet.norm(x.mean([-2, -1]))
        return x
    
class LinearOutput(nn.Module):
    def __init__(self, in_channel=64, out_channel=64, flatten=False) -> None:
        super().__init__()
        self.flatten = flatten
        self.linear = nn.Linear(in_channel, out_channel, bias=True)
        
    def forward(self, x):
        if self.flatten:
            x = torch.flatten(x, start_dim=1, end_dim=-1)
        return self.linear(x)

class LSTMHead(nn.Module):
	def __init__(self,feature_size,lstm_size=512):

		super(LSTMHead, self).__init__()

		self.lstm = nn.LSTM(feature_size,lstm_size,batch_first=True)

		self.hidden_state = None
		self.prev_feat = None

	def forward(self,x):

		x, hidden_state = self.lstm(x,self.hidden_state)
		self.hidden_state = tuple(h.detach() for h in hidden_state)

		return x

	def forward_wohidden(self,x):

		x, _ = self.lstm(x)
		return x

	def reset(self):

		self.hidden_state = None
		self.prev_feat = None
  
class ObsEncoder_LSTM(nn.Module):
    def __init__(self,
                 # ResNet18
                 pretrain=True,
                 To=32,
                 # LSTM and Linear Output
                 lstm_size=512,
                 out_channels=64,
                 flatten=False,
                 opts=None,
                 trainee=True,
                 ) -> None:
        super().__init__()
        # Long Net: pretrained BNPitfalls (ConvNeXt V2)
        self.cnn_long = ConvNeXt(opts.backbone, pretrain=pretrain, freeze=opts.freeze)
        
        # Short Net: ConvNeXt + LSTM + CA + CA LSTM
        ## cnn_short, lstm load Long Net param when training
        self.cnn_short = ConvNeXt(opts.backbone, pretrain=pretrain, freeze=opts.freeze)
        self.lstm = LSTMHead(self.cnn_short.feature_size,lstm_size)
        self.out_layer = nn.Linear(lstm_size, opts.num_classes) # 在DDPM_ForBackward_DACAT调用
        self.CA = CrossAttention(emb_dim=768)
        self.CA_prev = CrossAttention(emb_dim=768)
        self.CA_lstm = LSTMHead(self.cnn_short.feature_size,lstm_size)
        self.CA_out_layer = nn.Linear(lstm_size, opts.num_classes) # 在DDPM_ForBackward_DACAT调用

        # load model to Long Net
        # CATARACTS-load BNP
        # long_net_pretrain_path = f'.../BNPitfalls_CATARACTS_Recognition.pth.tar'
        # pretrain_checkp = torch.load(long_net_pretrain_path, 'cpu')['state_dict']
        # cnnchekp_for_short_long, lstmchekp_for_short, output_layer = {}, {}, {}
        # for k, v in pretrain_checkp.items():
        #     if 'out_layer' in k:
        #         if 'temporal_head' in k:
        #             output_layer[k[24:]] = v
        #     elif k.startswith('cnn'):
        #         cnnchekp_for_short_long[k[4:]] = v
        #     elif k.startswith('temporal_head'):
        #         lstmchekp_for_short[k[14:]] = v
        
        # CATARACTS-load BNP-DDPM
        long_net_pretrain_path = f'.../BNPitfalls-DDPM_CATARACTS_Recognition.pth.tar'
        pretrain_checkp = torch.load(long_net_pretrain_path, 'cpu')['ema_state_dict']
        cnnchekp_for_short_long, lstmchekp_for_short, output_layer = {}, {}, {}
        
        for k, v in pretrain_checkp.items():
            if 'ConvNeXt_LSTM_outlinear.' in k:
                output_layer[k[len('ConvNeXt_LSTM_outlinear.'):]] = v
            elif k.startswith('obs_encoder.resnet18.'):
                cnnchekp_for_short_long[k[len('obs_encoder.resnet18.'):]] = v
            elif k.startswith('obs_encoder.lstm.'):
                lstmchekp_for_short[k[len('obs_encoder.lstm.'):]] = v
        
        
        
        self.cnn_long.load_state_dict(cnnchekp_for_short_long)
        self.cnn_short.load_state_dict(cnnchekp_for_short_long)
        self.lstm.load_state_dict(lstmchekp_for_short)
        # self.CA_lstm.load_state_dict(lstmchekp_for_short) # OphNet
        self.out_layer.load_state_dict(output_layer)
        # self.CA_out_layer.load_state_dict(output_layer) # OphNet
        self.cnn_long.requires_grad_(False)
        # self.lstm.requires_grad_(False)
        # self.out_layer.requires_grad_(False)
        del cnnchekp_for_short_long, lstmchekp_for_short, output_layer
       
        self.To = To
        self.linearout = LinearOutput(
            in_channel=lstm_size,
            out_channel=out_channels,
            flatten=flatten
        )
        self.linearout_ca = LinearOutput(
            in_channel=lstm_size,
            out_channel=out_channels,
            flatten=flatten
        )
        self.output_shape = out_channels
        self.long_cache = [] # feature cache for DACAT
        self.trainee = trainee
        
    def forward(self, x):
        # 最终的输出写在DDPM_ForBackward_DACAT文件，此处只输出[obs, mean(lstm(x)+lstm_ca(x))]
        # Obs结果为两部分平均
        # x: B*To, 3, H, W -> B*To, 768
        
        long_term = self.extract_image_features_cache(x) # [1, B*To*m, 768]
        x = (self.cnn_short(x).mean([-2, -1]).reshape(1, -1, self.cnn_long.feature_size)) # [1, B*To, 768]
        # x = (self.cnn_short(x).reshape(1, -1, self.cnn_long.feature_size)) # [1, B*To, 768]
        
        ###### 2025.05.25
        long_term = self.CA_prev(long_term, x)
        
        if not self.trainee:
            single_response = (x * long_term).mean(dim=-1) # [1, t], fs_t with f_i
            multi_response =  torch.flip(torch.cumsum(torch.flip(single_response[0],dims=[0]),dim=0), dims=[0]) # the suffix accumulate
            idx = multi_response.max(dim=0)[-1].long()
            # idx2 = multi_response.shape[0] - 64
            # if idx < idx2:
            #     idx = idx2
        else:
            single_response = self.max_suffix_conv(x, long_term) # [1, t], fs_t with f_i
            multi_response =  torch.flip(torch.cumsum(torch.flip(single_response,dims=[0]),dim=0), dims=[0]) # the suffix accumulate
            idx = multi_response.max(dim=0)[-1].long()
        
        xca = self.CA(x, long_term[:,idx:,:])[0] # [B*To, 768]
        xca_lstm = self.CA_lstm(xca) # [B*To, 512]
        x_lstm = self.lstm(x[0]) # [B*To, 512]
        

        # x = (self.linearout(x_lstm) + self.linearout_ca(xca_lstm)) / 2
        x = self.linearout_ca(xca_lstm) # 20241128-2042 使用这个效果最好：lr1e-5,wd1e-2,woema,cnnv2,64,BNPFreezeall
        x = x.reshape(-1, self.output_shape)
        return x, (x_lstm, xca_lstm)
    
    def max_suffix_conv(self, x: torch.Tensor, long_term: torch.Tensor):
        b, Ts, C = x.shape
        _, TL, _ = long_term.shape
        single_response = torch.zeros((TL-Ts+1))
        for idx in range(TL-Ts+1):
            single_response[idx] = (x*long_term[:,idx:idx+Ts,:]).mean().unsqueeze(0)
        return single_response

    def extract_image_features_cache(self,x):
        x = self.cnn_long(x).mean([-2, -1]).reshape(-1, self.cnn_long.feature_size)
        # x = self.cnn_long(x).reshape(-1, self.cnn_long.feature_size)
        self.long_cache.append(x)
        return torch.vstack(self.long_cache).unsqueeze(0)
    
    def long_cache_reset(self):
        self.long_cache.clear()
        
if __name__ == "__main__":
    class opts():
        backbone = 'convnextv2'
        freeze = False
        num_classes = 7
        
    obs_encoder = ObsEncoder_LSTM(
            pretrain=False,
            # input_shape=(512,7,12), # resnet18
            # num_kp=32,
            # temperature=1.0,
            # noise_std=0.0,
            # out_channels=64,
            # flatten=True
            opts=opts,
            trainee=False
        )
    
    x = torch.randn(4*2,3,216,384)
    print(obs_encoder(x)[0].shape)