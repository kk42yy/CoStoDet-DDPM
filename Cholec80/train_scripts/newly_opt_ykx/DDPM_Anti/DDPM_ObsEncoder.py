import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as vision_models
import convnext
import numpy as np

class ResNet18Conv(nn.Module):
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super(ResNet18Conv, self).__init__()
        net = vision_models.resnet18(pretrained=pretrained)

        if input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))
    
    def forward(self, x):
        return self.nets(x) # [B*To, 3, 216, 384] -> [B*To, 512, 7, 12]
  
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
        return x

                        
    
class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.

    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """
    def __init__(
        self,
        input_shape=(512,7,12),
        num_kp=32,
        temperature=1.,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self._in_w),
                np.linspace(-1., 1., self._in_h)
                )
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial 
        probability distribution is created using a softmax, where the support is the 
        pixel locations. This distribution is used to compute the expected value of 
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.

        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(), feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints
    
class LinearOutput(nn.Module):
    def __init__(self, in_channel=64, out_channel=64, flatten=False) -> None:
        super().__init__()
        self.flatten = flatten
        self.linear = nn.Linear(in_channel, out_channel, bias=True)
        
    def forward(self, x):
        if self.flatten:
            x = torch.flatten(x, start_dim=1, end_dim=-1)
        return self.linear(x)
    
class ObsEncoder(nn.Module):
    def __init__(self,
                 # ResNet18
                 pretrain=False,
                 # Spatial Softmax
                 input_shape=(512, 7, 12),
                 num_kp=32,
                 temperature=1.0,
                 noise_std=0.0,
                 # Linear Output
                 out_channels=64,
                 flatten=True
                 ) -> None:
        super().__init__()
        # self.resnet18 = ResNet18Conv(pretrained=pretrain)
        self.resnet18 = ConvNeXt(pretrain=pretrain, freeze=True)
        self.spatial = SpatialSoftmax(
            input_shape=input_shape,
            num_kp=num_kp,
            temperature=temperature,
            noise_std=noise_std
        )
        self.linearout = LinearOutput(
            in_channel=2*num_kp,
            out_channel=out_channels,
            flatten=flatten
        )
        self.output_shape = out_channels
        
    def forward(self, x):
        return self.linearout(
            self.spatial(
                self.resnet18(x)
            )
        ), None

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
                 opts=None
                 ) -> None:
        super().__init__()
        self.resnet18 = ConvNeXt(pretrain=pretrain, freeze=opts.freeze)
        self.To = To
        
        self.lstm = LSTMHead(self.resnet18.feature_size,lstm_size)
        self.linearout = LinearOutput(
            in_channel=lstm_size,
            out_channel=out_channels,
            flatten=flatten
        )
        self.output_shape = out_channels
        
    def forward(self, x):
        
        # x = self.resnet18(x).mean([-2, -1]).reshape(-1, self.To, self.resnet18.feature_size)
        x = self.resnet18(x).mean([-2, -1]).reshape(1, -1, self.resnet18.feature_size)
        # x = self.lstm.forward_wohidden(x)
        lstmx = self.lstm(x)
        x = self.linearout(lstmx)
        x = x.reshape(-1, self.output_shape)
        return x, lstmx
        
if __name__ == "__main__":
    obs_encoder = ObsEncoder_LSTM(
            pretrain=False,
            # input_shape=(512,7,12), # resnet18
            input_shape=(768,6,12),
            num_kp=32,
            temperature=1.0,
            noise_std=0.0,
            out_channels=64,
            flatten=True
        )
    
    x = torch.randn(4*2,3,216,384)
    print(obs_encoder(x).shape)