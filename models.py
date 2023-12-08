import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


#################################
#           Encoder
#################################


class Encoder(nn.Module):
    def __init__(self, in_channels=1, dim=64, n_residual=3, n_downsample=2, style_dim=8):
        super(Encoder, self).__init__()
        self.content_encoder = ContentEncoder(in_channels, dim, n_residual, n_downsample)
        self.style_encoder = StyleEncoder(in_channels, dim, n_downsample, style_dim)

    def forward(self, x):
        content_code = self.content_encoder(x)
        style_code = self.style_encoder(x)
        return content_code, style_code


#################################
#            Decoder
#################################


class Decoder(nn.Module):
    def __init__(self, out_channels=1, dim=64, n_residual=3, n_upsample=2, style_dim=8):
        super(Decoder, self).__init__()

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="adain")]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.ConvTranspose1d(dim, dim // 2, 4, stride=2),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer
        # layers += [nn.ConvTranspose1d(dim, out_channels, 7, stride=1), nn.Tanh()]
        layers += [nn.ConvTranspose1d(dim, out_channels, 7, stride=1)]
        # layers += [nn.ConvTranspose1d(dim, out_channels, 7, stride=1), nn.Sigmoid()]

        self.model = nn.Sequential(*layers)

        # Initiate mlp (predicts AdaIN parameters)
        num_adain_params = self.get_num_adain_params()
        # print(num_adain_params,'num_adain_params') 256*12=3072
        self.mlp = MLP(style_dim, num_adain_params)

    def get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            # if m.__class__.__name__ == "InstanceNorm1d":
                num_adain_params += 2 * m.num_features
                # print( m.num_features,'m.num_features')  256
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
                # Extract mean and std predictions
                # print(m.num_features, 'm.num_features')
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features : 2 * m.num_features]
                # print(mean.shape,'mean.shape') [1,256]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                # print(m.bias.shape,'m.bias.shape') [256]
                m.weight = std.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features :]

    def forward(self, content_code, style_code):
        # Update AdaIN parameters by MLP prediction based off style code
        # print(style_code.shape, 'style_code.shape') [1,8,1]
        self.assign_adain_params(self.mlp(style_code))
        img = self.model(content_code)
        return img


#################################
#        Content Encoder
#################################


class ContentEncoder(nn.Module):
    def __init__(self, in_channels=1, dim=64, n_residual=3, n_downsample=2):
        super(ContentEncoder, self).__init__()

        # Initial convolution block
        layers = [
            nn.Conv1d(in_channels, dim, 7, stride=1),
            nn.InstanceNorm1d(dim),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv1d(dim, dim * 2, 4, stride=2),
                nn.InstanceNorm1d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="in")]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


#################################
#        Style Encoder
#################################


class StyleEncoder(nn.Module):
    def __init__(self, in_channels=1, dim=64, n_downsample=2, style_dim=8):
        super(StyleEncoder, self).__init__()

        # Initial conv block
        layers = [nn.Conv1d(in_channels, dim, 7, stride=4),
                  nn.InstanceNorm1d(dim),
                  nn.ReLU(inplace=True),
                  ]

        # Downsampling
        for _ in range(2):
            layers += [nn.Conv1d(dim, dim * 2, 7, stride=2), nn.ReLU(inplace=True)]
            dim *= 2

        # Downsampling with constant depth
        for _ in range(n_downsample - 2):
            layers += [nn.Conv1d(dim, dim, 7, stride=2), nn.ReLU(inplace=True)]

        # Average pool and output layer
        layers += [nn.AdaptiveAvgPool1d(1), nn.Conv1d(dim, style_dim, 1, 1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


######################################
#   MLP (predicts AdaIn parameters)
######################################


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ="relu"):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # print(self.model(x.view(x.size(0), -1)).shape, 'mlp') [1,3072]
        return self.model(x.view(x.size(0), -1))


##############################
#        Discriminator
##############################


class MultiDiscriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv1d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            return layers

        # Extracts three discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv1d(512, 1, 3, 1, padding=1)
                ),
            )

        self.downsample = nn.AvgPool1d(in_channels, stride=2, count_include_pad=False)

    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        # print(x.shape, 'compute_loss')
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            # print(x.shape,'x')
            outputs.append(m(x))
            # print(m(x).shape, 'm(x)') [1,1,300] [1,1,150] [1,1,75]
            x = self.downsample(x)
            # print(x.shape, 'Discriminator') [1,1,2400] [1,1,1200] [1,1,600]
        return outputs


##############################
#       Custom Blocks
##############################


class ResidualBlock(nn.Module):
    def __init__(self, features, norm="in"):
        super(ResidualBlock, self).__init__()

        norm_layer = AdaptiveInstanceNorm1d if norm == "adain" else nn.InstanceNorm1d
        # conv_layer = nn.ConvTranspose1d if norm == "adain" else nn.Conv1d
        conv_layer = nn.Conv1d
        self.block = nn.Sequential(
            conv_layer(features, features, 7, stride=1, padding=3),
            norm_layer(features),
            nn.ReLU(inplace=True),
            conv_layer(features, features, 7, stride=1, padding=3),
            norm_layer(features),
        )

    def forward(self, x):
        return x + self.block(x)


##############################
#        Custom Layers
##############################


class AdaptiveInstanceNorm1d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c,  w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c,  w)

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c,  w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.view(x.size(0), -1).mean(1).shape)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        # print(x.shape,'x.shape') [1,128,2396]
        # print(mean.shape,'mean.shape') [1,1,1]
        # print(std.shape,'std.shape') [1,1,1]
        x = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
