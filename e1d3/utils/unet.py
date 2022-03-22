import torch
from torch import nn

class UnetArchitecture3d(nn.Module):

    def __init__(self, config = None):
        super(UnetArchitecture3d, self).__init__()
        if config is not None:
            config_data = config['data']
            num_channels = len(config_data.get('channels'))
        else:
            num_channels = 4

        # Encoder
        # Level 1
        self.conv_enc_1_1 = Conv3dNormAct(num_channels, 32, kernel_size = 3, padding = 1)
        self.conv_enc_1_2 = Conv3dNormAct(32, 32, kernel_size = 3, padding = 1)

        # Level 2
        self.downsampleby_2 = nn.Conv3d(32, 32, kernel_size = 3, stride = 2, padding = 1)
        self.conv_enc_2_1 = Conv3dNormAct(32, 64, kernel_size = 3, padding = 1)
        self.conv_enc_2_2 = Conv3dNormAct(64, 64, kernel_size = 3, padding = 1)

        # Level 3
        self.downsampleby_3 = nn.Conv3d(64, 64, kernel_size = 3, stride = 2, padding = 1)
        self.conv_enc_3_1 = Conv3dNormAct(64, 128, kernel_size = 3, padding = 1)
        self.conv_enc_3_2 = Conv3dNormAct(128, 128, kernel_size = 3, padding = 1)

        # Level 4
        self.downsampleby_4 = nn.Conv3d(128, 128, kernel_size = 3, stride = 2, padding = 1)
        self.conv_enc_4_1 = Conv3dNormAct(128, 256, kernel_size = 3, padding = 1)
        self.conv_enc_4_2 = Conv3dNormAct(256, 256, kernel_size = 3, padding = 1)

        # Level 5
        self.downsampleby_5 = nn.Conv3d(256, 256, kernel_size = 3, stride = 2, padding = 1)
        self.conv_enc_5_1 = Conv3dNormAct(256, 320, kernel_size = 3, padding = 1)
        self.conv_enc_5_2 = Conv3dNormAct(320, 320, kernel_size = 3, padding = 1)

        # Decoder:
        # Level 5
        self.conv_dec_5_1 = Conv3dNormAct(320 + 320, 320, kernel_size = 3, padding = 1)
        self.conv_dec_5_2 = Conv3dNormAct(320, 256, kernel_size = 3, padding = 1)

        # Level 4
        self.upsample_4 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.conv_dec_4_1 = Conv3dNormAct(256 + 256, 256, kernel_size=3, padding=1)
        self.conv_dec_4_2 = Conv3dNormAct(256, 128, kernel_size=3, padding=1)

        # Level 3
        self.upsample_3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.conv_dec_3_1 = Conv3dNormAct(128 + 128, 128, kernel_size=3, padding=1)
        self.conv_dec_3_2 = Conv3dNormAct(128, 64, kernel_size=3, padding=1)

        # Level 2
        self.upsample_2 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.conv_dec_2_1 = Conv3dNormAct(64 + 64, 64, kernel_size=3, padding=1)
        self.conv_dec_2_2 = Conv3dNormAct(64, 32, kernel_size=3, padding=1)

        # Level 1
        self.upsample_1 = nn.ConvTranspose3d(32, 32, kernel_size=2, stride=2)
        self.conv_dec_1_1 = Conv3dNormAct(32 + 32, 32, kernel_size=3, padding=1)
        self.conv_dec_1_2 = Conv3dNormAct(32, 16, kernel_size=3, padding=1)
        self.conv = nn.Conv3d(16, 2, kernel_size=1)

    def forward(self, x_1):
        """"""

        ######################### Encoder:
        # Level 1
        x_1 = self.conv_enc_1_1(x_1)
        x_1 = self.conv_enc_1_2(x_1)

        # Level 2
        x_2 = self.downsampleby_2(x_1)
        x_2 = self.conv_enc_2_1(x_2)
        x_2 = self.conv_enc_2_2(x_2)

        # Level 3
        x_3 = self.downsampleby_3(x_2)
        x_3 = self.conv_enc_3_1(x_3)
        x_3 = self.conv_enc_3_2(x_3)

        # Level 4
        x_4 = self.downsampleby_4(x_3)
        x_4 = self.conv_enc_4_1(x_4)
        x_4 = self.conv_enc_4_2(x_4)

        # Level 5
        x_5 = self.downsampleby_5(x_4)
        x_5 = self.conv_enc_5_1(x_5)
        x_5 = self.conv_enc_5_2(x_5)

        ######################### Decoder:
        ############### Level 5
        # Concatenation & Processing
        x_path = self.conv_dec_5_2(x_5)

        ############### Level 4
        # Upsampling
        x_path = self.upsample_4(x_path)
        # Concatenation & Processing
        x_path = self.conv_dec_4_2(
            self.conv_dec_4_1(
                torch.cat(
                    (x_path, x_4)
                    , dim=1)
            )
        )

        ############### Level 3
        # Upsampling
        x_path = self.upsample_3(x_path)
        # Concatenation & Processing
        x_path = self.conv_dec_3_2(
            self.conv_dec_3_1(
                torch.cat(
                    (x_path, x_3)
                    , dim=1)
            )
        )
        
        ############### Level 2
        # Upsampling
        x_path = self.upsample_2(x_path)
        # Concatenation & Processing
        x_path = self.conv_dec_2_2(
            self.conv_dec_2_1(
                torch.cat(
                    (x_path, x_2)
                    , dim=1)
            )
        )
        
        ############### Level 1
        # Upsampling
        x_path = self.upsample_1(x_path)
        # Concatenation & Processing
        x_path = self.conv_dec_1_2(
            self.conv_dec_1_1(
                torch.cat(
                    (x_path, x_1)
                    , dim=1)
            )
        )
        
        # Classification
        x_path = self.conv(x_path)

        return x_path

    def print_model_parameters(self):
        """Helper to print out model parameters"""
        for param_tensor in self.state_dict():
            print(param_tensor, '\t', self.state_dict()[param_tensor].size())
        print("Total Parameters:", sum(param.numel() for param in self.parameters()))
        
class Conv3dNormAct(nn.Module):
    """Convolution3d -> Norm3d -> Activation"""

    def __init__(self, in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                 dilation=(1, 1, 1)):
        super(Conv3dNormAct, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)
        self.norm = torch.nn.InstanceNorm3d(num_features=out_channels, affine=True)
        self.acti = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        """"""
        return self.acti(self.norm(self.conv(x)))

if __name__ == '__main__':
    net = UnetArchitecture3d().cuda().train()
    net.print_model_parameters()

    x = torch.randn(3, 4, 96, 96, 96).cuda()
    y = net(x)
    print(y.shape)
    loss = y.sum()
    loss.backward()