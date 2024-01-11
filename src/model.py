import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from torchvision.models import vgg16

# Load the pre-trained VGG16 model
model_vgg16 = vgg16(pretrained=True)
for param in model_vgg16.parameters():
    param.requires_grad = False
class VGGConvExtractor(nn.Module):
    def __init__(self, model_vgg16):
        super(VGGConvExtractor, self).__init__()
        self.model_vgg16 = model_vgg16
        self.conv_feature_extractors = self.conv_extractors()

    def conv_extractors(self):
        vgg_features = self.model_vgg16.features
        conv_feature_extractors = nn.ModuleList()
        for layer in vgg_features.children():
            if isinstance(layer, nn.Conv2d):
                conv_feature_extractors.append(nn.Sequential(layer, nn.ReLU()))
        return nn.ModuleList(conv_feature_extractors)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convLayers = VGGConvExtractor(model_vgg16).conv_feature_extractors

        # Down part of UNET
        # Encoder
        for i in range(0, 4, 2):
            self.downs.append(
                nn.Sequential(self.convLayers[i],self.convLayers[i+1])
            )
        for i in range(4, len(self.convLayers)-3, 3):
            self.downs.append(
                nn.Sequential(self.convLayers[i], self.convLayers[i + 1], self.convLayers[i + 2])
            )

        # Up part of UNET
        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
