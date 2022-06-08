
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.decode = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.PixelShuffle(upscale_factor=2))

    def forward(self, x):
        return self.decode(x)

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[32, 64, 128, 256], fpn_out=32):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1) for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] * (len(feature_channels) - 1))
        self.conv_fusion = nn.Sequential(nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
                                         nn.BatchNorm2d(fpn_out),
                                         nn.ReLU(inplace=True))

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


class rgb_net(nn.Module):
    def __init__(self, num_classes, filters=32):
        super().__init__()
        self.rgb = resnet50(pretrained=True)

        # decoder
        self.dec5 = DecoderBlock(2048, filters * 16)

        self.dec4 = DecoderBlock(2048 + filters * 4, filters * 16)
        self.dec3 = DecoderBlock(1024 + filters * 4, filters * 8)
        self.dec2 = DecoderBlock(512 + filters * 2, filters * 4)
        self.dec1 = DecoderBlock(256 + filters * 1, filters * 2)

    def forward(self, rgb):
        rgb0 = self.rgb.conv1(rgb)
        rgb0 = self.rgb.bn1(rgb0)
        rgb0 = self.rgb.relu(rgb0)
        rgb0 = self.rgb.maxpool(rgb0)

        rgb1 = self.rgb.layer1(rgb0)
        rgb2 = self.rgb.layer2(rgb1)
        rgb3 = self.rgb.layer3(rgb2)
        rgb4 = self.rgb.layer4(rgb3)

        dec5 = self.dec5(nn.functional.max_pool2d(rgb4, kernel_size=2, stride=2))

        dec4 = self.dec4(torch.cat((rgb4, dec5), dim=1))
        dec3 = self.dec3(torch.cat((rgb3, dec4), dim=1))
        dec2 = self.dec2(torch.cat((rgb2, dec3), dim=1))
        dec1 = self.dec1(torch.cat((rgb1, dec2), dim=1))

        return dec1, dec2, dec3, dec4

class nnn_net(nn.Module):
    def __init__(self, num_classes, filters=32):
        super().__init__()
        self.nnn = resnet50(pretrained=True)

        # self.nnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.nnn.conv1.weight.data = torch.unsqueeze(torch.mean(self.nnn.conv1.weight.data, dim=1), dim=1)

        # decoder
        self.dec5 = DecoderBlock(2048, filters * 16)

        self.dec4 = DecoderBlock(2048 + filters * 4, filters * 16)
        self.dec3 = DecoderBlock(1024 + filters * 4, filters * 8)
        self.dec2 = DecoderBlock(512 + filters * 2, filters * 4)
        self.dec1 = DecoderBlock(256 + filters * 1, filters * 2)

    def forward(self, nnn):
        nnn0 = self.nnn.conv1(nnn)
        nnn0 = self.nnn.bn1(nnn0)
        nnn0 = self.nnn.relu(nnn0)
        nnn0 = self.nnn.maxpool(nnn0)

        nnn1 = self.nnn.layer1(nnn0)
        nnn2 = self.nnn.layer2(nnn1)
        nnn3 = self.nnn.layer3(nnn2)
        nnn4 = self.nnn.layer4(nnn3)

        dec5 = self.dec5(nn.functional.max_pool2d(nnn4, kernel_size=2, stride=2))

        dec4 = self.dec4(torch.cat((nnn4, dec5), dim=1))
        dec3 = self.dec3(torch.cat((nnn3, dec4), dim=1))
        dec2 = self.dec2(torch.cat((nnn2, dec3), dim=1))
        dec1 = self.dec1(torch.cat((nnn1, dec2), dim=1))

        return dec1, dec2, dec3, dec4


class MSNet(nn.Module):
    def __init__(self, num_classes):
        super(MSNet, self).__init__()
        self.rgb = rgb_net(num_classes)
        self.nnn = nnn_net(num_classes)

        self.FPN = FPN_fuse([32, 64, 128, 256], 32)

        self.fuse = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

    def forward(self, rgbnnd):
        input_size = (rgbnnd.size()[2], rgbnnd.size()[3])
        rgb = rgbnnd[:, :3]
        nnn = rgbnnd[:, 3:]

        rgb_dec1, rgb_dec2, rgb_dec3, rgb_dec4 = self.rgb(rgb)
        nnn_dec1, nnn_dec2, nnn_dec3, nnn_dec4 = self.nnn(nnn)

        dec1 = torch.cat((rgb_dec1, nnn_dec1), dim=1)
        dec2 = torch.cat((rgb_dec2, nnn_dec2), dim=1)
        dec3 = torch.cat((rgb_dec3, nnn_dec3), dim=1)
        dec4 = torch.cat((rgb_dec4, nnn_dec4), dim=1)

        features = [dec1, dec2, dec3, dec4]
        fpn = self.FPN(features)

        # x = self.fuse(fpn)
        # x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        x = F.interpolate(fpn, size=input_size, mode='bicubic', align_corners=True)
        x = self.fuse(x)

        return x
