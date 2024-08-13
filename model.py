import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import SyncBatchNorm


class BasicBlock(nn.Module):

    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=SyncBatchNorm):

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, previous_dilation,
                               dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:

            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=SyncBatchNorm):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride,
                               dilation, dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class DenoisedNLOutput(nn.Module):
    def __init__(self, in_plane, mid_plane, num_classes, norm_layer=SyncBatchNorm):
        super(DenoisedNLOutput, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_plane, mid_plane, 3, stride=1, padding=1, bias=False),
                                  norm_layer(mid_plane),
                                  nn.ReLU(),
                                  nn.Dropout2d(0.05, False),
                                  nn.Conv2d(mid_plane, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class LocalRelation(nn.Module):
    def __init__(self):
        super(LocalRelation, self).__init__()
        self.kernel = 3
        self.sigmoid = nn.Sigmoid()
        self.unfold = nn.Unfold(kernel_size=[3, 3], dilation=[1, 1], padding=1, stride=[1, 1])

    def forward(self, q, k):
        batch_size, c, height, width = q.size()
        q = q.permute(0, 2, 3, 1).contiguous().view(-1, 1, c)
        k_patches = self.unfold(k)  # [b, c*k*k, L]
        k_patches = k_patches.view(batch_size, c, -1, height*width).permute(0, 3, 1, 2).contiguous().view(-1, c, self.kernel**2)

        energy = torch.matmul(q, k_patches)
        local_relation = self.sigmoid(energy)
        return local_relation


class DNL(nn.Module):
    def __init__(self, plane, num_classes, norm_layer=SyncBatchNorm):
        super(DNL, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(plane, plane//8, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(plane, plane//8, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(plane, plane, 1, stride=1, padding=0, bias=False)
        self.conv = nn.Sequential(nn.Conv2d(plane, plane, 3, stride=1, padding=1, bias=False),
                                  norm_layer(plane),
                                  nn.ReLU())
        self.conv_out3 = DenoisedNLOutput(1024, 512, num_classes)

        self.local = LocalRelation()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.softmax_pred = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.unfold = nn.Unfold(kernel_size=[3, 3], dilation=[1, 1], padding=1, stride=[1, 1])

    def forward(self, x, x2):
        batch_size, _, height, width = x.size()

        q = self.conv1(x)
        k = self.conv2(x)
        v = self.conv3(x)
        local_relation = self.local(q, k)  # [bhw, 1, kk]

        q = q.view(batch_size, -1, height * width).permute(0, 2, 1).contiguous()
        k = k.view(batch_size, -1, height * width)
        v = v.view(batch_size, -1, height * width).permute(0, 2, 1).contiguous()
        energy = torch.matmul(q, k)
        relation = self.softmax(energy)  # [b, hw, hw]

        # global rectify
        aux2 = self.conv_out3(x2)
        pred = F.interpolate(aux2, (height, width), mode='bilinear', align_corners=True)
        pred = self.softmax_pred(pred).view(batch_size, self.num_classes, -1)
        relation = relation * torch.sigmoid(torch.matmul(pred.permute(0, 2, 1).contiguous(), pred))

        # local retention
        relation = self.unfold(relation.view(batch_size, height * width, height, width))  # [b, hw*k*k, hw]
        relation = relation.view(batch_size, height * width, -1, height * width)
        local_relation = local_relation.view(batch_size, height * width, -1).permute(0, 2, 1).contiguous()  # [b, kk, hw]
        aug_relation = relation * torch.unsqueeze(local_relation, 1)  # [b, hw, kk, hw]
        aug_relation = torch.sum(aug_relation, dim=2)

        aug = torch.bmm(aug_relation, v).view(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        out = self.gamma * aug + x
        out = F.interpolate(out, (2*height, 2*width), mode='bilinear', align_corners=True)
        out = self.conv(out)
        return out, aux2


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=19, dilated=True, deep_stem=True, norm_layer=SyncBatchNorm,
                 multi_grid=True, multi_dilation=[4, 8, 16]):

        self.inplanes = 128 if deep_stem else 64
        super(ResNet, self).__init__()

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2, 1, bias=False),
                norm_layer(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                norm_layer(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False)
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)

        if dilated:
            if multi_grid:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                               dilation=2, norm_layer=norm_layer)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer,
                                               multi_grid=multi_grid, multi_dilation=multi_dilation)
            else:
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                               dilation=2, norm_layer=norm_layer)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                               dilation=4, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)

    def load_pretrained(self, path):
        pre_trained = torch.load(path, map_location='cpu')
        state_dict = self.state_dict()
        for key, weights in pre_trained.items():
            if key in state_dict:
                state_dict[key].copy_(weights)

        self.load_state_dict(state_dict)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None, multi_grid=False,
                    multi_dilation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        if multi_grid == False:
            layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                                previous_dilation=dilation, norm_layer=norm_layer))
        else:
            layers.append(block(self.inplanes, planes, stride, dilation=multi_dilation[0],
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        if multi_grid:
            div = len(multi_dilation)
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=multi_dilation[i % div], previous_dilation=dilation,
                                    norm_layer=norm_layer))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        h, w = x.size()[2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x1 = self.layer2(x)
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)

        return x1, x2, x3
        

class DenoisedNL(nn.Module):
    def __init__(self, block, layers, num_classes=19, dilated=True, deep_stem=True, norm_layer=SyncBatchNorm,
                 multi_grid=True, multi_dilation=[4, 8, 16]):

        self.inplanes = 128 if deep_stem else 64
        super(DenoisedNL, self).__init__()

        self.backbone = ResNet(block, layers, num_classes=num_classes, dilated=dilated, deep_stem=deep_stem, norm_layer=SyncBatchNorm,
                               multi_grid=multi_grid, multi_dilation=multi_dilation)

        self.conv = nn.Sequential(nn.Conv2d(2048, 512, 3, stride=1, padding=1, bias=False),
                                  norm_layer(512),
                                  nn.ReLU())
        self.interp = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
                                    norm_layer(512),
                                    nn.ReLU())
        self.down_sample = BasicBlock(512, 512, stride=2, downsample=self.interp)

        self.dnl = DNL(512, num_classes)

        self.conv_out1 = DenoisedNLOutput(2048 + 512, 512, num_classes)
        self.conv_out2 = DenoisedNLOutput(512, 64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
            elif isinstance(m, SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h, w = x.size()[2:]

        x1, x2, x3 = self.backbone(x)

        feats = self.conv(x3)
        feats = self.down_sample(feats)

        dnl_feats, aux2 = self.dnl(feats, x2)

        dnl_feats = self.conv_out1(torch.cat([x3, dnl_feats], 1))
        aux1 = self.conv_out2(x1)

        output = F.interpolate(dnl_feats, (h, w), mode='bilinear', align_corners=True)
        aux1 = F.interpolate(aux1, (h, w), mode='bilinear', align_corners=True)
        aux2 = F.interpolate(aux2, (h, w), mode='bilinear', align_corners=True)

        outs = [output, aux1, aux2]

        return outs


def compile_model(num_classes, pre_train=True, path='./pretrained_ckpt/resnet101-deep.pth'):
    model = DenoisedNL(Bottleneck, [3, 4, 23, 3], num_classes)

    if pre_train:
        model.backbone.load_pretrained(path)

    return model


