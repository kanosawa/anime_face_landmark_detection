import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


# Cacaded Face Alignment
class CFA(nn.Module):
    def __init__(self, output_channel_num, checkpoint_name=None):
        super(CFA, self).__init__()

        self.output_channel_num = output_channel_num
        self.stage_channel_num = 128
        self.stage_num = 2

        self.features = nn.Sequential(
            nn.Conv2d(  3,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d( 64,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d( 64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),

            # nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))

            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))
        
        self.CFM_features = nn.Sequential(
            #nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, self.stage_channel_num, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        # cascaded regression
        stages = [self.make_stage(self.stage_channel_num)]
        for _ in range(1, self.stage_num):
            stages.append(self.make_stage(self.stage_channel_num + self.output_channel_num))
        self.stages = nn.ModuleList(stages)
        
        # initialize weights
        if checkpoint_name:
            snapshot = torch.load(checkpoint_name)
            self.load_state_dict(snapshot['state_dict'])
        else:
            self.load_weight_from_dict()
    

    def forward(self, x):
        feature = self.features(x)
        feature = self.CFM_features(feature)
        heatmaps = [self.stages[0](feature)]
        for i in range(1, self.stage_num):
            heatmaps.append(self.stages[i](torch.cat([feature, heatmaps[i - 1]], 1)))
        return heatmaps
    

    def make_stage(self, nChannels_in):
        layers = []
        layers.append(nn.Conv2d(nChannels_in, self.stage_channel_num, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(4):
            layers.append(nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(self.stage_channel_num, self.output_channel_num, kernel_size=3, padding=1))
        return nn.Sequential(*layers)


    def load_weight_from_dict(self):
        model_urls = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        weight_state_dict = model_zoo.load_url(model_urls)
        all_parameter = self.state_dict()
        all_weights   = []
        for key, value in all_parameter.items():
            if key in weight_state_dict:
                all_weights.append((key, weight_state_dict[key]))
            else:
                all_weights.append((key, value))
        all_weights = OrderedDict(all_weights)
        self.load_state_dict(all_weights)
