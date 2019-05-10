from fastai.imports import torch
from torch import nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):

    def __init__(self, attented_layer, task_specific_output_list, shared_feature_extractor):
        super().__init__()
        self.attented_layer = attented_layer
        self.task_specific_output_list = task_specific_output_list
        self.shared_feature_extractor = shared_feature_extractor

    def forward(self, input_trunk, input_task_specific=None):
        if input_task_specific is not None:
            self.task_specific_output_list.append(input_task_specific)
        return self.attented_layer(input_trunk)

    def att_layer(self, channel):
        att_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
            nn.BatchNorm2d(channel[2]),
            nn.Sigmoid(),
        )
        return att_block

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=channel[1]),
                nn.ReLU(inplace=True),
            )
        else:
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=channel[0], out_channels=channel[0], kernel_size=3, padding=1),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
            )
        return conv_block


class AttentionBlockEncoder(AttentionBlock):
    def __init__(self, attented_layer, task_specific_output_list, filter, filter_next_block, first_block=False, shared_feature_extractor=None):
        if shared_feature_extractor is None:
            shared_feature_extractor = self.conv_layer([filter, filter_next_block])

        super().__init__(attented_layer, task_specific_output_list, shared_feature_extractor)
        self.first_block = first_block

        if not self.first_block:
            filter = [2*filter, filter, filter]
        else:
            filter = [filter, filter, filter]
        self.specific_feature_extractor = self.att_layer(filter)

    def forward(self, input_trunk, input_task_specific=torch.Tensor([]), index_intermediate=None):

        if not self.first_block and len(input_task_specific) == 0:
            raise ValueError('Is not the first attention block, but has no input from task-specific route.')

        if index_intermediate is None:

            output_trunk = self.attented_layer(input_trunk)

            input_attention = torch.cat((output_trunk, input_task_specific.type_as(output_trunk)), dim=1)

        else:

            output_trunk_intermediate = self.attented_layer[0:index_intermediate](input_trunk)
            output_trunk = self.attented_layer[index_intermediate:](output_trunk_intermediate)

            input_attention = torch.cat((output_trunk_intermediate, input_task_specific.type_as(output_trunk)), dim=1)

        output_attention = self.specific_feature_extractor(input_attention)
        output_attention = (output_attention) * output_trunk
        # encoder_block_att are shared
        output_attention = self.shared_feature_extractor(output_attention)
        output_attention = F.max_pool2d(output_attention, kernel_size=2, stride=2)

        self.task_specific_output_list.append(output_attention)
        return output_trunk


class AttentionBlockDecoder(AttentionBlock):
    def __init__(self, attented_layer, task_specific_output_list, filter, filter_previous_block, shared_feature_extractor=None):
        if shared_feature_extractor is None:
            shared_feature_extractor = self.conv_layer([filter_previous_block, filter])

        super().__init__(attented_layer, task_specific_output_list, shared_feature_extractor)

        filter = [filter_previous_block + filter, filter, filter]
        self.specific_feature_extractor = self.att_layer(filter)

    def forward(self, input_trunk, input_task_specific, index_intermediate=None):

        if index_intermediate is None:

            output_trunk = self.attented_layer(input_trunk)

            input_attention = torch.cat((output_trunk, input_task_specific.type_as(output_trunk)), dim=1)

        else:

            output_trunk_intermediate = self.attented_layer[0:index_intermediate](input_trunk)
            output_trunk = self.attented_layer[index_intermediate:](output_trunk_intermediate)

            input_attention = torch.cat((output_trunk_intermediate, input_task_specific.type_as(output_trunk)), dim=1)

        output_attention = F.interpolate(input_task_specific, scale_factor=2, mode='bilinear',
                                               align_corners=True)
        output_attention = self.shared_feature_extractor(output_attention)
        output_attention = self.specific_feature_extractor(input_attention)
        output_attention = output_attention * output_trunk

        self.task_specific_output_list.append(output_attention)
        return output_trunk

