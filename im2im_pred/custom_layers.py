from fastai.imports import torch
from torch import nn
import torch.nn.functional as F


def conv_layer(channel, pred=False):
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

def att_layer(channel):
    att_block = nn.Sequential(
        nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=1, padding=0),
        nn.BatchNorm2d(channel[1]),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=1, padding=0),
        nn.BatchNorm2d(channel[2]),
        nn.Sigmoid(),
    )
    return att_block


class EncoderBlock(nn.Module):

    def __init__(self, filter, filter_next, additional_conv_layer=False,
                 down_sampling=nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)):
        super().__init__()
        seq_modules = []
        if down_sampling is not None:
            seq_modules.append(down_sampling)
        seq_modules.extend([conv_layer([filter, filter_next]), conv_layer([filter_next, filter_next])])
        if additional_conv_layer:
            seq_modules.append(conv_layer([filter_next, filter_next]))
        self.layers = nn.Sequential(*seq_modules)

    def forward(self, input, index_intermediate=None):
        if index_intermediate is None:
            if hasattr(self.layers[0], 'return_indices') and self.layers[0].return_indices:
                    input, pool_indices = self.layers[0](input)
                    return (self.layers[1:](input), pool_indices)

            return self.layers(input)
        else:
            if hasattr(self.layers[0], 'return_indices') and self.layers[0].return_indices:
                    input, pool_indices = self.layers[0](input)
                    output_intermediate = self.layers[1:index_intermediate + 1](input)
                    return (self.layers[index_intermediate + 1:](output_intermediate), pool_indices), output_intermediate

            output_intermediate = self.layers[:index_intermediate+1](input)
            return self.layers[index_intermediate+1:](output_intermediate), output_intermediate


class DecoderBlock(nn.Module):

    def __init__(self, filter, filter_next, additional_conv_layer=False,
                 up_sampling=nn.MaxUnpool2d(kernel_size=2, stride=2)):
        super().__init__()
        seq_modules = [
            up_sampling,
            conv_layer([filter, filter_next]),
            conv_layer([filter_next, filter_next])]
        if additional_conv_layer:
            seq_modules.append(conv_layer([filter_next, filter_next]))
        self.layers = nn.Sequential(*seq_modules)

    def forward(self, input, pool_indices, index_intermediate=None):
        assert type(self.layers[0]) is nn.MaxUnpool2d, \
            'First module in every decoder block has to be unpooling, otherwise change forward method.'

        output = self.layers[0](input, pool_indices)
        if index_intermediate is None:
            output = self.layers[1:](output)
            return output
        else:
            if index_intermediate == 0:
                output_intermediate = output
            else:
                output_intermediate = self.layers[1:index_intermediate+1](output)
            output = self.layers[index_intermediate+1:](output_intermediate)
            return output, output_intermediate


class AttentionBlock(nn.Module):

    def __init__(self, attented_layer, shared_feature_extractor, n_tasks, save_attention_mask=True):
        super().__init__()
        self.attented_layer = attented_layer
        self.shared_feature_extractor = shared_feature_extractor
        self.n_tasks = n_tasks
        self.save_attention_mask = save_attention_mask
        self.attention_mask = None


    def forward(self, input_trunk, input_task_specific=None):
        raise ValueError('Not sensible for Superclass.')

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
    def __init__(self, attented_layer, filter, filter_next_block, n_tasks, first_block=False, downsampling=True):
        shared_feature_extractor = self.conv_layer([filter, filter_next_block])

        super().__init__(attented_layer, shared_feature_extractor, n_tasks)
        self.first_block = first_block

        self.downsampling = downsampling

        if not self.first_block:
            # TODO this is not okay for resnet
            filter = [2*filter, filter, filter]
        else:
            filter = [filter, filter, filter]

        self.specific_feature_extractor = nn.ModuleList([self.att_layer(filter) for _ in range(self.n_tasks)])

    def forward(self, input_trunk, input_task_specific=None, index_intermediate=None):

        if not self.first_block and input_task_specific is None:
            raise ValueError('Is not the first attention block, but has no input from task-specific route.')

        if input_task_specific is None:
            input_task_specific = [torch.Tensor([]) for _ in range(self.n_tasks)]

        if len(input_task_specific) != self.n_tasks:
            raise ValueError('Input from task-specific route not same count as tasks.')

        if index_intermediate is None:
            output_trunk = self.attented_layer(input_trunk)
        else:
            output_trunk, output_trunk_intermediate = self.attented_layer(input_trunk,
                                                                          index_intermediate=index_intermediate)
        if type(output_trunk) is tuple:
            output_trunk_ = output_trunk[0]
        else:
            output_trunk_ = output_trunk

        if index_intermediate is None:
            output_trunk_intermediate = output_trunk_

        output_attentions = []

        for i in range(self.n_tasks):
            input_attention = torch.cat((output_trunk_intermediate,
                                         input_task_specific[i].type_as(output_trunk_intermediate)), dim=1)
            output_attention = self.specific_feature_extractor[i](input_attention)
            if self.save_attention_mask:
                self.attention_mask = output_attention.data.cpu().numpy()
            output_attention = output_attention * output_trunk_
            # encoder_block_att are shared
            output_attention = self.shared_feature_extractor(output_attention)
            if self.downsampling:
                output_attention = F.max_pool2d(output_attention, kernel_size=2, stride=2)
            output_attentions.append(output_attention)

        return output_trunk, tuple(output_attentions)


class AttentionBlockDecoder(AttentionBlock):
    def __init__(self, attented_layer, filter, filter_next, n_tasks, index_intermediate=None, resnet=False, upsampling=True,
                 last_block_resnet=False, before_last_block_resnet=False):
        shared_feature_extractor = self.conv_layer([filter, filter_next])

        super().__init__(attented_layer, shared_feature_extractor, n_tasks)

        self.index_intermediate = index_intermediate
        self.resnet = resnet

        self.upsampling = upsampling
        self.last_block_resnet = last_block_resnet

        # hack for the layer before the last
        self.before_last_block_resnet = before_last_block_resnet
        if self.before_last_block_resnet:
            self.last_block_resnet = True
            filter = 32

        filter = [filter + filter_next if (self.index_intermediate is not None and not self.resnet) or self.last_block_resnet
                  else 2 * filter_next,
                  filter_next, filter_next]
        self.specific_feature_extractor = nn.ModuleList([self.att_layer(filter) for _ in range(self.n_tasks)])

    def forward(self, input_trunk, input_task_specific):

        if len(input_task_specific) != self.n_tasks:
            raise ValueError('Input from task-specific route not same count as tasks.')

        if self.index_intermediate is None:
            if type(input_trunk) is tuple:
                output_trunk = self.attented_layer(*input_trunk)
            else:
                output_trunk = self.attented_layer(input_trunk)

            output_trunk_ = output_trunk
        else:
            if type(input_trunk) is tuple:
                output_trunk, output_trunk_intermediate = self.attented_layer(*input_trunk,
                                                                              index_intermediate=self.index_intermediate)
            else:
                output_trunk, output_trunk_intermediate = self.attented_layer(input_trunk,
                                                                              index_intermediate=self.index_intermediate)

            output_trunk_ = output_trunk_intermediate

        output_attentions = []

        for i in range(self.n_tasks):
            if self.upsampling:
                output_attention = F.interpolate(input_task_specific[i], scale_factor=2, mode='bilinear',
                                                       align_corners=True)
            else:
                output_attention = input_task_specific[i]
            output_attention = self.shared_feature_extractor(output_attention)

            input_attention = torch.cat((output_trunk_, output_attention.type_as(output_trunk_)), dim=1)

            output_attention = self.specific_feature_extractor[i](input_attention)
            if self.save_attention_mask:
                self.attention_mask = output_attention.data.cpu().numpy()
            output_attention = output_attention * output_trunk
            output_attentions.append(output_attention)

        return output_trunk, tuple(output_attentions)

