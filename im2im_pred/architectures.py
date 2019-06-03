import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from custom_layers import AttentionBlockEncoder, EncoderBlock, DecoderBlock, AttentionBlockDecoder, AttentionBlock
from gradient_logger import create_last_conv_hook_at, add_grad_hook, last_conv
from resnet_unet import unet_learner_without_skip_connections


class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

    def model_fit(self, x_pred1, x_output1, x_pred2, x_output2, x_pred3, x_output3):
        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(x_output2, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(self.device)

        # semantic loss: depth-wise cross entropy
        x_output1 = x_output1.to(self.device)
        loss1 = F.nll_loss(x_pred1, x_output1, ignore_index=-1)
        x_output1.to('cpu')

        # depth loss: l1 norm
        x_output2 = x_output2.to(self.device)
        loss2 = torch.sum(torch.abs(x_pred2 - x_output2) * binary_mask) / torch.nonzero(binary_mask).size(0)
        x_output2.to('cpu')

        # normal loss: dot product
        x_output3 = x_output3.to(self.device)
        loss3 = 1 - torch.sum((x_pred3 * x_output3) * binary_mask) / torch.nonzero(binary_mask).size(0)
        x_output3.to('cpu')

        return torch.stack([loss1, loss2, loss3])

    def compute_miou(self, x_pred, x_output):
        x_output = x_output.to(self.device)

        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            true_class = 0
            first_switch = True
            for j in range(self.class_nb):
                pred_mask = torch.eq(x_pred_label[i], torch.full(x_pred_label[i].shape, j, dtype=torch.long, device=self.device))
                true_mask = torch.eq(x_output_label[i], torch.full(x_output_label[i].shape, j, dtype=torch.long, device=self.device))
                mask_comb = pred_mask + true_mask
                union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
                intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
                if union == 0:
                    continue
                if first_switch:
                    class_prob = intsec / union
                    first_switch = False
                else:
                    class_prob = intsec / union + class_prob
                true_class += 1
            if i == 0:
                batch_avg = class_prob / true_class
            else:
                batch_avg = class_prob / true_class + batch_avg

        x_output = x_output.to('cpu')
        return batch_avg / batch_size

    def compute_iou(self, x_pred, x_output):
        x_output = x_output.to(self.device)

        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            if i == 0:
                pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                                      torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))
            else:
                pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
                    torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor)))

        x_output = x_output.to('cpu')
        return pixel_acc / batch_size

    def depth_error(self, x_pred, x_output, rmse=True):
        x_output = x_output.to(self.device)

        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(self.device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)

        if not rmse:
            abs_err = torch.abs(x_pred_true - x_output_true)
        else:
            abs_err = torch.sqrt((x_pred_true - x_output_true)**2)
        rel_err = abs_err / x_output_true

        x_output = x_output.to('cpu')
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

    def normal_error(self, x_pred, x_output):
        x_output = x_output.to(self.device)

        binary_mask = (torch.sum(x_output, dim=1) != 0)
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)

        x_output = x_output.to('cpu')
        return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)

    def write_gradient_loggers(self):
        for gradient_logger in self.gradient_loggers:
            gradient_logger.write_grad_metrics()

    def update_gradient_loggers(self, epoch):
        for gradient_logger in self.gradient_loggers:
            gradient_logger.update_grad_metrics(epoch)

    def forward(self, x):
        pass

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

    def gradient_logger_hooks(self, grad_save_path): pass

    def last_shared_layer(self):
        pass


class SegNetWithoutAttention(MultiTaskModel):
    def __init__(self, device):
        super().__init__()

        self.device = device

        # initialise network parameters
        self.filter = [3, 64, 128, 256, 512, 512]
        self.class_nb = 13

        self.n_tasks = 3

        # add batchnorm -> in paper they are just in the task specific parts
        # he uses it: see conv_layer

        # define pooling and unpooling functions
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # define encoder decoder layers

        # encoder

        self.encoder = nn.ModuleList()

        down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        for i in range(len(self.filter) - 1):
            self.encoder.append(
                EncoderBlock(self.filter[i], self.filter[i+1], additional_conv_layer=True if i > 1 else False,
                             down_sampling=None if i == 0 else down_sampling)
            )
        self.encoder.append(down_sampling)

        # decoder

        dec_filter = self.filter[::-1]
        dec_filter[-1] = 64

        self.decoder = nn.ModuleList()

        for i in range(len(dec_filter) - 1):
            self.decoder.append(
                DecoderBlock(dec_filter[i], dec_filter[i+1], additional_conv_layer=True if i < 3 else False)
            )

        filter = self.filter[1:]
        filter.append(filter[-1])

        output_sizes = (self.class_nb, 1, 3)
        if len(output_sizes) != self.n_tasks:
            raise ValueError('Number of output_sizes does not match tasks.')

        self.pred_task = nn.ModuleList()
        for output_size in output_sizes:
            self.pred_task.append(self.conv_layer([filter[0], output_size], pred=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.to(self.device)

    def forward(self, x):

        pool_indices = []
        for i, layer in enumerate(self.encoder[:-1]):
            x = layer(x)
            if type(x) is tuple and len(x) == 2:
                pool_indices.append(x[1])
                x = x[0]
        x, pool_ind = self.encoder[-1](x)
        pool_indices.append(pool_ind)

        pool_indices = pool_indices[::-1]
        for i, layer in enumerate(self.decoder):
            x = layer(x, pool_indices[i])
        del pool_indices

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task[0](x), dim=1)
        t2_pred = self.pred_task[1](x)
        t3_pred = self.pred_task[2](x)
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], self.logsigma

    def gradient_logger_hooks(self, grad_save_path):

        self.grad_save_path = grad_save_path
        self.gradient_loggers = []

        def gradient_logger_hooks_encoder_decoder(enc_dec, block_name):

            def name(i, ind):
                return '{}_block_{}'.format(block_name, i)

            i = 0
            for layer in enc_dec:
                if isinstance(layer, EncoderBlock) or isinstance(layer, DecoderBlock):
                    add_grad_hook(layer, self.n_tasks,
                                  name(i, None),
                                  self.grad_save_path, self.gradient_loggers)
                    i += 1

        gradient_logger_hooks_encoder_decoder(self.encoder, 'encoder')
        gradient_logger_hooks_encoder_decoder(self.decoder, 'decoder')

    def last_shared_layer(self):
        return last_conv(self.decoder)

    def grad_norm_hook(self):
        self.grad_norms_last_shared_layer = []

        def append_grad_norm(module, grad_input, grad_output):
            self.grad_norms_last_shared_layer.append(grad_input[1].norm())

        self.last_shared_layer().register_backward_hook(append_grad_norm)


class SegNet(MultiTaskModel):
    def __init__(self, device, grad_hook_at_axon=True):
        super(SegNet, self).__init__()

        self.device = device
        self.grad_hook_at_axon = grad_hook_at_axon

        # initialise network parameters
        self.filter = [3, 64, 128, 256, 512, 512]
        self.class_nb = 13

        self.n_tasks = 3

        # add batchnorm -> in paper they are just in the task specific parts
        # he uses it: see conv_layer

        # define pooling and unpooling functions
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # define encoder decoder layers

        # encoder

        self.encoder = nn.ModuleList()

        down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        for i in range(len(self.filter) - 1):
            self.encoder.append(
                EncoderBlock(self.filter[i], self.filter[i+1], additional_conv_layer=True if i > 1 else False,
                             down_sampling=None if i == 0 else down_sampling)
            )
        self.encoder.append(down_sampling)

        # decoder

        dec_filter = self.filter[::-1]
        dec_filter[-1] = 64

        self.decoder = nn.ModuleList()

        for i in range(len(dec_filter) - 1):
            self.decoder.append(
                DecoderBlock(dec_filter[i], dec_filter[i+1], additional_conv_layer=True if i < 3 else False)
            )

        # define task attention layers

        filter = self.filter[1:]
        filter.append(filter[-1])
        for i in range(len(self.encoder) - 1):
            self.encoder[i] = AttentionBlockEncoder(self.encoder[i], filter[i], filter[i+1], self.n_tasks,
                                                    first_block=True if i == 0 else False)

        for i in range(len(self.decoder)):
            self.decoder[i] = AttentionBlockDecoder(self.decoder[i], dec_filter[i], dec_filter[i+1], self.n_tasks,
                                                    index_intermediate=0)

        output_sizes = (self.class_nb, 1, 3)
        if len(output_sizes) != self.n_tasks:
            raise ValueError('Number of output_sizes does not match tasks.')

        self.pred_task = nn.ModuleList()
        for output_size in output_sizes:
            self.pred_task.append(self.conv_layer([filter[0], output_size], pred=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.to(self.device)

    def forward(self, x):

        pool_indices = []
        x_task_specific = None
        for i, layer in enumerate(self.encoder[:-1]):
            x, x_task_specific = layer(x, input_task_specific=x_task_specific,
                                           index_intermediate=0 if i == 0 else 1)
            if type(x) is tuple and len(x) == 2:
                pool_indices.append(x[1])
                x = x[0]
        x, pool_ind = self.encoder[-1](x)
        pool_indices.append(pool_ind)

        pool_indices = pool_indices[::-1]
        for i, layer in enumerate(self.decoder):
            x, x_task_specific = layer((x, pool_indices[i]), x_task_specific)
        del pool_indices
        del x

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task[0](x_task_specific[0]), dim=1)
        t2_pred = self.pred_task[1](x_task_specific[0])
        t3_pred = self.pred_task[2](x_task_specific[0])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], self.logsigma

    def gradient_logger_hooks(self, grad_save_path):

        self.grad_save_path = grad_save_path
        self.gradient_loggers = []

        def gradient_logger_hooks_encoder_decoder(enc_dec, block_name):

            def name(i, ind):
                if self.grad_hook_at_axon:
                    return '{}_block_{}'.format(block_name, i)
                else:
                    return '{}_block_{}_conv_{}'.format(block_name, i, ind)

            i = 0
            for layer in enc_dec:
                if hasattr(layer, 'attented_layer'):
                    if self.grad_hook_at_axon:
                        add_grad_hook(layer.attented_layer, self.n_tasks,
                                                     name(i, None),
                                                     self.grad_save_path, self.gradient_loggers)
                    else:
                        for conv_ind in (0, len(layer.attented_layer.layers) -1):
                            if type(layer.attented_layer.layers[conv_ind]) is nn.MaxPool2d or \
                                    type(layer.attented_layer.layers[conv_ind]) is nn.MaxUnpool2d:
                                conv_ind_ = conv_ind + 1
                            else:
                                conv_ind_ = conv_ind
                            create_last_conv_hook_at(layer.attented_layer.layers[conv_ind_], self.n_tasks,
                                                     name(i, conv_ind),
                                                     self.grad_save_path, self.gradient_loggers)
                    i += 1

        gradient_logger_hooks_encoder_decoder(self.encoder, 'encoder')
        gradient_logger_hooks_encoder_decoder(self.decoder, 'decoder')

    def last_shared_layer(self):
        return last_conv(self.decoder)

# Doesn't work that well, probably too few parameters
class ResNetUnet(MultiTaskModel):
    def __init__(self, device, pretrained=False, skip_connections=True, grad_hook_at_axon=True):
        super().__init__()
        self.device = device
        self.class_nb = 13
        self.n_tasks = 3

        self.grad_hook_at_axon = grad_hook_at_axon


        self.resnet_unet = unet_learner_without_skip_connections(64, self.device,
                                                                 models.resnet34, pretrained=pretrained, metrics=None,
                                                                 skip_connections=skip_connections)

        filter = self.resnet_unet.filter
        attended_layers = self.resnet_unet.attended_layers

        # assert len(filter) == len(attended_layers), \
        #     'Is okay with filter, filter+1 because first layer of encoder is not attended.'

        middle_ind = 6
        enc_filter = filter[:middle_ind]
        enc_filter.append(enc_filter[-1])


        # define task attention layers

        def replace_in(module_list, search_module, replace_module):
            for i in range(len(module_list)):
                if module_list[i] is search_module:
                    module_list[i] = replace_module

        # encoder part
        for i in range(1, middle_ind):   # first layer is just for conversion from 3 to 64 channels
            replace_in(self.resnet_unet.layers,
                       attended_layers[i],
                       AttentionBlockEncoder(attended_layers[i], enc_filter[i], filter[i+1],
                                             self.n_tasks,
                                             first_block=True if i == 1 else False,
                                             downsampling=i < middle_ind-2    # no downsampling in last two attended layers
                                             )
                       )

        # decoder part
        # TODO check notability as this can be done as for segnet decoder blocks
        #  because the problem is that the block has to start at the beginning not end
        for i in range(middle_ind, len(attended_layers)):
            replace_in(self.resnet_unet.layers,
                       attended_layers[i],
                       AttentionBlockDecoder(attended_layers[i], filter[i], filter[i+1],
                                             self.n_tasks, index_intermediate=None if i == middle_ind else 0,
                                             resnet=True, upsampling=i != middle_ind,
                                             last_block_resnet=i == len(attended_layers)-1,
                                             before_last_block_resnet=i==len(attended_layers)-2)
                       )



        output_sizes = (self.class_nb, 1, 3)
        if len(output_sizes) != self.n_tasks:
            raise ValueError('Number of output_sizes does not match tasks.')

        self.pred_task = nn.ModuleList()
        for output_size in output_sizes:
            self.pred_task.append(self.conv_layer([filter[-1], output_size], pred=True))

        # TODO add attention blocks
        self.to(self.device)

        split_enc_dec_ind = 9
        self.encoder = self.resnet_unet.layers[:split_enc_dec_ind]
        self.decoder = self.resnet_unet.layers[split_enc_dec_ind:]

    def forward(self, x):
        x_task_specific = None

        for i, layer in enumerate(self.encoder):
            if not isinstance(layer, AttentionBlock):
                x = layer(x)
            else:
                x, x_task_specific = layer(x, input_task_specific=x_task_specific)

        # encoder works :)

        for i, layer in enumerate(self.decoder):
            x, x_task_specific = layer(x, x_task_specific)
        del x

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task[0](x_task_specific[0]), dim=1)
        t2_pred = self.pred_task[1](x_task_specific[0])
        t3_pred = self.pred_task[2](x_task_specific[0])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], self.logsigma

    def gradient_logger_hooks(self, grad_save_path):

        self.grad_save_path = grad_save_path
        self.gradient_loggers = []

        def gradient_logger_hooks_encoder_decoder(enc_dec, block_name):

            def name(i):
                return '{}_block_{}_conv_last'.format(block_name, i)

            i = 0
            for layer in enc_dec:
                if hasattr(layer, 'attented_layer'):
                    create_last_conv_hook_at(layer.attented_layer, self.n_tasks,
                                             name(i),
                                             self.grad_save_path, self.gradient_loggers)
                    i += 1

        gradient_logger_hooks_encoder_decoder(self.encoder, 'encoder')
        gradient_logger_hooks_encoder_decoder(self.decoder, 'decoder')


# TODO still has to be transformed to Unet like architecture
class WideResNet(MultiTaskModel):
    def __init__(self, num_classes, depth=28, widen_factor=4): # params from paper
        super(WideResNet, self).__init__()
        self.in_planes = 16
        n = int((depth - 4) / 6)
        filter = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # this is just the encoder part
        # TODO: have to be transformed to some kind of u-net as fastai did
        self.conv1 = self.conv3x3(3, filter[0], stride=1)
        self.layer1 = self._wide_layer(WideResNet.wide_basic, filter[1], n, stride=2)
        self.layer2 = self._wide_layer(WideResNet.wide_basic, filter[2], n, stride=2)
        self.layer3 = self._wide_layer(WideResNet.wide_basic, filter[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(filter[3], momentum=0.9)


        # TODO: remove linear layer because its no classification
        # self.linear = nn.ModuleList([nn.Sequential(
        #     nn.Linear(filter[3], num_classes[0]),
        #     nn.Softmax(dim=1))])

        # attention modules
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])

        self.n_tasks = 3

        for j in range(self.n_tasks):
            if j < self.n_tasks-1:   # because he already started the list with one entry
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
            for i in range(3):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))

        for i in range(3):
            if i < 2:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel):
        conv_block = nn.Sequential(
            nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=channel[1]),
            nn.ReLU(inplace=True),
        )
        return conv_block

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

    def conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, k):
        # TODO k picks out one dataset (many-to-many was just single task but with one network with multiple datasets)
        # -> do it all at once

        g_encoder = [0] * 4

        atten_encoder = [0] * self.n_tasks
        for i in range(self.n_tasks):
            atten_encoder[i] = [0] * 4
        for i in range(self.n_tasks):
            for j in range(4):
                atten_encoder[i][j] = [0] * 3

        g_encoder[0] = self.conv1(x)
        g_encoder[1] = self.layer1(g_encoder[0])
        g_encoder[2] = self.layer2(g_encoder[1])
        g_encoder[3] = F.relu(self.bn1(self.layer3(g_encoder[2])))

        # apply attention modules on each of the 4 encoder conv blocks (this part is like segnet)
        # but just at the final output of (first conv, wide layer 1, wide layer 2, wide layer 3 + BN + ReLU)
        for j in range(4):
            if j == 0:  # at the beginning there is nothing to concat with
                atten_encoder[k][j][0] = self.encoder_att[k][j](g_encoder[0])
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[0]
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1])
                atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2)
            else:
                atten_encoder[k][j][0] = self.encoder_att[k][j](
                    torch.cat((g_encoder[j], atten_encoder[k][j - 1][2]), dim=1))
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[j]
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1])
                if j < 3:   # at the end there is not max_pool but avg_pool2d instead -> probably for the linear layers
                    atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2)

        pred = F.avg_pool2d(atten_encoder[k][-1][-1], 8)
        pred = pred.view(pred.size(0), -1)

        out = self.linear[k](pred)
        return out

    class wide_basic(nn.Module):
        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
                )

        def forward(self, x):
            out = self.conv1(F.relu(self.bn1(x)))
            out = self.conv2(F.relu(self.bn2(out)))
            out += self.shortcut(x)

            return out
