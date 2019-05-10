import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from torchvision import models

from gradient_logger import GradientLogger, create_last_conv_hook_at
from resnet_unet import unet_learner_without_skip_connections


class MultiTaskModel(nn.Module):
    def __init__(self):
        super().__init__()

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

        return [loss1, loss2, loss3]

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

    def depth_error(self, x_pred, x_output):
        x_output = x_output.to(self.device)

        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(self.device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true

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


class SegNet(MultiTaskModel):
    def __init__(self, device):
        super(SegNet, self).__init__()

        self.device = device

        # initialise network parameters
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13

        # add batchnorm -> in paper they are just in the task specific parts
        # he uses it: see conv_layer

        # define encoder decoder layers
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i + 1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1], filter[i]]))

        # define convolution layer
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i], filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([filter[i + 1], filter[i + 1]]),
                                                         self.conv_layer([filter[i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([filter[i], filter[i]]),
                                                         self.conv_layer([filter[i], filter[i]])))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        for j in range(3):
            if j < 2:   # these are the first that don't concat
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):  # these are from the second that concat and therefore have double the input channels
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)

        # define pooling and unpooling functions
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))

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

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))

        # define attention list for tasks
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in range(2))

        # define global shared network
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])

        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])

        # define task dependent attention module
        for i in range(3):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][0])
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    # encoder_block_att are shared
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](torch.cat((g_encoder[j][0], atten_encoder[i][j - 1][2]), dim=1))
                    atten_encoder[i][j][1] = (atten_encoder[i][j][0]) * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][j][2], kernel_size=2, stride=2)

            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i][-1][-1], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i][j - 1][2], scale_factor=2, mode='bilinear', align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch.cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = (atten_decoder[i][j][1]) * g_decoder[j][-1]

        # define task prediction layers
        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]), dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)

        return [t1_pred, t2_pred, t3_pred], self.logsigma

    def gradient_logger_hooks(self, grad_save_path):
        self.grad_save_path = grad_save_path
        self.gradient_loggers = []

        def gradient_logger_hooks_encoder_decoder(enc_dec_block, conv_block_enc_dec, block_name):

            def name():
                return '{}_block_{}_conv_{}'.format(block_name, i, conv_ind)

            for i in range(len(enc_dec_block)):
                conv_ind = 0

                create_last_conv_hook_at(enc_dec_block[i][0], 3, name(), self.grad_save_path, self.gradient_loggers)
                conv_ind += 1

                if not type(conv_block_enc_dec[i][0]) is nn.Conv2d:
                    for conv_block in conv_block_enc_dec[i]:
                        create_last_conv_hook_at(conv_block[0], 3, name(), self.grad_save_path,
                                                 self.gradient_loggers)
                        conv_ind += 1
                else:
                    create_last_conv_hook_at(conv_block_enc_dec[i][0], 3, name(), self.grad_save_path,
                                             self.gradient_loggers)
                    conv_ind += 1

        gradient_logger_hooks_encoder_decoder(self.encoder_block, self.conv_block_enc, 'encoder')
        gradient_logger_hooks_encoder_decoder(self.decoder_block, self.conv_block_dec, 'decoder')


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


class ResNetUnet(MultiTaskModel):
    def __init__(self, device, pretrained=False):
        super().__init__()
        self.device = device
        self.class_nb = 13

        self.resnet_unet = unet_learner_without_skip_connections(64, self.device,
                                                                 models.resnet34, pretrained=pretrained, metrics=None)

        filter = self.resnet_unet.filter
        attended_layers = self.resnet_unet.attended_layers

        assert len(filter) == len(attended_layers) + 1, 'In decoder size_i & size_(i+1) needed'

        # define task attention layers
        self.encoder_att = nn.ModuleList()
        self.decoder_att = nn.ModuleList()
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[0]])])

        # j: index of tasks
        for j in range(3):
            self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
            self.decoder_att.append(nn.ModuleList([self.att_layer([2 * filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] + filter[i], filter[i], filter[i]]))

        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))


        # done
        self.pred_task1 = self.conv_layer([filter[-1], self.class_nb], pred=True)
        self.pred_task2 = self.conv_layer([filter[-1], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[-1], 3], pred=True)

        # TODO add attention blocks

    def forward(self, x):
        # TODO use attention blocks
        pass