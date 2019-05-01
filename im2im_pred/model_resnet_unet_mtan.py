import glob
import random
import sys
import gc
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
from fastai.vision import models

from tqdm import tqdm

from create_dataset import *
from torch.autograd import Variable

from datetime import datetime

from gradient_logger import GradientLogger
from resnet_unet import unet_learner_without_skip_connections

torch.backends.cudnn.benchmark = True   # may speed up training if input sizes do not vary

seed = 43
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--weight', default='dwa', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='data/nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
opt = parser.parse_args()


def inspect_gpu_tensors():
    import gc
    l = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                l.append((type(obj), obj.size()))
        except:
            pass
    return l

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
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
            if j < 2:
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

    def model_fit(self, x_pred1, x_output1, x_pred2, x_output2, x_pred3, x_output3):
        # binary mark to mask out undefined pixel space
        binary_mask = (torch.sum(x_output2, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(device)

        # semantic loss: depth-wise cross entropy
        x_output1 = x_output1.to(device)
        loss1 = F.nll_loss(x_pred1, x_output1, ignore_index=-1)
        x_output1.to('cpu')

        # depth loss: l1 norm
        x_output2 = x_output2.to(device)
        loss2 = torch.sum(torch.abs(x_pred2 - x_output2) * binary_mask) / torch.nonzero(binary_mask).size(0)
        x_output2.to('cpu')

        # normal loss: dot product
        x_output3 = x_output3.to(device)
        loss3 = 1 - torch.sum((x_pred3 * x_output3) * binary_mask) / torch.nonzero(binary_mask).size(0)
        x_output3.to('cpu')

        return [loss1, loss2, loss3]

    def compute_miou(self, x_pred, x_output):
        x_output = x_output.to(device)

        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            true_class = 0
            first_switch = True
            for j in range(self.class_nb):
                pred_mask = torch.eq(x_pred_label[i], Variable(j * torch.ones(x_pred_label[i].shape).type(torch.LongTensor).to(device)))
                true_mask = torch.eq(x_output_label[i], Variable(j * torch.ones(x_output_label[i].shape).type(torch.LongTensor).to(device)))
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
        x_output = x_output.to(device)

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
        x_output = x_output.to(device)

        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true

        x_output = x_output.to('cpu')
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

    def normal_error(self, x_pred, x_output):
        x_output = x_output.to(device)

        binary_mask = (torch.sum(x_output, dim=1) != 0)
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)

        x_output = x_output.to('cpu')
        return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)

    def gradient_logger_hooks(self, grad_save_path):
        self.grad_save_path = grad_save_path
        self.gradient_loggers = []

        def append_and_return(gradient_logger):
            self.gradient_loggers.append(gradient_logger)
            return self.gradient_loggers[-1]

        def gradient_logger_hooks_encoder_decoder(enc_dec_block, conv_block_enc_dec, block_name):
            for i in range(len(enc_dec_block)):
                conv_ind = 0
                name = '{}_block_{}_conv_{}'.format(block_name, i, conv_ind)
                enc_dec_block[i][0].register_backward_hook(
                    append_and_return(GradientLogger(3, name, self.grad_save_path)).update_grad_list)
                conv_ind += 1

                if not type(conv_block_enc_dec[i][0]) is nn.Conv2d:
                    for conv_block in conv_block_enc_dec[i]:
                        conv_block[0].register_backward_hook(
                            append_and_return(GradientLogger(3, name, self.grad_save_path)).update_grad_list)
                        conv_ind += 1
                else:
                    conv_block_enc_dec[i][0].register_backward_hook(
                        append_and_return(GradientLogger(3, name, self.grad_save_path)).update_grad_list)
                    conv_ind += 1

        gradient_logger_hooks_encoder_decoder(self.encoder_block, self.conv_block_enc, 'encoder')
        gradient_logger_hooks_encoder_decoder(self.decoder_block, self.conv_block_dec, 'decoder')

    def write_gradient_loggers(self, epoch):
        for gradient_logger in self.gradient_loggers:
            gradient_logger.write_grad_metrics(epoch)





# define model, optimiser and scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cleanup_gpu_memory_every_batch = True

pretrained = False
resnet_unet = unet_learner_without_skip_connections(64, device, models.resnet34, pretrained=pretrained, metrics=None)

optimizer = optim.Adam(resnet_unet.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

no_debug = False
gettrace = getattr(sys, 'gettrace', None)
if gettrace is not None:
    if gettrace():
        import shutil

        log_every_nth = 1

        run_number = 'debug'
        name_model_run = 'mtan_segnet_run_{}'.format(run_number)
        if os.path.exists('./{}'.format(name_model_run)):
            shutil.rmtree('./{}'.format(name_model_run))
    else:
        no_debug = True
else:
    no_debug = True
if no_debug:
    log_every_nth = 10

    name_model_run = 'mtan_segnet_run_'
    run_number = len(glob.glob('./{}[0-9]*'.format(name_model_run)))
    name_model_run += str(run_number)
os.makedirs('./logs/{}'.format(name_model_run))

print(name_model_run)
sys.exit(0)

# add gradient logger hooks
SegNet_MTAN.gradient_logger_hooks('./{}/gradient_logs/'.format(name_model_run))


# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}\n'.format(count_parameters(SegNet_MTAN),
                                                           count_parameters(SegNet_MTAN)/24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30\n')

# define dataset path
dataset_path = opt.dataroot
nyuv2_train_set = NYUv2(root=dataset_path, train=True)
nyuv2_test_set = NYUv2(root=dataset_path, train=False)

batch_size = 2  # is the current max for segnet on 12gb gpu
nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    num_workers=2,
    shuffle=True)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    num_workers=2,
    shuffle=False)


# define parameters
total_epoch = 100   # he trained for 200
train_batch = len(nyuv2_train_loader)
test_batch = len(nyuv2_test_loader)
T = opt.temp
avg_cost = np.zeros([total_epoch, 24], dtype=np.float32)
lambda_weight = np.ones([3, total_epoch])
for epoch in range(total_epoch):
    start_time_epoch = datetime.now()

    index = epoch
    cost = np.zeros(24, dtype=np.float32)
    scheduler.step()

    # apply Dynamic Weight Average
    if opt.weight == 'dwa':
        if index == 0 or index == 1:
            lambda_weight[:, index] = 1.0
        else:
            w_1 = avg_cost[index - 1, 0] / avg_cost[index - 2, 0]
            w_2 = avg_cost[index - 1, 3] / avg_cost[index - 2, 3]
            w_3 = avg_cost[index - 1, 6] / avg_cost[index - 2, 6]
            lambda_weight[0, index] = 3 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            lambda_weight[1, index] = 3 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))
            lambda_weight[2, index] = 3 * np.exp(w_3 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T) + np.exp(w_3 / T))

    # iteration for all batches
    nyuv2_train_dataset = iter(nyuv2_train_loader)
    for k in tqdm(range(train_batch), desc='Training'):
        train_data, train_label, train_depth, train_normal = nyuv2_train_dataset.next()
        train_data = train_data.to(device)
        train_label = train_label.type(torch.LongTensor)

        train_pred, logsigma = SegNet_MTAN(train_data)

        train_loss = SegNet_MTAN.model_fit(train_pred[0], train_label, train_pred[1], train_depth, train_pred[2], train_normal)

        if opt.weight == 'equal' or opt.weight == 'dwa':
            # loss = torch.mean(sum(lambda_weight[i, index] * train_loss[i] for i in range(3)))
            loss = list(lambda_weight[i, index] * train_loss[i] for i in range(3))
        else:
            loss = list(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(3))

        for n, nth_loss in enumerate(loss):
            nth_loss.backward(retain_graph=True)

        optimizer.step()    # creates states for moving average -> more memory than for first batch
        optimizer.zero_grad()

        with torch.no_grad():
            cost[0] = train_loss[0].item()
            cost[1] = SegNet_MTAN.compute_miou(train_pred[0], train_label).item()
            cost[2] = SegNet_MTAN.compute_iou(train_pred[0], train_label).item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = SegNet_MTAN.depth_error(train_pred[1], train_depth)
            cost[4], cost[5] = cost[4].item(), cost[5].item()
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = SegNet_MTAN.normal_error(train_pred[2], train_normal)
            avg_cost[index, :12] += cost[:12] / train_batch

        if cleanup_gpu_memory_every_batch:
            # train_loss = [train_loss[i].detach().cpu() for i in range(len(train_loss))]
            # loss = loss.detach().cpu()
            del train_pred
            del logsigma
            del train_loss
            del loss
            torch.cuda.empty_cache()
            gc.collect()

    # evaluating test data
    with torch.no_grad():  # operations inside don't track history
        nyuv2_test_dataset = iter(nyuv2_test_loader)
        for k in tqdm(range(test_batch), desc='Testing'):
            test_data, test_label, test_depth, test_normal = nyuv2_test_dataset.next()
            test_data, test_label = test_data.to(device),  test_label.type(torch.LongTensor).to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)

            test_pred, _ = SegNet_MTAN(test_data)
            test_loss = SegNet_MTAN.model_fit(test_pred[0], test_label, test_pred[1], test_depth, test_pred[2], test_normal)

            cost[12] = test_loss[0].item()
            cost[13] = SegNet_MTAN.compute_miou(test_pred[0], test_label).item()
            cost[14] = SegNet_MTAN.compute_iou(test_pred[0], test_label).item()
            cost[15] = test_loss[1].item()
            cost[16], cost[17] = SegNet_MTAN.depth_error(test_pred[1], test_depth)
            cost[18] = test_loss[2].item()
            cost[19], cost[20], cost[21], cost[22], cost[23] = SegNet_MTAN.normal_error(test_pred[2], test_normal)

            avg_cost[index, 12:] += cost[12:] / test_batch

    time_elapsed_epoch = datetime.now() - start_time_epoch
    print('Elapsted Minutes: {}'.format(time_elapsed_epoch))
    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
          'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
          .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))

    if index % log_every_nth == 0:
        CHECKPOINT_PATH = Path('./{}/model_checkpoints/'.format(name_model_run))
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)

        checkpoint_name = 'checkpoint_segnet_mtan_epoch_{}'.format(index)
        torch.save({
            'epoch': index,
            'model_state_dict': SegNet_MTAN.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_cost': avg_cost
        }, CHECKPOINT_PATH/checkpoint_name)

        SegNet_MTAN.write_gradient_loggers(index)

