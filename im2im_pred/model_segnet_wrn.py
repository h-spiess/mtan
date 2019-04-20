from pathlib import Path

import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
from tqdm import tqdm

from create_dataset import *
from torch.autograd import Variable

from datetime import datetime

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


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
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


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        n = int((depth - 4) / 6)
        filter = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = conv3x3(3, filter[0], stride=1)
        self.layer1 = self._wide_layer(wide_basic, filter[1], n, stride=2)
        self.layer2 = self._wide_layer(wide_basic, filter[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, filter[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(filter[3], momentum=0.9)


        # TODO: remove linear layer because its no classification
        # TODO: have to be transformed to some kind of u-net as fastai did
        self.linear = nn.ModuleList([nn.Sequential(
            nn.Linear(filter[3], num_classes[0]),
            nn.Softmax(dim=1))])

        # attention modules
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0], filter[1]])])

        n_tasks = len(num_classes)

        for j in range(n_tasks):
            if j < 9:
                self.encoder_att.append(nn.ModuleList([self.att_layer([filter[0], filter[0], filter[0]])]))
                self.linear.append(nn.Sequential(nn.Linear(filter[3], num_classes[j + 1]),
                                                 nn.Softmax(dim=1)))
            for i in range(3):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1], filter[i + 1], filter[i + 1]]))

        for i in range(3):
            if i < 2:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 2]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1], filter[i + 1]]))

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

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, k):
        # TODO k picks out one dataset (many-to-many was just single task but with one network with multiple datasets)

        g_encoder = [0] * 4

        n_tasks = 3
        atten_encoder = [0] * n_tasks
        for i in range(n_tasks):
            atten_encoder[i] = [0] * 4
        for i in range(n_tasks):
            for j in range(4):
                atten_encoder[i][j] = [0] * 3

        g_encoder[0] = self.conv1(x)
        g_encoder[1] = self.layer1(g_encoder[0])
        g_encoder[2] = self.layer2(g_encoder[1])
        g_encoder[3] = F.relu(self.bn1(self.layer3(g_encoder[2])))

        # apply attention modules
        for j in range(4):
            if j == 0:
                atten_encoder[k][j][0] = self.encoder_att[k][j](g_encoder[0])
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[0]
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1])
                atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2)
            else:
                atten_encoder[k][j][0] = self.encoder_att[k][j](
                    torch.cat((g_encoder[j], atten_encoder[k][j - 1][2]), dim=1))
                atten_encoder[k][j][1] = (atten_encoder[k][j][0]) * g_encoder[j]
                atten_encoder[k][j][2] = self.encoder_block_att[j](atten_encoder[k][j][1])
                if j < 3:
                    atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2], kernel_size=2, stride=2)

        pred = F.avg_pool2d(atten_encoder[k][-1][-1], 8)
        pred = pred.view(pred.size(0), -1)

        out = self.linear[k](pred)
        return out

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
                pred_mask = torch.eq(x_pred_label[i],
                                     Variable(j * torch.ones(x_pred_label[i].shape).type(torch.LongTensor).to(device)))
                true_mask = torch.eq(x_output_label[i], Variable(
                    j * torch.ones(x_output_label[i].shape).type(torch.LongTensor).to(device)))
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
                pixel_acc = pixel_acc + torch.div(
                    torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor)),
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
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), torch.sum(rel_err) / torch.nonzero(
            binary_mask).size(0)

    def normal_error(self, x_pred, x_output):
        x_output = x_output.to(device)

        binary_mask = (torch.sum(x_output, dim=1) != 0)
        error = torch.acos(
            torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)

        x_output = x_output.to('cpu')
        return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)


# define model, optimiser and scheduler
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
cleanup_gpu_memory_every_batch = False
WideResNet_MTAN = WideResNet().to(device)
optimizer = optim.Adam(WideResNet_MTAN.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}\n'.format(count_parameters(WideResNet_MTAN),
                                                           count_parameters(WideResNet_MTAN) / 24981069))
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

        train_pred, logsigma = WideResNet_MTAN(train_data)

        train_loss = WideResNet_MTAN.model_fit(train_pred[0], train_label, train_pred[1], train_depth, train_pred[2], train_normal)

        if opt.weight == 'equal' or opt.weight == 'dwa':
            loss = torch.mean(sum(lambda_weight[i, index] * train_loss[i] for i in range(3)))
        else:
            loss = sum(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(3))

        loss.backward()
        optimizer.step()    # creates states for moving average -> more memory than for first batch
        optimizer.zero_grad()

        with torch.no_grad():
            cost[0] = train_loss[0].item()
            cost[1] = WideResNet_MTAN.compute_miou(train_pred[0], train_label).item()
            cost[2] = WideResNet_MTAN.compute_iou(train_pred[0], train_label).item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = WideResNet_MTAN.depth_error(train_pred[1], train_depth)
            cost[4], cost[5] = cost[4].item(), cost[5].item()
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = WideResNet_MTAN.normal_error(train_pred[2], train_normal)
            avg_cost[index, :12] += cost[:12] / train_batch

        if cleanup_gpu_memory_every_batch:
            train_loss = [train_loss[i].detach().cpu() for i in range(len(train_loss))]
            loss = loss.detach().cpu()
            torch.cuda.empty_cache()
            del train_pred
            del logsigma
            del train_loss
            del loss

    # evaluating test data
    with torch.no_grad():  # operations inside don't track history
        nyuv2_test_dataset = iter(nyuv2_test_loader)
        for k in tqdm(range(test_batch), desc='Testing'):
            test_data, test_label, test_depth, test_normal = nyuv2_test_dataset.next()
            test_data, test_label = test_data.to(device),  test_label.type(torch.LongTensor).to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)

            test_pred, _ = WideResNet_MTAN(test_data)
            test_loss = WideResNet_MTAN.model_fit(test_pred[0], test_label, test_pred[1], test_depth, test_pred[2], test_normal)

            cost[12] = test_loss[0].item()
            cost[13] = WideResNet_MTAN.compute_miou(test_pred[0], test_label).item()
            cost[14] = WideResNet_MTAN.compute_iou(test_pred[0], test_label).item()
            cost[15] = test_loss[1].item()
            cost[16], cost[17] = WideResNet_MTAN.depth_error(test_pred[1], test_depth)
            cost[18] = test_loss[2].item()
            cost[19], cost[20], cost[21], cost[22], cost[23] = WideResNet_MTAN.normal_error(test_pred[2], test_normal)

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

    if index % 10 == 0:
        CHECKPOINT_PATH = Path('./model_checkpoints/')
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)

        checkpoint_name = 'checkpoint_segnet_mtan_epoch_{}'.format(index)
        torch.save({
            'epoch': index,
            'model_state_dict': WideResNet_MTAN.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_cost': avg_cost
        }, CHECKPOINT_PATH/checkpoint_name)

