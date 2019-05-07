import glob
import random
import sys
from datetime import datetime
from pathlib import Path

import gc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse

from torch.utils.data import RandomSampler, SubsetRandomSampler, Subset
from tqdm import tqdm

from architectures import SegNet, ResNetUnet
from create_dataset import *
from torch.autograd import Variable

from gradient_logger import GradientLogger
from utils import inspect_gpu_tensors

torch.backends.cudnn.benchmark = True   # may speed up training if input sizes do not vary

seed = 43
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--weight', default='dwa', type=str, help='multi-task weighting: equal, uncert, dwa')
parser.add_argument('--dataroot', default='data/nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--model', default='segnet', type=str, help="Model architecture to use: ('segnet' or 'resnet')")
opt = parser.parse_args()

model = opt.model

gettrace = getattr(sys, 'gettrace', None)

no_debug = False
if gettrace is not None:
    if not gettrace():
        no_debug = True
else:
    no_debug = True

name_model_run = 'mtan_{}_run_{}'
if no_debug:
    log_every_nth = 10

    old_run_dirs = glob.glob('./logs/{}[0-9]*'.format(name_model_run.format(model, '')))
    if len(old_run_dirs) > 0:
        run_number = int(sorted(old_run_dirs)[-1].split('_')[-1]) + 1
    else:
        run_number = 0
    name_model_run = name_model_run.format(model, run_number)

    exist_ok = False
else:
    import shutil

    model = 'resnet'

    log_every_nth = 1

    name_model_run = name_model_run.format(model, 'debug')
    if os.path.exists('./logs/{}'.format(name_model_run)):
        shutil.rmtree('./logs/{}'.format(name_model_run))

    exist_ok = True

os.makedirs('./logs/{}'.format(name_model_run), exist_ok=exist_ok)

print(name_model_run)

if model == 'resnet':
    arch = ResNetUnet
else:
    model = 'segnet'
    arch = SegNet

# define model, optimiser and scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cleanup_gpu_memory_every_batch = True
model_MTAN = arch(device)
optimizer = optim.Adam(model_MTAN.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# add gradient logger hooks
model_MTAN.gradient_logger_hooks('./logs/{}/gradient_logs/'.format(name_model_run))


# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}\n'.format(count_parameters(model_MTAN),
                                                           count_parameters(model_MTAN)/24981069))
print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30\n')

# define dataset path
dataset_path = opt.dataroot
nyuv2_train_set = NYUv2(root=dataset_path, train=True)
nyuv2_test_set = NYUv2(root=dataset_path, train=False)

batch_size = 2  # is the current max for segnet on 12gb gpu

if not no_debug:
    subsample = 0.1

    shuffled = random.sample(list(range(len(nyuv2_train_set))), len(nyuv2_train_set))
    nyuv2_train_set = Subset(nyuv2_train_set, shuffled[:int(len(nyuv2_train_set)*subsample)])

    shuffled = random.sample(list(range(len(nyuv2_test_set))), len(nyuv2_test_set))
    nyuv2_test_set = Subset(nyuv2_test_set, shuffled[:int(len(nyuv2_test_set) * subsample)])

    num_workers = 0
else:
    num_workers = 2

nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False)


# define parameters
total_epoch = 100   # he trained for 200
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
    for train_data, train_label, train_depth, train_normal in tqdm(nyuv2_train_loader, desc='Training'):
        train_data = train_data.to(device)
        train_label = train_label.type(torch.LongTensor)

        train_pred, logsigma = model_MTAN(train_data)

        train_loss = model_MTAN.model_fit(train_pred[0], train_label, train_pred[1], train_depth, train_pred[2], train_normal)

        if opt.weight == 'equal' or opt.weight == 'dwa':
            # loss = torch.mean(sum(lambda_weight[i, index] * train_loss[i] for i in range(3)))
            loss = list(lambda_weight[i, index] * train_loss[i] for i in range(3))
        else:
            loss = list(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(3))

        for n, nth_loss in enumerate(loss):
            if n < len(loss) - 1:
                retain = True
            else:   # don't retain after last loss
                retain = False
            nth_loss.backward(retain_graph=retain)    # accumulates single backward passes

        optimizer.step()    # creates states for moving average -> more memory than for first batch
        optimizer.zero_grad()

        with torch.no_grad():
            cost[0] = train_loss[0].item()
            cost[1] = model_MTAN.compute_miou(train_pred[0], train_label).item()
            cost[2] = model_MTAN.compute_iou(train_pred[0], train_label).item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = model_MTAN.depth_error(train_pred[1], train_depth)
            cost[4], cost[5] = cost[4].item(), cost[5].item()
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = model_MTAN.normal_error(train_pred[2], train_normal)
            avg_cost[index, :12] += cost[:12] / len(nyuv2_train_loader)

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
        for test_data, test_label, test_depth, test_normal in tqdm(nyuv2_test_loader, desc='Testing'):
            test_data, test_label = test_data.to(device),  test_label.type(torch.LongTensor).to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)

            test_pred, _ = model_MTAN(test_data)
            test_loss = model_MTAN.model_fit(test_pred[0], test_label, test_pred[1], test_depth, test_pred[2], test_normal)

            cost[12] = test_loss[0].item()
            cost[13] = model_MTAN.compute_miou(test_pred[0], test_label).item()
            cost[14] = model_MTAN.compute_iou(test_pred[0], test_label).item()
            cost[15] = test_loss[1].item()
            cost[16], cost[17] = model_MTAN.depth_error(test_pred[1], test_depth)
            cost[18] = test_loss[2].item()
            cost[19], cost[20], cost[21], cost[22], cost[23] = model_MTAN.normal_error(test_pred[2], test_normal)

            avg_cost[index, 12:] += cost[12:] / len(nyuv2_test_loader)

    time_elapsed_epoch = datetime.now() - start_time_epoch
    print('Elapsted Minutes: {}'.format(time_elapsed_epoch))
    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'
          '               TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} '
          .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8], avg_cost[index, 9],
                avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23]))

    model_MTAN.update_gradient_loggers(index)

    if index % log_every_nth == 0:
        CHECKPOINT_PATH = Path('./logs/{}/model_checkpoints/'.format(name_model_run))
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)

        checkpoint_name = 'checkpoint.chk'
        torch.save({
            'epoch': index,
            'model_state_dict': model_MTAN.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_cost': avg_cost
        }, CHECKPOINT_PATH/checkpoint_name)

        model_MTAN.write_gradient_loggers()

