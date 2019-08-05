import sys
import platform
import argparse
import os
from collections import defaultdict

from metrics import IntersectionOverUnion, PixelAccuracy, DepthErrors

available_architectures = ('resnet', 'segnet_without_attention', 'segnet')
available_optimizers = ('adam', 'multitask_adam', 'multitask_adam_hd', 'multitask_adam_mixing_hd')

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--weight', default='dwa', type=str, help='multi-task weighting: equal, uncert, dwa, gradnorm')
parser.add_argument('--dataroot', default='data/nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--model', default='segnet', type=str, help="Model architecture to use: {}"
                    .format(available_architectures))
parser.add_argument('--gpu', default=-1, type=int, help='gpu to run on')
parser.add_argument('--shrink', default=1, type=int, help='shrinkage_factor on image sizes')
parser.add_argument('--bsm', default=1, type=int, help='batch size multiplies, i.e. batch size = 2 * bsm')
parser.add_argument('--optimizer', default='adam', type=str, help='optimizer to use: {}'.format(available_optimizers))
parser.add_argument('--not_per_filter', default=False, type=bool,
                    help='Whether to use learnable lr or mixing factors per filter.')
opt = parser.parse_args()

gettrace = getattr(sys, 'gettrace', None)

no_debug = False
if gettrace is not None:
    if not gettrace():
        no_debug = True
else:
    no_debug = True

if opt.gpu == -1:
    if platform.node() == 'eltanin':
        gpu = 3
    elif platform.node() in ('sabik', 'risha'):
        gpu = 0
    else:
        print('Specify gpu for host: ' + platform.node())
        sys.exit(-1)
else:
    gpu = opt.gpu

if not no_debug:
    gpu = 1

# define model, optimiser and scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
print('RUN ON GPU: ' + str(gpu))


import glob

import random
from datetime import datetime
from pathlib import Path

import gc
import math

from tqdm import tqdm

import torch.optim as optim
from torch.utils.data import RandomSampler, Subset

from architectures import SegNet, ResNetUnet, SegNetWithoutAttention
from multitask_optimizers import MultitaskAdam, MultitaskAdamHD, MultitaskOptimizer, MultitaskAdamMixingHD
from create_dataset import *
from model_testing import evaluate_model, write_performance

if opt.model not in available_architectures:
    print("Specified unknown model_name: {}. Changed to 'segnet''".format(opt.model))
    opt.model = 'segnet'

if opt.optimizer not in available_optimizers:
    print("Specified unknown optimizer_name: {}. Changed to 'adam''".format(opt.optimizer))
    opt.optimizer = 'adam'


def shrink_str(shrink):
    if shrink != 1:
        return 'shrink_' + str(shrink) + '_'
    else:
        return ''


name_model_run = 'mtan_{}_{}_{}_{}run_{}'
if no_debug:
    log_every_nth = 2       # way more memory necessary for gradient logging

    old_run_dirs = glob.glob('./logs/{}[0-9]*'.format(name_model_run.
                                                      format(opt.model, opt.weight, opt.optimizer, shrink_str(opt.shrink), '')))
    if len(old_run_dirs) > 0:
        run_number = int(sorted(old_run_dirs)[-1].split('_')[-1]) + 1
    else:
        run_number = 0
    name_model_run = name_model_run.format(opt.model, opt.weight, opt.optimizer, shrink_str(opt.shrink), run_number)

    exist_ok = False
else:
    import shutil

    opt.model = 'segnet_without_attention'
    opt.optimizer = 'multitask_adam_mixing_hd'
    opt.not_per_filter = False
    # opt.weight = 'gradnorm'
    # opt.shrink = 4

    log_every_nth = 1

    name_model_run = name_model_run.format(opt.model, opt.weight, opt.optimizer, shrink_str(opt.shrink), 'debug')
    if os.path.exists('./logs/{}'.format(name_model_run)):
        shutil.rmtree('./logs/{}'.format(name_model_run))

    exist_ok = True

os.makedirs('./logs/{}'.format(name_model_run), exist_ok=exist_ok)
with open('./logs/{}/cmd_parameters.txt'.format(name_model_run), 'w') as f:
    print(opt, file=f)

print(name_model_run)

batch_size = 2 * (opt.shrink**2) * opt.bsm  # 2 is the current max for segnet on 12gb gpu with original sizes

if opt.model == 'resnet':
    arch = ResNetUnet
    # batch_size = 7  # is the maximum for resnet without skip connection
    batch_size = 5  # is the maximum for resnet without skip connection
elif opt.model == 'segnet_without_attention':
    arch = SegNetWithoutAttention
else:
    arch = SegNet

optimizer_map = {'adam': optim.Adam, 'multitask_adam': MultitaskAdam,
                 'multitask_adam_hd': MultitaskAdamHD, 'multitask_adam_mixing_hd': MultitaskAdamMixingHD}
optimizer = optimizer_map.get(opt.optimizer, optim.Adam)

device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
cleanup_gpu_memory_every_batch = True
torch.backends.cudnn.benchmark = True   # may speed up training if input sizes do not vary

seed = 43
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

model = arch(device)

if opt.weight == 'gradnorm':
    model.register_parameter('task_weights', torch.nn.Parameter(torch.ones(3, device=device)))
    model.grad_norm_hook()
    initial_task_loss = None

if opt.weight not in ('equal', 'dwa', 'gradnorm'):
    # weight uncertainty
    model.register_parameter('logsigma', torch.nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5], device=device)))

kwargs = {}
if opt.optimizer in ('multitask_adam_hd', 'multitask_adam_mixing_hd'):
    kwargs = {'per_filter': not opt.not_per_filter}

optimizer = optimizer(model.parameters(), lr=1e-4*opt.shrink*math.sqrt(opt.bsm), **kwargs)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# add gradient logger hooks
model.gradient_logger_hooks('./logs/{}/gradient_logs/'.format(name_model_run))

# log task_weights only for the params which gradients are logged
for group in optimizer.param_groups:
    if group.get('logging'):
        d = {}
        for hooked in model.hooked_params_or_modules:
            d[hooked] = True
        group.setdefault('param_to_log_map', d)


# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}\n'.format(count_parameters(model),
                                                           count_parameters(model) / 24981069))
loss_str = 'LOSS FORMAT: SEMANTIC_LOSS | MEAN_IOU PIX_ACC | DEPTH_LOSS | ABS_ERR REL_ERR | NORMAL_LOSS | MEAN MED <11.25 <22.5 <30\n'
print(loss_str)

# define dataset path
dataset_path = opt.dataroot
nyuv2_train_set = NYUv2(root=dataset_path, train=True, shrinkage_factor=opt.shrink)
nyuv2_test_set = NYUv2(root=dataset_path, train=False, shrinkage_factor=opt.shrink)

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
test_avg_cost = np.zeros(12, dtype=np.float32)
test_cost = np.zeros(12, dtype=np.float32)
lambda_weight = np.ones([3, total_epoch])

model.train()
for epoch in range(total_epoch):
    start_time_epoch = datetime.now()

    index = epoch
    cost = np.zeros(24, dtype=np.float32)
    # scheduler.step()

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

    miou_metric = IntersectionOverUnion(-1, model.class_nb)
    iou_metric = PixelAccuracy(-1, model.class_nb)
    depth_errors = DepthErrors(rmse=True)
    metric_callbacks = [(miou_metric, 0), (iou_metric, 0), (depth_errors, 1)]

    for cb, _ in metric_callbacks:
        cb.on_epoch_begin()

    loop_run = False
    iteration = 0
    # iteration for all batches
    for train_data, train_label, train_depth, train_normal in tqdm(nyuv2_train_loader, desc='Training'):
        loop_run = True
        train_data = train_data.to(device)
        train_label = train_label.type(torch.LongTensor).to(device)
        train_depth = train_depth.to(device)
        train_normal = train_normal.to(device)
        train_labels = [train_label, train_depth, train_normal]

        model.calculate_metrics_gradient_loggers()

        train_pred = model(train_data)
        train_loss = model.model_fit(train_pred[0], train_label, train_pred[1], train_depth, train_pred[2],
                                     train_normal)

        if opt.weight == 'equal' or opt.weight == 'dwa':
            # loss = torch.mean(sum(lambda_weight[i, index] * train_loss[i] for i in range(3)))
            loss = (torch.FloatTensor(lambda_weight)[:, index]).to(device) * train_loss
        elif opt.weight == 'gradnorm':
            loss = model.task_weights.data * train_loss

            if initial_task_loss is None:
                initial_task_loss = loss.data
        else:
            loss = torch.stack(
                list(1 / (2 * torch.exp(model.logsigma[i])) * train_loss[i] + model.logsigma[i] / 2 for i in range(3)))

        for n, nth_loss in enumerate(loss):
            if n < len(loss) - 1:
                retain = True
            else:   # don't retain after last loss
                retain = False
            nth_loss.backward(retain_graph=retain)    # accumulates single backward passes
            if isinstance(optimizer, MultitaskOptimizer):
                optimizer.zero_grad()   # TODO check if this is memory okay
                # -> zero_grad just sets gradients to zero and detaches them

        if opt.weight == 'gradnorm':
            # model.task_weights.grad.data.zero_()
            model.grad_norms_last_shared_layer = torch.stack(model.grad_norms_last_shared_layer)
            model.grad_norms_last_shared_layer = model.task_weights * model.grad_norms_last_shared_layer

            alpha = 0.16
            mean_norm = torch.mean(model.grad_norms_last_shared_layer.data)
            loss_ratio = train_loss.data / initial_task_loss
            inverse_train_rate = loss_ratio / torch.mean(loss_ratio)

            diff = model.grad_norms_last_shared_layer - (inverse_train_rate ** alpha) * mean_norm
            grad_norm_loss = torch.mean(torch.abs(diff))
            grad_norm_loss.backward()
            if isinstance(optimizer, MultitaskOptimizer):
                optimizer.zero_grad()

            model.grad_norms_last_shared_layer = []

        optimizer.step()    # creates states for moving average -> more memory than for first batch
        optimizer.zero_grad()

        if opt.weight == 'gradnorm':
            # Renormalize
            normalize_coeff = model.n_tasks / torch.sum(model.task_weights.data)
            model.task_weights.data *= normalize_coeff

        # log 2 averages per epoch if task weights per filter (<- logic in the optimizer logging methods)
        if iteration == len(nyuv2_train_loader) // 2:
            if any([group.get('logging') for group in optimizer.param_groups]):
                optimizer.update_optimizer_data_logs(model.named_parameters(), index)

        with torch.no_grad():
            for cb, ind in metric_callbacks:
                cb.on_batch_end(train_pred[ind], train_labels[ind].to(model.device))

            cost[0] = train_loss[0].item()
            # cost[1] = model.compute_miou(train_pred[0], train_label).item()
            # cost[2] = model.compute_iou(train_pred[0], train_label).item()
            cost[3] = train_loss[1].item()
            # cost[4], cost[5] = model.depth_error(train_pred[1], train_depth)
            cost[4], cost[5] = cost[4].item(), cost[5].item()
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = model.normal_error(train_pred[2], train_normal)
            avg_cost[index, :12] += cost[:12] / len(nyuv2_train_loader)

        if cleanup_gpu_memory_every_batch:
            # train_loss = [train_loss[i].detach().cpu() for i in range(len(train_loss))]
            # loss = loss.detach().cpu()
            del train_pred
            del train_loss
            del loss
            torch.cuda.empty_cache()
            gc.collect()

        iteration += 1

    if loop_run:
        model.calculate_metrics_gradient_loggers()

    for cb, _ in metric_callbacks:
        cb.on_epoch_end()

    avg_cost[index, 1] = miou_metric.metric
    avg_cost[index, 2] = iou_metric.metric
    avg_cost[index, 4] = depth_errors.metric[0]
    avg_cost[index, 5] = depth_errors.metric[1]

    avg_cost[index, 12:], test_performance = evaluate_model(model, nyuv2_test_loader, device, index,
                                                            test_avg_cost, test_cost)

    time_elapsed_epoch = datetime.now() - start_time_epoch
    print('Elapsted Minutes: {}'.format(time_elapsed_epoch))

    performance = '''Epoch: {:04d} | TRAIN: {:.4f} | {:.4f} {:.4f} | {:.4f} | {:.4f} {:.4f} | {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}
    {}'''.format(
            index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
            avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
            avg_cost[index, 9],
            avg_cost[index, 10], avg_cost[index, 11],
            test_performance)

    print(performance)

    model.update_gradient_loggers(index)
    if any([group.get('logging') for group in optimizer.param_groups]):
        optimizer.update_optimizer_data_logs(model.named_parameters(), index)

    if index % log_every_nth == 0 or index == total_epoch-1:
        CHECKPOINT_PATH = Path('./logs/{}/model_checkpoints/'.format(name_model_run))
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)

        checkpoint_name = 'checkpoint.chk'
        torch.save({
            'architecture': type(model).__name__,
            'epoch': index,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_cost': avg_cost
        }, CHECKPOINT_PATH/checkpoint_name)

        model.write_gradient_loggers()
        if any([group.get('logging') for group in optimizer.param_groups]):
            optimizer.log_optimizer_data('./logs/{}/optimizer_logs/'.format(name_model_run))

write_performance(name_model_run, performance, loss_str)
