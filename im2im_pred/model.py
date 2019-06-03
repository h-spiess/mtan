import argparse
import glob
import platform
import random
import sys
from datetime import datetime
from pathlib import Path

import gc
import torch.optim as optim
from torch.utils.data import RandomSampler, Subset
from tqdm import tqdm

from architectures import SegNet, ResNetUnet, SegNetWithoutAttention
from create_dataset import *
from model_testing import evaluate_model, write_performance

parser = argparse.ArgumentParser(description='Multi-task: Attention Network')
parser.add_argument('--weight', default='dwa', type=str, help='multi-task weighting: equal, uncert, dwa, gradnorm')
parser.add_argument('--dataroot', default='data/nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=2.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--model', default='segnet', type=str, help="Model architecture to use: ('segnet' or 'resnet')")
parser.add_argument('--gpu', default=-1, type=int, help='gpu to run on')
opt = parser.parse_args()

model_name = opt.model

gettrace = getattr(sys, 'gettrace', None)

no_debug = False
if gettrace is not None:
    if not gettrace():
        no_debug = True
else:
    no_debug = True

if model_name not in ('resnet', 'segnet_without_attention', 'segnet'):
    print("Specified unknown model_name: {}. Changed to 'segnet''".format(model_name))
    model_name = 'segnet'

name_model_run = 'mtan_{}_{}_run_{}'
if no_debug:
    log_every_nth = 10

    old_run_dirs = glob.glob('./logs/{}[0-9]*'.format(name_model_run.format(model_name, opt.weight, '')))
    if len(old_run_dirs) > 0:
        run_number = int(sorted(old_run_dirs)[-1].split('_')[-1]) + 1
    else:
        run_number = 0
    name_model_run = name_model_run.format(model_name, opt.weight, run_number)

    exist_ok = False
else:
    import shutil

    model_name = 'segnet'
    # opt.weight = 'gradnorm'

    log_every_nth = 1

    name_model_run = name_model_run.format(model_name, opt.weight, 'debug')
    if os.path.exists('./logs/{}'.format(name_model_run)):
        shutil.rmtree('./logs/{}'.format(name_model_run))

    exist_ok = True

os.makedirs('./logs/{}'.format(name_model_run), exist_ok=exist_ok)

print(name_model_run)

batch_size = 2  # is the current max for segnet on 12gb gpu

if model_name == 'resnet':
    arch = ResNetUnet
    # batch_size = 7  # is the maximum for resnet without skip connection
    batch_size = 5  # is the maximum for resnet without skip connection
elif model_name == 'segnet_without_attention':
    arch = SegNetWithoutAttention
else:
    arch = SegNet

if opt.gpu == -1:
    if platform.node() == 'eltanin':
        gpu = 3
    elif platform.node() == 'sabik':
        gpu = 0
    else:
        print('Specify gpu for host: ' + platform.node())
        sys.exit(-1)
else:
    gpu = opt.gpu

if not no_debug:
    gpu = 2


# define model, optimiser and scheduler
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
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

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# add gradient logger hooks
model.gradient_logger_hooks('./logs/{}/gradient_logs/'.format(name_model_run))


# compute parameter space
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Parameter Space: ABS: {:.1f}, REL: {:.4f}\n'.format(count_parameters(model),
                                                           count_parameters(model) / 24981069))
loss_str = 'LOSS FORMAT: SEMANTIC_LOSS | MEAN_IOU PIX_ACC | DEPTH_LOSS | ABS_ERR REL_ERR | NORMAL_LOSS | MEAN MED <11.25 <22.5 <30\n'
print(loss_str)

# define dataset path
dataset_path = opt.dataroot
nyuv2_train_set = NYUv2(root=dataset_path, train=True)
nyuv2_test_set = NYUv2(root=dataset_path, train=False)

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

        train_pred, logsigma = model(train_data)

        train_loss = model.model_fit(train_pred[0], train_label, train_pred[1], train_depth, train_pred[2], train_normal)

        if opt.weight == 'equal' or opt.weight == 'dwa':
            # loss = torch.mean(sum(lambda_weight[i, index] * train_loss[i] for i in range(3)))
            loss = (torch.FloatTensor(lambda_weight)[:, index]).to(device) * train_loss
        elif opt.weight == 'gradnorm':
            loss = model.task_weights.data * train_loss

            if initial_task_loss is None:
                initial_task_loss = loss.data
        else:
            loss = torch.stack(list(1 / (2 * torch.exp(logsigma[i])) * train_loss[i] + logsigma[i] / 2 for i in range(3)))

        for n, nth_loss in enumerate(loss):
            if n < len(loss) - 1:
                retain = True
            else:   # don't retain after last loss
                retain = False
            nth_loss.backward(retain_graph=retain)    # accumulates single backward passes

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

            model.grad_norms_last_shared_layer = []

        optimizer.step()    # creates states for moving average -> more memory than for first batch
        optimizer.zero_grad()

        if opt.weight == 'gradnorm':
            # Renormalize
            normalize_coeff = model.n_tasks / torch.sum(model.task_weights.data)
            model.task_weights.data *= normalize_coeff

        with torch.no_grad():
            cost[0] = train_loss[0].item()
            cost[1] = model.compute_miou(train_pred[0], train_label).item()
            cost[2] = model.compute_iou(train_pred[0], train_label).item()
            cost[3] = train_loss[1].item()
            cost[4], cost[5] = model.depth_error(train_pred[1], train_depth)
            cost[4], cost[5] = cost[4].item(), cost[5].item()
            cost[6] = train_loss[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = model.normal_error(train_pred[2], train_normal)
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

    avg_cost[index, 12:], test_performance = evaluate_model(model, nyuv2_test_loader, device, index)

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

    if index % log_every_nth == 0:
        CHECKPOINT_PATH = Path('./logs/{}/model_checkpoints/'.format(name_model_run))
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)

        checkpoint_name = 'checkpoint.chk'
        torch.save({
            'epoch': index,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_cost': avg_cost
        }, CHECKPOINT_PATH/checkpoint_name)

        model.write_gradient_loggers()

write_performance(name_model_run, performance, loss_str)
