import glob
import itertools
import os
import random
from copy import copy

import matplotlib
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE, SpectralEmbedding
from umap import UMAP
from torch.utils.data import Subset

import myAggClustering

os.environ['PATH'] = '/home/spiess/anaconda3/envs/thesis/bin:/home/spiess/anaconda3/condabin:' + os.environ['PATH']
import shutil
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import scipy.stats
import seaborn as sns
import sklearn.manifold
import torch
import xarray as xr
import pandas as pd
from torch.nn import functional as F
from tqdm import tqdm

import architectures
from create_dataset import NYUv2
from metrics import IntersectionOverUnion, PixelAccuracy, DepthErrors
from multitask_optimizers import MultitaskAdamMixingHD

import matplotlib.pyplot as plt
from matplotlib import rc, patches

rc('text', usetex=True)
# charter as first font
plt.rcParams['text.latex.preamble'] = [r'\usepackage[libertine]{newtxmath}']
plt.rcParams['font.serif'][0], plt.rcParams['font.serif'][-2] = plt.rcParams['font.serif'][-2], \
                                                                plt.rcParams['font.serif'][0]
plt.rcParams['font.family'] = 'serif'

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-1)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

plt.rc('legend', handletextpad=0.1)
plt.rc('legend', columnspacing=1.0)

plt.interactive(False)
import math


def correlation_matrix(activations, SPLITROWS=100):
    # number of rows in one chunk

    numrows = activations.shape[0]

    # subtract means form the input data
    activations -= np.mean(activations, axis=1)[:, None]

    # normalize the data
    activations /= np.sqrt(np.sum(activations * activations, axis=1))[:, None]

    # reserve the resulting table onto HDD
    # res = np.memmap("/tmp/mydata.dat", 'float64', mode='w+', shape=(numrows, numrows))
    res = np.zeros((numrows, numrows))

    for r in tqdm(range(0, numrows, SPLITROWS), desc='outer'):
        for c in tqdm(range(0, numrows, SPLITROWS), desc='inner'):
            r1 = r + SPLITROWS
            c1 = c + SPLITROWS
            chunk1 = activations[r:r1]
            chunk2 = activations[c:c1]
            res[r:r1, c:c1] = np.dot(chunk1, chunk2.T)

    del activations

    return res


def append_attentions(l, self, input, output):
    if isinstance(output, tuple):   # output, output_intermediate for attention network
        output = output[0]
    for i in range(output.size()[0]):
        l.append(output[i, :, :, :].detach().clone().cpu().numpy())


def append_attentions_maxpool(c, n, l, self, input, output):
    if c.val % 5 != n:  # only for encoder
        c.val += 1
        return
    if isinstance(output, tuple):   # MaxPool2D
        output = output[0]
    else:
        raise ValueError()
    for i in range(output.size()[0]):
        l.append(output[i, :, :, :].detach().clone().cpu().numpy())
    c.val += 1


def append_attentions_attention_blocks(l, self, input):
    for i in range(input[0].size()[0]):
        l.append(input[0][i, :, :, :].detach().clone().cpu().numpy())


def append_attentions_attention_module(ls, self, input, output):
    assert len(output[1]) == 3, 'Wrong output length.'
    for j, output in enumerate(output[1]):
        for i in range(output.size()[0]):
            ls[j].append(output[i, :, :, :].detach().clone().cpu().numpy())


def intersectionAndUnion(imPred, imLab, numClass, missing=-1):
    """
        This function takes the prediction and label of a single image, returns intersection and union areas for each class
        To compute over many images do:
        for i in range(Nimages):
            (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
        IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
        """
    imPred = np.asarray(imPred)
    imLab = np.asarray(imLab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * ((imLab != missing) * 2 - 1)

    # Compute area intersection:
    intersection = imPred * ((imPred == imLab) * 2 - 1)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(0, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(0, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(0, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


def pixelAccuracy(imPred, imLab, missing=-1):
    # This function takes the prediction and label of a single image, returns pixel-wise accuracy
    # To compute over many images do:
    # for i = range(Nimages):
    #	(pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = pixelAccuracy(imPred[i], imLab[i])
    # mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))

    imPred = np.asarray(imPred)
    imLab = np.asarray(imLab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab != missing)
    pixel_correct = np.sum((imPred == imLab) * (imLab != missing))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled

    return (pixel_accuracy, pixel_correct, pixel_labeled)


def evaluate_model_input_dissimilarity(model, test_loader, device, block_n=None, intrinsic_dimension=False):
    model.eval()

    if block_n is None:
        last_layer = True
        representation_layer = model.last_shared_layer()
    else:
        last_layer = False
        representation_layer = model.modules_forward_hook_activity(block_n)

    if not isinstance(representation_layer, tuple):
        representation_layer = (representation_layer,)

    ress = []
    for i, representation_l in tqdm(enumerate(representation_layer), desc='Run Input Dissimilarity'):
        l = []
        multiple_lists = False
        if i < 1:
            if last_layer or block_n >= 5:
                appender = partial(append_attentions, l)
                handle = representation_l.register_forward_hook(appender)
            else:
                class Counter:
                    def __init__(self):
                        self.val = 0

                c = Counter()

                appender = partial(append_attentions_maxpool, c, block_n, l)
                handle = representation_l.register_forward_hook(appender)
        else:
            if last_layer:
                appender_attention_blocks = partial(append_attentions_attention_blocks, l)
                handle = representation_l.register_forward_pre_hook(appender_attention_blocks)
            else:
                l2 = []
                l3 = []
                appender_attention_module = partial(append_attentions_attention_module, [l, l2, l3])
                handle = representation_l.register_forward_hook(appender_attention_module)
                multiple_lists = True

        with torch.no_grad():  # operations inside don't track history
            for test_data, _, _, _ in tqdm(test_loader, desc='Testing'):
                test_data = test_data.to(device)
                _ = model(test_data)

            if not multiple_lists:
                ls = [l]
            else:
                ls = [l, l2, l3]
            for l in ls:
                activations = np.array(l)
                del l[:]
                activations = activations.reshape(activations.shape[0], -1)
                if not intrinsic_dimension:
                    res = correlation_matrix(activations, SPLITROWS=100)
                    res = 1 - res  # dissimilarity
                else:
                    res = squareform(pdist(activations))
                ress.append(res)

            handle.remove()

    return ress


def plot_sample_and_labels(input, pred_label, label, depth, normal, model):
    plot_two_truths = False

    if not plot_two_truths:
        input, pred_label, pred_depth, pred_normal, label, depth, normal = input[0], pred_label[0][0], pred_label[1][0], \
                                                                           pred_label[2][0], label[0], depth[0], normal[0]
    else:
        input, input2, pred_label, pred_depth, pred_normal, label, depth, normal = input[1], input[0], label[0], depth[
            0], normal[0], label[1], depth[1], normal[1]

    class_nb = model.class_nb

    miou_metric = IntersectionOverUnion(-1, class_nb)
    pixel_acc_metric = PixelAccuracy(-1, class_nb, mean_over_classes=True)
    depth_errors = DepthErrors(rmse=True)
    metric_callbacks = [(miou_metric, 0), (pixel_acc_metric, 0), (depth_errors, 1)]

    test_labels = [label.unsqueeze(0), depth.unsqueeze(0), normal.unsqueeze(0)]
    test_pred = [pred_label.unsqueeze(0), pred_depth.unsqueeze(0), pred_normal.unsqueeze(0)]

    if not plot_two_truths:
        for cb, ind in metric_callbacks:
            cb.on_epoch_begin()
            cb.on_batch_end(test_pred[ind], test_labels[ind].to(pred_label.device))
            cb.on_epoch_end()

        normal_err = model.normal_error(test_pred[-1], test_labels[-1])

    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0, class_nb, class_nb + 1)
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    depth_vmin, depth_vmax = min(depth[0].min().item(), pred_depth[0].min().item()), max(depth[0].max().item(),
                                                                                       pred_depth[0].max().item())

    def ticks_off():
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False)

    if not plot_two_truths:
        pred_label = pred_label.argmax(dim=0)

    plt.figure()
    plt.subplot(2, 4, 1)
    plt.imshow(input.permute(1, -1, 0), interpolation='none')
    ticks_off()

    plt.subplot(2, 4, 2)
    plt.imshow(label, cmap=cmap, norm=norm, interpolation='none')
    ticks_off()

    plt.subplot(2, 4, 3)
    plt.imshow(depth[0], 'gray', interpolation='none', vmin=depth_vmin, vmax=depth_vmax)
    ticks_off()

    plt.subplot(2, 4, 4)
    plt.imshow(normal.permute(1, -1, 0), interpolation='none')
    ticks_off()

    if plot_two_truths:
        ax = plt.subplot(2, 4, 5)
        plt.imshow(input2.permute(1, -1, 0), interpolation='none')
        ticks_off()

    ax=plt.subplot(2, 4, 6)
    plt.imshow(pred_label, cmap=cmap, norm=norm, interpolation='none')
    ticks_off()
    if not plot_two_truths:
        ax.set_title('$IoU$: {:.3f}\n$PA$: {:.3f}'.format(miou_metric.metric, pixel_acc_metric.metric))

    ax=plt.subplot(2, 4, 7)
    plt.imshow(pred_depth[0], 'gray', interpolation='none', vmin=depth_vmin, vmax=depth_vmax)
    ticks_off()
    if not plot_two_truths:
        ax.set_title('$RMSE$: {:.3f}\n$SRD$: {:.3f}'.format(depth_errors.metric[0], depth_errors.metric[1]))

    ax=plt.subplot(2, 4, 8)
    plt.imshow(pred_normal.permute(1, -1, 0), interpolation='none')
    ticks_off()
    if not plot_two_truths:
        ax.set_title('$MAD$: {:.3f},\n$MMAD$: {:.3f}'.format(normal_err[0], normal_err[1]))
    # plt.subplots_adjust(0.05, 0.05, 0.95, 0.95)
    plt.tight_layout()
    if plot_two_truths:
        plt.savefig('./plots/dataset_samples.png', dpi=300)
    else:
        plt.savefig('./plots/dataset_sample_and_predictions.png', dpi=300)
    plt.show()


def evaluate_model(model, test_loader, device, index, avg_cost, cost, pixel_acc_mean_over_classes=True, test=False,
                   input_dissimilarity=False, plot_sample=False):
    model.eval()

    if not hasattr(model, 'class_weights'):
        class_weights = torch.ones(model.class_nb)
        model.class_weights = class_weights.to(device)

    miou_metric = IntersectionOverUnion(-1, model.class_nb)
    pixel_acc_metric = PixelAccuracy(-1, model.class_nb, mean_over_classes=pixel_acc_mean_over_classes)
    depth_errors = DepthErrors(rmse=True)
    metric_callbacks = [(miou_metric, 0), (pixel_acc_metric, 0), (depth_errors, 1)]

    area_intersections = []
    area_unions = []
    pixel_accuracies = []
    pixel_corrects = []
    pixel_labeled = []

    if test:
        original_batch_costs = np.zeros(4, np.float32)

        metric_batches = {miou_metric: [],
                          pixel_acc_metric: []}

    for cb, _ in metric_callbacks:
        cb.on_epoch_begin()

    if input_dissimilarity:
        l = []
        appender = partial(append_attentions, l)
        representation_layer = model.last_shared_layer()

        representations_attention_blocks = False
        if isinstance(representation_layer, tuple):
                ls = [[], [], []]
                appender_attention_blocks = partial(append_attentions_attention_blocks, ls)

                representation_layer[0].register_forward_hook(appender)
                representation_layer[1].register_forward_hook(appender_attention_blocks)
                representations_attention_blocks = True
        else:
            representation_layer.register_forward_hook(appender)

    # evaluating test data
    avg_cost.fill(0.), cost.fill(0.)
    with torch.no_grad():  # operations inside don't track history

        if plot_sample:
            seed = 5
            random.seed(seed)
            np.random.seed(seed)
            torch.random.manual_seed(seed)
            test_loader = NYUv2(root='data/nyuv2', train=False, shrinkage_factor=1)
            shuffled = random.sample(list(range(len(test_loader))), len(test_loader))
            test_loader = Subset(test_loader, shuffled[:int(len(test_loader) * 1.0)])
            test_loader = torch.utils.data.DataLoader(
                dataset=test_loader,
                batch_size=2,
                num_workers=1,
                shuffle=False)

        for test_data, test_label, test_depth, test_normal in tqdm(test_loader, desc='Testing'):
            test_data, test_label = test_data.to(device), test_label.type(torch.LongTensor).to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)
            test_labels = [test_label, test_depth, test_normal]

            test_pred = model(test_data)
            test_loss = model.model_fit(test_pred[0], test_label, test_pred[1], test_depth, test_pred[2], test_normal)

            if plot_sample:
                plot_sample_and_labels(test_data, test_pred, test_label, test_depth, test_normal, model)
                return

            cost[0] = test_loss[0].item()
            if test:
                original_batch_costs[0] += model.compute_miou(test_pred[0], test_label).item() / len(test_loader)
                original_batch_costs[1] += model.compute_iou(test_pred[0], test_label).item() / len(test_loader)

            for cb, ind in metric_callbacks:
                cb.on_batch_end(test_pred[ind], test_labels[ind].to(model.device))
                if test:
                    if cb is miou_metric or cb is pixel_acc_metric:
                        cb.on_epoch_end()
                        metric_batches[cb].append(cb.metric)
                        # intersectionAndUnion(test_pred[ind].numpy(), test_labels[ind].numpy(), model.class_nb)
                        # pixel_accuracies(test_pred[ind].numpy(), test_labels[ind].numpy())

            if test:
                for sample in range(test_pred[0].size()[0]):
                    a_i, a_u = intersectionAndUnion(test_pred[0][sample, :, :, :].argmax(dim=0).cpu().numpy(),
                                                    test_labels[0][sample, :, :].cpu().numpy(),
                                                    model.class_nb)
                    area_intersections.append(a_i)
                    area_unions.append(a_u)

                    p_a, p_c, p_l = pixelAccuracy(test_pred[0][sample, :, :, :].argmax(dim=0).cpu().numpy(),
                                                  test_labels[0][sample, :, :].cpu().numpy())
                    pixel_accuracies.append(p_a)
                    pixel_corrects.append(p_c)
                    pixel_labeled.append(p_l)

            cost[3] = test_loss[1].item()
            if test:
                de_0, de_1 = model.depth_error(test_pred[1], test_depth)
                original_batch_costs[2] += de_0 / len(test_loader)
                original_batch_costs[3] += de_1 / len(test_loader)
            cost[6] = test_loss[2].item()
            # averaging normal error is okay, if batch size remains the same
            # calculates the mean of medians -> no on-line algo for medians available / possible
            cost[7], cost[8], cost[9], cost[10], cost[11] = model.normal_error(test_pred[2], test_normal)

            avg_cost += cost / len(test_loader)

        for cb, _ in metric_callbacks:
            cb.on_epoch_end()

        if input_dissimilarity:
            ls = [l].append(ls) if representations_attention_blocks else [l]
            ress = []
            for l in ls:
                activations = np.array(l)
                del l
                activations = activations.reshape(activations.shape[0], -1)
                res = correlation_matrix(activations, SPLITROWS=100)
                res = 1 - res  # dissimilarity
                ress.append(res)

        if test:
            area_intersections = np.array(area_intersections)
            area_unions = np.array(area_unions)
            pixel_corrects = np.array(pixel_corrects)
            pixel_labeled = np.array(pixel_labeled)

            IoU = 1.0 * np.sum(area_intersections, axis=0) / np.sum(np.spacing(1) + area_unions, axis=0)
            mean_pixel_accuracy = 1.0 * np.sum(pixel_corrects) / (np.spacing(1) + np.sum(pixel_labeled))

        avg_cost[1] = miou_metric.metric
        avg_cost[2] = pixel_acc_metric.metric
        avg_cost[4] = depth_errors.metric[0]
        avg_cost[5] = depth_errors.metric[1]

        performance = '''Epoch: {:04d} | TEST : {:.4f} | {:.4f} {:.4f} | {:.4f} | {:.4f} {:.4f} | {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'''.format(
            index,
            avg_cost[0],
            avg_cost[1], avg_cost[2],
            avg_cost[3],
            avg_cost[4], avg_cost[5],
            avg_cost[6],
            avg_cost[7], avg_cost[8], avg_cost[9], avg_cost[10],
            avg_cost[11])

        if test:
            ret = (avg_cost, performance, [np.array(l).mean() for _, l in metric_batches.items()], original_batch_costs, \
                   [mean_pixel_accuracy, IoU])
            if input_dissimilarity:
                ret += (res,)
            return ret
        else:
            ret = (avg_cost, performance)
            if input_dissimilarity:
                ret += (res,)
            return ret


def load_model(CHECKPOINT_PATH, device, ModelClass=None, **kwargs):
    checkpoint = torch.load(CHECKPOINT_PATH)

    if ModelClass is None:
        if 'architecture' not in checkpoint:
            raise ValueError('Name of architecture not in checkpoint. Specify "ModelClass" kwarg.')
        else:
            ModelClass = getattr(sys.modules['architectures'], checkpoint['architecture'])

    model = ModelClass(device, **kwargs)
    if 'gradnorm' in str(CHECKPOINT_PATH):
        model.register_parameter('task_weights', torch.nn.Parameter(torch.ones(3, device=device)))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def test_multitask_adam_mixing_hd_grads(CHECKPOINT_PATH, device, ModelClass=None, **kwargs):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input = torch.randn(1, 3, 100, 100).to(device)
    target_r = torch.rand(1).to(device)

    test_grad_eps = 1e-4

    # plus eps
    model = load_model(CHECKPOINT_PATH, device, ModelClass=ModelClass, **kwargs)
    model.train()
    optimizer = MultitaskAdamMixingHD(model.parameters(), lr=1e-4 * math.sqrt(2), test_grad_eps=test_grad_eps,
                                      per_filter=False)

    def forw(model, optimizer):
        output = model.forward(input)
        loss_1 = F.mse_loss(output[0], target_r * torch.ones_like(output[0]))
        loss_2 = F.l1_loss(output[1], target_r * torch.ones_like(output[1]))
        loss_1_square = F.mse_loss(output[2], target_r * torch.ones_like(output[2])) ** 2

        loss_1.backward(retain_graph=True)
        optimizer.zero_grad()
        loss_2.backward(retain_graph=True)
        optimizer.zero_grad()
        loss_1_square.backward()
        optimizer.zero_grad()

        h_mix = optimizer.step()  # this updates one mixing weight only
        optimizer.zero_grad()
        return h_mix

    _ = forw(model, optimizer)
    h_mixing_grad_0 = forw(model, optimizer)

    output = model.forward(input)
    output = output[0]
    target = target_r * torch.ones_like(output)
    loss_plus_eps = F.mse_loss(output, target).item()

    # minus eps
    model = load_model(CHECKPOINT_PATH, device, ModelClass=ModelClass, **kwargs)
    model.train()
    optimizer = MultitaskAdamMixingHD(model.parameters(), lr=1e-4 * math.sqrt(2), test_grad_eps=-test_grad_eps,
                                      per_filter=False)

    _ = forw(model, optimizer)
    h_mixing_grad_1 = forw(model, optimizer)

    output = model.forward(input)
    output = output[0]
    target = target_r * torch.ones_like(output)
    loss_minus_eps = F.mse_loss(output, target).item()

    assert h_mixing_grad_0 == h_mixing_grad_1, 'Different gradients between two runs! Should be equal.'

    finite_diff_gradient = (loss_plus_eps - loss_minus_eps) / (2 * test_grad_eps)

    grad_difference = finite_diff_gradient - h_mixing_grad_0
    pass


def calculate_artificial_css(CHECKPOINT_PATH, device, ModelClass=None, one_head=True, segmentation=True, real_images=True,
                             **kwargs):

    # def dice_loss(true, logits, eps=1e-7):
    #     """Computes the Sørensen–Dice loss.
    #     Note that PyTorch optimizers minimize a loss. In this
    #     case, we would like to maximize the dice loss so we
    #     return the negated dice loss.
    #     Args:
    #         true: a tensor of shape [B, 1, H, W].
    #         logits: a tensor of shape [B, C, H, W]. Corresponds to
    #             the raw output or logits of the model.
    #         eps: added to the denominator for numerical stability.
    #     Returns:
    #         dice_loss: the Sørensen–Dice loss.
    #     """
    #     num_classes = logits.shape[1]
    #     if num_classes == 1:
    #         true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
    #         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    #         true_1_hot_f = true_1_hot[:, 0:1, :, :]
    #         true_1_hot_s = true_1_hot[:, 1:2, :, :]
    #         true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
    #         pos_prob = torch.sigmoid(logits)
    #         neg_prob = 1 - pos_prob
    #         probas = torch.cat([pos_prob, neg_prob], dim=1)
    #     else:
    #         true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
    #         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    #         probas = F.softmax(logits, dim=1)
    #     true_1_hot = true_1_hot.type(logits.type())
    #     dims = (0,) + tuple(range(2, true.ndimension()))
    #     intersection = torch.sum(probas * true_1_hot, dims)
    #     cardinality = torch.sum(probas + true_1_hot, dims)
    #     dice_loss = (2. * intersection / (cardinality + eps)).mean()
    #     return (1 - dice_loss)

    def to_one_hot(tensor, nClasses):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, nClasses+1, h, w).to(tensor.device).scatter_(1, tensor.view(n, 1, h, w)+1, 1)
        one_hot = one_hot[:, 1:, :, :]
        return one_hot

    def dice_loss(pred, target, classes=13):
        """This definition generalize to real valued pred and target vector.
    This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        """

        pred = pred.exp()
        target = to_one_hot(target.argmax(dim=1), classes)

        smooth = 1.

        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)

        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

    def jaccard_loss(inputs, target, classes=13, eps=1e-7):

        inputs = inputs.exp()
        target = to_one_hot(target.argmax(dim=1), classes)

        """Paper: Optimizing Intersection-Over-Union in Deep Neural Networks for Image Segmentation.
        """
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W

        N = inputs.size()[0]

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, classes, -1).sum(2)

        # Denominator
        union = inputs + target- (inputs * target)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, classes, -1).sum(2)

        loss = inter / union

        ## Return average loss over classes and batch
        return 1-loss.mean()

    model = load_model(CHECKPOINT_PATH, device, ModelClass=ModelClass, **kwargs)
    model.train()
    save_path = Path('./logs/artificial_css/')
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    model.gradient_logger_hooks(save_path)

    nrepeat = 200
    if not real_images:
        fake_input = [torch.randn(1, 3, 100, 100) for _ in range(nrepeat)]
        fake_target = [torch.randint((1, 13, 100, 100), 0, 13) for _ in range(nrepeat)]

        loader = itertools.zip_longest(fake_input, fake_target, [None], [None])
    else:
        dataset_path = 'data/nyuv2'
        # TODO change to train=False again
        nyuv2_test_set = NYUv2(root=dataset_path, train=False)

        batch_size = 1
        num_workers = 1

        loader = torch.utils.data.DataLoader(
            dataset=nyuv2_test_set,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True)
    # TODO: send real pictures
    n = 0
    for input, target, _, _ in loader:
        input = input.to(device)
        target = target.to(device).long()
        target = to_one_hot(target, 13)

        if n >= nrepeat:
            break
        n += 1

        output = model.forward(input)
        if one_head:
            output = output[0]  # is log softmax
            # target = torch.rand(1).to(device) * torch.ones_like(output)

            if segmentation:
                loss_dice = dice_loss(output, target)
                loss_jaccard = jaccard_loss(output, target)
                target_max = target.argmax(dim=1)
                loss_cross = F.nll_loss(output, target_max.long())
                losses = (loss_dice, loss_jaccard, loss_cross)
                # print(losses)
                loss_vs_names = ('Soft-Dice, Jaccard', 'Soft-Dice, $L^1$', 'Jaccard, $L^1$')
            else:
                loss_1 = F.mse_loss(output, target)
                loss_2 = F.l1_loss(output, target)
                loss_1_square = F.mse_loss(output, target) ** 2
                losses = (loss_1, loss_2, loss_1_square)
                loss_vs_names = ('L1, L2', 'L1, L1$^2$', 'L2, L1$^2$')
        else:
            rand_scalar = torch.rand(1).to(device)
            if segmentation:
                raise ValueError("Multihead and segmentation together makes no sense!")
            else:
                loss_1 = F.mse_loss(output[0], rand_scalar * torch.ones_like(output[0]))
                loss_2 = F.l1_loss(output[1], rand_scalar * torch.ones_like(output[1]))
                loss_1_square = F.mse_loss(output[2], rand_scalar * torch.ones_like(output[2])) ** 2
                losses = (loss_1, loss_2, loss_1_square)
                loss_vs_names = ('L1, L2', 'L1, L1$^2$', 'L2, L1$^2$')
        losses[0].backward(retain_graph=True)
        losses[1].backward(retain_graph=True)
        losses[2].backward()
        model.calculate_metrics_gradient_loggers()
    artificial_css = [[[] for _ in range(3)] for _ in range(len(model.gradient_loggers))]
    blocks = []
    
    def change_layer_name(layer_name):
        l = layer_name.split('_')
        return ' '.join([l[0].capitalize(), l[1].capitalize(), str(int(l[2])+1)])
    
    for i, gradient_logger in enumerate(model.gradient_loggers):
        blocks.append(change_layer_name(gradient_logger.layer_name))
        artificial_css[i][0] += [np.array(gradient_logger.grad_metrics['cosine_similarity_separate_filter_grad_weights_task1_task2_per_filter']).mean(axis=1)]
        artificial_css[i][1] += [np.array(gradient_logger.grad_metrics['cosine_similarity_separate_filter_grad_weights_task1_task3_per_filter']).mean(axis=1)]
        artificial_css[i][2] += [np.array(gradient_logger.grad_metrics['cosine_similarity_separate_filter_grad_weights_task2_task3_per_filter']).mean(axis=1)]
    artificial_css = np.array(artificial_css)[:, :, 0, :]
    return artificial_css, blocks, loss_vs_names


def plot_artificial_css(artificial_css, blocks, one_head, loss_vs_names):
    x = xr.DataArray(artificial_css, dims=('Layer', 'Losses', 'Observations'),
                     coords={'Layer': blocks,
                             'Losses': list(loss_vs_names),
                             'Observations': [i for i in
                                              range(artificial_css.shape[-1])]})
    df = x.to_dataframe('CSS').reset_index()
    plt.figure()
    ax = sns.barplot(x="Layer", y="CSS", hue='Losses', data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right',
                      # fontsize=MEDIUM_SIZE
                      )
    ax.set_xlabel('')
    plt.subplots_adjust(0.1, 0.15, 0.95, 0.95)
    plt.savefig(Path('./logs/artificial_css/{}.png'.format('one_head' if one_head else 'separate_heads')), dpi=300)
    plt.show()


def evaluate_saved_model(CHECKPOINT_PATH, device, ModelClass=None, test=False, input_dissimilarity=False,
                         intrinsic_dimension=False,
                         plot_sample=False, block_n=None, **kwargs):
    dataset_path = 'data/nyuv2'
    #TODO change to train=False again
    nyuv2_test_set = NYUv2(root=dataset_path, train=False)

    assert not input_dissimilarity or not intrinsic_dimension, "Can not compute input dissimilarity " \
                                                               "and intrinsic dimension at the same time."

    batch_size = 1 if input_dissimilarity or intrinsic_dimension else 2
    num_workers = 2

    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)

    test_avg_cost = np.zeros(12, dtype=np.float32)
    test_cost = np.zeros(12, dtype=np.float32)

    model = load_model(CHECKPOINT_PATH, device, ModelClass=ModelClass, **kwargs)
    if input_dissimilarity:
        return evaluate_model_input_dissimilarity(model, nyuv2_test_loader, device, block_n=block_n)
    elif intrinsic_dimension:
        return evaluate_model_input_dissimilarity(model, nyuv2_test_loader, device, block_n=block_n, 
                                                  intrinsic_dimension=True)
    else:
        return evaluate_model(model, nyuv2_test_loader, device, -1, test_avg_cost, test_cost, test=test,
                              plot_sample=plot_sample)


def write_performance(name_model_run, performance, loss_str):
    PERFORMANCE_PATH = Path('./logs/{}/'.format(name_model_run))
    os.makedirs(PERFORMANCE_PATH, exist_ok=True)

    final_performance_str = 'final_performance{}.txt'
    if os.path.exists(PERFORMANCE_PATH / 'final_performance.txt'):
        final_performance_str = final_performance_str.format('_extra_test')
    else:
        final_performance_str = final_performance_str.format('')

    with open(PERFORMANCE_PATH / final_performance_str, 'w+') as handle:
        handle.write(loss_str)
        handle.write(performance)


def write_input_dissimilarity(name_model_run, input_dissimilarity, att_block_n=None, block_n=None, epoch_n=None,
                              intrinsic_dimension=False):
    epoch_str = '_epoch_{}'.format(epoch_n) if epoch_n is not None else ''
    block_str = '_block_{}'.format(block_n) if block_n is not None else ''
    att_block_str = '_att_block_{}'.format(att_block_n) if att_block_n else ''
    SAVE_PATH = Path('./logs/{}/'.format(name_model_run))
    os.makedirs(SAVE_PATH, exist_ok=True)
    name = 'input_dissimilarity' if not intrinsic_dimension else 'activation_distances'
    np.save(SAVE_PATH / '{}{}{}{}.npy'.format(name, block_str, att_block_str, epoch_str), input_dissimilarity)


def plot_input_dissimilarity_from_file(path):
    input_dissimilarity = np.load(path)
    plot_input_dissimilarity(input_dissimilarity)


def plot_input_dissimilarity(input_dissimilarity):
    f = plt.figure()
    plt.imshow(input_dissimilarity if np.all(np.diag(input_dissimilarity) == 0) else 1 - input_dissimilarity)
    plt.colorbar()
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()


def create_label(name_model):
    retrain_name_dict = {
        'mtan_segnet_without_attention_equal_adam_single_task_0_retrain_head_run_0':'SN Task $2 \\rightarrow 1$',
        'mtan_segnet_without_attention_equal_adam_single_task_0_retrain_head_run_1':'SN Task $3 \\rightarrow 1$',
        'mtan_segnet_without_attention_equal_adam_single_task_1_retrain_head_run_0':'SN Task $1 \\rightarrow 2$',
        'mtan_segnet_without_attention_equal_adam_single_task_1_retrain_head_run_1':'SN Task $3 \\rightarrow 2$',
        'mtan_segnet_without_attention_equal_adam_single_task_2_retrain_head_run_0':'SN Task $1 \\rightarrow 3$',
        'mtan_segnet_without_attention_equal_adam_single_task_2_retrain_head_run_1':'SN Task $2 \\rightarrow 3$',
        'mtan_segnet_without_attention_dwa_adam_retrain_head_run_0':'SN+DWA+A All $\\rightarrow$ All',
    }
    if 'retrain_head' in name_model:
        return retrain_name_dict[name_model]
    if 'single_task' in name_model:
        return 'Single Task ' + str(int(name_model[41+len('single_task')+1])+1)
    n = 'SN' if 'segnet_without_attention' in name_model else 'SNA'
    n += '+'
    if 'equal' in name_model:
        n += 'EQ'
    elif 'dwa' in name_model:
        n += 'DWA'
    elif 'gradnorm' in name_model:
        n += 'GN'
    n += '+'
    if 'multitask_adam_hd' in name_model:
        n += 'MTAHD'
    elif 'multitask_adam_linear_combination_hd' in name_model:
        n += 'ALC'
    elif 'multitask_adam' in name_model:
        n += 'MTA'
    elif 'adam' in name_model:
        n += 'A'
    return n


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def plot_metrics(name_models, metrics, ncols=2, start_epoch=0, subsampled=False, finetuning=False, just_values=False,
                 best_subsampled=False, bar=False):

    def merge_single_task_models(name_models, metrics_datas, metrics_epochs=None):
        best_subsampled = metrics_epochs is not None
        
        inds = []
        vals = []
        for i, name_model in enumerate(name_models):
            if name_model is not None and 'Single Task' in name_model:
                inds.append(i)
                vals.append(int(name_model[-1]))
        if len(vals) == 0:
            return
        assert len(vals) == 3
        inds = [x for _, x in sorted(zip(vals, inds))]
        merged_metrics = metrics_datas[inds[0]]
        merged_metrics[:, [3, 4, 5, 15, 16, 17]] = metrics_datas[inds[1]][:, [3, 4, 5, 15, 16, 17]]
        merged_metrics[:, [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]] = \
            metrics_datas[inds[2]][:, [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]]
        if best_subsampled:
            merged_epochs = metrics_epochs[inds[0]]
            merged_epochs[:, [3, 4, 5, 15, 16, 17]] = metrics_epochs[inds[1]][:, [3, 4, 5, 15, 16, 17]]
            merged_epochs[:, [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]] = \
                metrics_epochs[inds[2]][:, [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]]
        for i in sorted(inds, reverse=True):
            del metrics_datas[i]
            del name_models[i]
            if best_subsampled:
                del metrics_epochs[i]

        metrics_datas.insert(0, merged_metrics)
        name_models.insert(0, 'SN ST')
        if best_subsampled:
            metrics_epochs.insert(0, merged_epochs)


    metric_name_to_ind = {
        # formatter
        # second, third: epochs mean, std
        # fourth, fifth: values mean, std

        'segmentation_loss': (0, '$L^1$', FormatStrFormatter('%1.1f')),
        'segmentation_miou': (1, '$IoU$', FormatStrFormatter('%.2f'),
                              FormatStrFormatter('%2d'), FormatStrFormatter('%2.1f'),
                              FormatStrFormatter('%.3f'), FormatStrFormatter('%.3f')),
        'segmentation_pix_acc': (2, '$PA$', FormatStrFormatter('%.2f'),
                              FormatStrFormatter('%2d'), FormatStrFormatter('%2.1f'),
                              FormatStrFormatter('%.3f'), FormatStrFormatter('%.3f')),

        'depth_loss': (3, '$L^2$', FormatStrFormatter('%.2f')),
        'depth_abs_err': (4, '$RMSE$', FormatStrFormatter('%1.2f'),
                              FormatStrFormatter('%2d'), FormatStrFormatter('%2.1f'),
                              FormatStrFormatter('%.3f'), FormatStrFormatter('%.3f')),
        'depth_rel_err': (5, '$SRD$', FormatStrFormatter('%.2f'),
                              FormatStrFormatter('%2d'), FormatStrFormatter('%2.1f'),
                              FormatStrFormatter('%.3f'), FormatStrFormatter('%.3f')),

        'normals_loss': (6, '$L^3$', FormatStrFormatter('%.2f')),
        'normals_angle_dist_mean': (7, '$MAD$', FormatStrFormatter('%2d'),
                              FormatStrFormatter('%2d'), FormatStrFormatter('%2.1f'),
                              FormatStrFormatter('%2.1f'), FormatStrFormatter('%2.1f')),
        'normals_angle_dist_median': (8, '$MMAD$', FormatStrFormatter('%2d'),
                              FormatStrFormatter('%2d'), FormatStrFormatter('%2.1f'),
                              FormatStrFormatter('%2.1f'), FormatStrFormatter('%2.1f')),
        'normals_within_11.5': (9, '$ADR_{<11.5^\circ}$', FormatStrFormatter('%.2f'),
                              FormatStrFormatter('%2d'), FormatStrFormatter('%2.1f'),
                              FormatStrFormatter('%.3f'), FormatStrFormatter('%.3f')),
        'normals_within_22.5': (10, '$ADR_{<22.5^\circ}$', FormatStrFormatter('%.2f'),
                              FormatStrFormatter('%2d'), FormatStrFormatter('%2.1f'),
                              FormatStrFormatter('%.3f'), FormatStrFormatter('%.3f')),
        'normals_within_30': (11, '$ADR_{<30^\circ}$', FormatStrFormatter('%.2f'),
                              FormatStrFormatter('%2d'), FormatStrFormatter('%2.1f'),
                              FormatStrFormatter('%.3f'), FormatStrFormatter('%.3f'))
    }

    metric_name_to_opt_funcs = {
        'segmentation_loss': (np.nanargmin, np.nanmin),

        'segmentation_miou': (np.nanargmax, np.nanmax),
        'segmentation_pix_acc': (np.nanargmax, np.nanmax),

        'depth_loss': (np.nanargmin, np.nanmin),
        'depth_abs_err': (np.nanargmin, np.nanmin),
        'depth_rel_err': (np.nanargmin, np.nanmin),

        'normals_loss': (np.nanargmin, np.nanmin),
        'normals_angle_dist_mean': (np.nanargmin, np.nanmin),
        'normals_angle_dist_median': (np.nanargmin, np.nanmin),
        'normals_within_11.5': (np.nanargmax, np.nanmax),
        'normals_within_22.5': (np.nanargmax, np.nanmax),
        'normals_within_30': (np.nanargmax, np.nanmax)
    }

    for m in metrics:
        assert m in metric_name_to_ind, 'Unknown metric: ' + str(m)

    if isinstance(name_models[0], str):
        with_train = True

        CHECKPOINT_PATHS = [Path('./logs/{}/model_checkpoints/checkpoint.chk'.format(name_model)) for name_model in name_models]
        metrics_datas = [torch.load(CHECKPOINT_PATH)['avg_cost'] for CHECKPOINT_PATH in CHECKPOINT_PATHS]
        metrics_datas = [metrics_data[~np.all(metrics_data == 0, axis=1)] for metrics_data in metrics_datas]
    else:
        with_train = False

        metrics_datass = []
        metrics_epochss = []

        for name_model_list in name_models:
            if name_model_list is None:
                metrics_datass.append(None)
                metrics_epochss.append(None)
                continue

            CHECKPOINT_PATHS = [Path('./logs/{}/model_checkpoints/checkpoint.chk'.format(name_model)) for name_model in
                                name_model_list]
            metrics_datas = [torch.load(CHECKPOINT_PATH)['avg_cost'] for CHECKPOINT_PATH in CHECKPOINT_PATHS]

            if not subsampled:
                # todo: min, middle, max
                sizes = [metrics_data.shape[0] for metrics_data in metrics_datas]
                mins, maxs = min(sizes), max(sizes)
                if mins != maxs:
                    for i, size in enumerate(sizes):
                        if size < maxs:
                            metrics_datas[i] = np.pad(metrics_datas[i], (0, maxs-size, 0, 0))

                metrics_datas = np.stack(metrics_datas, axis=2)
                metrics_datas.sort(axis=2)
                metrics_datass.append(metrics_datas[np.all(metrics_datas != 0, axis=(1, 2))])

            else:
                if not best_subsampled:
                    metrics_datas = [metrics_data[-1] for metrics_data in metrics_datas]
                else:
                    ind_to_opt_funcs = {value[0]: metric_name_to_opt_funcs[key] for key, value in
                                        metric_name_to_ind.items()}
                    metrics_datas_ = []
                    metrics_epochs = []
                    for metrics_data in metrics_datas:
                        metrics_data_ = np.zeros(metrics_data.shape[1]) 
                        metrics_epochs_ = np.zeros(metrics_data.shape[1])
                        for ind in range(metrics_data.shape[1]):
                            argopt, opt = ind_to_opt_funcs[ind if ind < 12 else ind - 12]
                            metrics_epochs_[ind] = argopt(metrics_data[:, ind]) + 1
                            metrics_data_[ind] = opt(metrics_data[:, ind])
                        metrics_datas_.append(metrics_data_)
                        metrics_epochs.append(metrics_epochs_)
                    metrics_datas = metrics_datas_
                    metrics_epochs = np.stack(metrics_epochs, axis=0)
                        
                metrics_datas = np.stack(metrics_datas, axis=0)
                metrics_datass.append(metrics_datas)
                if best_subsampled:
                    metrics_epochss.append(metrics_epochs)

        metrics_datas = metrics_datass
        if best_subsampled:
            metrics_epochs = metrics_epochss

    fig, axs = plt.subplots(math.ceil(len(metrics) / ncols), ncols, sharex=True, figsize=(5.5, 8))
    axs = axs.reshape(-1)
    # fig.suptitle('Metrics')

    if not subsampled:
        nepochs = [metrics_data.shape[0] for metrics_data in metrics_datas if metrics_data is not None]
        if len(set(nepochs)) != 1:
            smaller_inds = np.where(np.array(nepochs) < np.max(nepochs))[0]
            for si in smaller_inds:
                if with_train:
                    padding = ((0, np.max(nepochs) - metrics_datas[si].shape[0]), (0, 0))
                else:
                    padding = ((0, np.max(nepochs) - metrics_datas[si].shape[0]), (0, 0), (0, 0))

                metrics_datas[si] = np.pad(metrics_datas[si], padding, 'constant',
                       constant_values=np.nan)
        nepochs = nepochs[0]

        xtrain = np.arange(0.5+start_epoch, 0.5 + nepochs)
        xtest = np.arange(1+start_epoch, 1 + nepochs)

    else:
        start_epoch = 0
        xtrain = np.array([0.25, 0.5, 0.75, 1.0])
        xtest = xtrain

    if not with_train:
        ind = 0 if not subsampled else -1
        name_models = [name_model[ind] if name_model is not None else None for name_model in name_models]

    name_models = [create_label(name_model) if name_model is not None else None for name_model in name_models]

    if finetuning:
        assert name_models[0] is None and name_models[1] is None and name_models[2] is None
        name_models = name_models[2:]
        metrics_datas = metrics_datas[2:]
    else:
        if best_subsampled:
            merge_single_task_models(name_models, metrics_datas, metrics_epochs=metrics_epochs)
        else:
            merge_single_task_models(name_models, metrics_datas)

    if just_values:
        headers = ['Model']
        table = []

        # model metric mean std
        columns = ['Model', 'Metric', 'Observation']
        rows = []

        row_dict = {}
        for j, metrics_data in enumerate(metrics_datas):
            for i, m in enumerate(metrics):
                if 'loss' in m:
                    continue

                headers.append('\makecell{'+metric_name_to_ind[m][1] + ' $\{}$\\\\(Epoch)}}'.format(
                    'uparrow' if metric_name_to_opt_funcs[m][0] is np.nanargmax else 'downarrow'
                ))

                best_epochs = metric_name_to_opt_funcs[m][0](
                    metrics_data[start_epoch:, metric_name_to_ind[m][0] + 12], axis=0) + start_epoch + 1
                best_values = metric_name_to_opt_funcs[m][1](
                    metrics_data[start_epoch:, metric_name_to_ind[m][0] + 12], axis=0)

                values = ['\makecell{{${} \pm {}$ \\\\ $({} \pm {})$}}'.format(
                    metric_name_to_ind[m][5](best_values.mean()),
                    metric_name_to_ind[m][6](best_values.std()),
                    metric_name_to_ind[m][3](best_epochs.mean()),
                    metric_name_to_ind[m][4](best_epochs.std())
                    )
                ]

                for best_value in best_values:
                    row_df = [
                        name_models[j],
                        '' + metric_name_to_ind[m][1] + ' $\\{}$'.format(
                            'uparrow' if metric_name_to_opt_funcs[m][0] is np.nanargmax else 'downarrow'),
                        best_value
                    ]
                    rows.append(row_df)

                if name_models[j] in row_dict:
                    row_dict[name_models[j]] += values
                else:
                    row_dict[name_models[j]] = [name_models[j]] + values

        for row in row_dict.values():
            table.append(row)

        metrics_df = pd.DataFrame(rows, columns=columns)

        if bar:
            plot_metrics_bar(metrics_df)
            return metrics_df
        else:
            retrain_head = name_models[0][-1] == '$'
            with open('./plots/metrics_table{}.tex'.format('' if not retrain_head else '_retrained_head'), 'w+') as f:
                from tabulate import tabulate
                print(tabulate(table, headers, tablefmt="latex_raw"), file=f)
            return

    for i, m in enumerate(metrics):
        best_epochs = []
        for j, metrics_data in enumerate(metrics_datas):
            colors = sns.color_palette()
            if metrics_data is None:
                continue

            if not subsampled:
                if with_train:
                    axs[i].plot(xtrain, metrics_data[start_epoch:, metric_name_to_ind[m][0]], '--', color=colors[j % len(colors)])
                    axs[i].plot(xtest, metrics_data[start_epoch:, metric_name_to_ind[m][0] + 12], color=colors[j % len(colors)],
                                label=name_models[j].replace('_', ' '))
                    opt_ind = metric_name_to_opt_funcs[m][0](metrics_data[start_epoch:, metric_name_to_ind[m][0] + 12])
                    opt = metric_name_to_opt_funcs[m][1](metrics_data[start_epoch:, metric_name_to_ind[m][0] + 12])

                else:

                    axs[i].plot(xtest, metrics_data[start_epoch:, metric_name_to_ind[m][0] + 12, 1], color=colors[j % len(colors)],
                                label=name_models[j].replace('_', ' '))
                    axs[i].fill_between(xtest, metrics_data[start_epoch:, metric_name_to_ind[m][0] + 12, 0],
                                        metrics_data[start_epoch:, metric_name_to_ind[m][0] + 12, 2],
                                        facecolor=colors[j % len(colors)], alpha=0.5)
                    opt_ind = metric_name_to_opt_funcs[m][0](metrics_data[start_epoch:, metric_name_to_ind[m][0] + 12].mean(axis=1))
                    opt = metric_name_to_opt_funcs[m][1](metrics_data[start_epoch:, metric_name_to_ind[m][0] + 12].mean(axis=1))

                axs[i].scatter([xtest[opt_ind]], [opt], facecolors='none', marker='o', color=colors[j % len(colors)])
                axs[i].axvline(x=xtest[opt_ind], color=colors[j % len(colors)], alpha=0.5)
                axs[i].axhline(y=opt, color=colors[j % len(colors)], alpha=0.5)

            else:
                axs[i].plot(xtest, metrics_data[start_epoch:, metric_name_to_ind[m][0] + 12],
                            color=colors[j % len(colors)],
                            label=name_models[j].replace('_', ' '))
                if best_subsampled:
                    best_epochs.append('({}, {}, {}, {})'.format(*metrics_epochs[j][:, i+12].astype(int)))

            axs[i].yaxis.set_major_formatter(metric_name_to_ind[m][2])

        if i == 0:
            legend_handles_labels = copy(axs[i].get_legend_handles_labels())
            if not finetuning and not best_subsampled:
                axs[i].set_ylim(top=4.5)

        axs[i].set_title(metric_name_to_ind[m][1], pad=3)
        
        if best_subsampled:
            handles, labels = axs[i].get_legend_handles_labels()
            axs[i].legend(handles, best_epochs)

    for ax in axs[-ncols:]:
        if not subsampled and not finetuning:
            ax.set_xlabel('Epochs')
        elif finetuning:
            ax.set_xlabel('Fine-tuning Epochs')
        else:
            xticklabels = [0, 25, 50, 75, 100]
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel('\% of Training Set')

    fig.subplots_adjust(bottom=0.105, top=0.97, left=0.08, right=0.98, wspace=0.3)
    # plt.tight_layout(rect=(0, 0.08, 1, 1))
    if with_train:
        labels = legend_handles_labels[1] + ['Train', 'Test']
        lines = legend_handles_labels[0] + \
                [Line2D([0], [0], color='black', linewidth=1, linestyle='--' if l == 'Train' else None) for l in labels[-2:]]
    else:
        labels = legend_handles_labels[1]
        lines = legend_handles_labels[0]
    # axs[-ncols].legend(lines, labels, ncol=5, loc='upper center', bbox_to_anchor=(1, -0.45))
    legend_cols = 5
    fig.legend(flip(lines, legend_cols), flip(labels, legend_cols), ncol=legend_cols, loc='lower center')
    plt.savefig('./plots/metrics{}{}{}{}{}.png'.format('_subsampled' if subsampled else '',
                                                       '_best' if best_subsampled else '',
                                                     '_finetuning' if finetuning else '',
                                                     '' if not with_train else '_with_train',
                                                     '_se{}'.format(start_epoch) if start_epoch > 0 else ''), dpi=300)
    plt.show()


def correlation_matrix_of_rdms_from_names(model_names, reduce=False, all_blocks=False, over_time=False):
    return correlation_matrix_of_rdms(rdms_from_names(model_names, reduce=reduce, all_blocks=all_blocks, over_time=over_time))


def intrinsic_dimension_from_names(model_names, reduce=False, all_blocks=False, over_time=False):
    return rdms_from_names(model_names, reduce=reduce, all_blocks=all_blocks, over_time=over_time,
                           intrinsic_dimension=True)


def rdms_from_names(model_names, reduce=True, all_blocks=False, over_time=False, intrinsic_dimension=False):
    if all_blocks:
        inputs = ['_block_{}'.format(i) for i in range(0, 9)] + ['']
    else:
        inputs = ['']

    if over_time:
        times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, -1]
        inputs = [i + '_epoch_{}'.format(t) if t != -1 else i for (i, t) in itertools.product(inputs, times)]

    representation_type = 'input_dissimilarity' if not intrinsic_dimension else 'activation_distances'

    if isinstance(model_names[0], str):
        paths = [Path('./logs/{}/{}{}.npy'.format(name_model_run, representation_type, block)) for (name_model_run, block)
                 in itertools.product(model_names, inputs)]
        return rdms_from_paths(paths, intrinsic_dimension=intrinsic_dimension)
    elif isinstance(model_names[0], list):
        rdms = []
        for model_name_list in model_names:
            for block in inputs:
                paths = [Path('./logs/{}/{}{}.npy'.format(name_model_run, representation_type, block))
                         for name_model_run in model_name_list]
                # TODO pay attention to dimensions
                rdms_model = rdms_from_paths(paths, intrinsic_dimension=intrinsic_dimension)

                if reduce:
                    if 'without' in model_name_list[0]:
                        rdm = rdms_model[0]
                        for rdm_n in rdms_model[1:]:
                            rdm += rdm_n
                        rdm /= len(rdms_model)
                        rdms.append(rdm)
                    else:
                        rdm = rdms_model[0]
                        rdm1 = rdms_model[1]
                        rdm2 = rdms_model[2]
                        rdm3 = rdms_model[3]
                        for i in range(1, len(rdms_model)//4):
                            rdm  += rdms_model[0 + i * 4]
                            rdm1 += rdms_model[1 + i * 4]
                            rdm2 += rdms_model[2 + i * 4]
                            rdm3 += rdms_model[3 + i * 4]
                        rdm /= len(rdms_model)//4
                        rdm1 /= len(rdms_model) // 4
                        rdm2 /= len(rdms_model) // 4
                        rdm3 /= len(rdms_model) // 4
                        rdms.append(rdm)
                        rdms.append(rdm1)
                        rdms.append(rdm2)
                        rdms.append(rdm3)
                else:
                    for rdm in rdms_model:
                        rdms.append(rdm)
        return rdms
    else:
        raise ValueError()


def rdms_from_paths(paths_, intrinsic_dimension=False):
    representation_type = 'input_dissimilarity' if not intrinsic_dimension else 'activation_distances'


    rdms = []
    for p in paths_:
        if os.path.exists(p):
            if 'without' in str(p):
                rdms.append(np.load(p))
            else:
                sp = str(p.with_suffix(''))
                # case input_dissimilarity_block_xx_epoch_xx.npy
                if 'block' in str(p) and 'epoch' in str(p):
                    ep = sp[sp.find('_epoch'):]
                    bn = representation_type[-4:] + sp[sp.find('_block'):sp.find('_epoch')]
                    sp = sp.rstrip(ep)

                    paths = glob.glob(sp + '*')  # grab all input rdms of a specific model
                    paths = [p for p in paths if ep in p and bn in p]
                    paths.sort(key=lambda x: x.rstrip(ep+'.npy'))  # ignore the epoch, otherwise alphabetical ordering
                    if len(paths) != 4:
                        print("")
                # case input_dissimilarity_block_xx.npy
                elif 'block' in str(p) and 'epoch' not in str(p):
                    bn = representation_type[-4:] + sp[sp.find('_block'):]

                    paths = glob.glob(sp + '*')  # grab all input rdms of a specific model
                    paths = [p for p in paths if 'epoch' not in p and bn in p]
                    paths.sort()  # ignore the epoch, otherwise alphabetical ordering
                    if len(paths) != 4:
                        print("")
                # case input_dissimilarity_epoch_xx.npy
                elif 'block' not in str(p) and 'epoch' in str(p):
                    ep = sp[sp.find('_epoch'):]
                    sp = sp.rstrip(ep)

                    paths = glob.glob(sp + '*')  # grab all input rdms of a specific model
                    paths = [p for p in paths if ep in p and representation_type[-4:]+'_block' not in p]
                    paths.sort(key=lambda x: x.rstrip(ep+'.npy'))  # ignore the epoch, otherwise alphabetical ordering
                    if len(paths) != 4:
                        print("")
                # case input_dissimilarity.npy
                else:
                    paths = glob.glob(sp + '*')  # grab all input rdms of a specific model
                    paths = [p for p in paths if 'epoch' not in p and representation_type[-4:]+'_block' not in p]
                    paths.sort()  # ignore the epoch, otherwise alphabetical ordering
                    if len(paths) != 4:
                        print("")

                for path in paths:
                    rdms.append(np.load(path))
        else:
            block_n = None
            if 'block' in p.name:
                block_n = int(p.stem.split('_')[p.stem.split('_').index('block')+1])
            epoch_n = None
            if 'epoch' in p.name:
                epoch_n = int(p.stem.split('_')[p.stem.split('_').index('epoch')+1])
            epoch_str = '' if epoch_n is None else '_epoch{}'.format(epoch_n)

            input_dissimilarities = evaluate_saved_model(
                p.parent / 'model_checkpoints/checkpoint{}.chk'.format(epoch_str),
                device, ModelClass=None,
                input_dissimilarity=not intrinsic_dimension,
                intrinsic_dimension=intrinsic_dimension,
                block_n=block_n)
            assert len(input_dissimilarities) == 1 or 'without' not in str(p)
            for i, input_dissimilarity in enumerate(input_dissimilarities):
                write_input_dissimilarity(p.parent.name, input_dissimilarity, att_block_n=i if i > 0 else None,
                                          block_n=block_n, epoch_n=epoch_n, intrinsic_dimension=intrinsic_dimension)
                rdms.append(input_dissimilarity)
    return rdms


def correlation_matrix_of_rdms_from_paths(paths):
    return correlation_matrix_of_rdms(rdms_from_paths(paths))


def correlation_matrix_of_rdms(rdms):
    rdms = np.array(rdms)
    rdms = rdms.reshape(rdms.shape[0], -1)
    correlation_matrix_of_rdms, _ = scipy.stats.spearmanr(rdms.T)
    return 1. - correlation_matrix_of_rdms


def plot_correlation_matrix_of_rdms(correlation_matrix_of_rdms, model_names, reduce=True):

    if isinstance(model_names[0], list) and reduce:
        model_names = [model_name_list[0] for model_name_list in model_names]

    if isinstance(model_names[0], list) and not reduce:
        model_names = [item for sublist in model_names for item in sublist]

    model_names = [create_label(name_model) for name_model in model_names]

    while True:
        index_segnet = -1
        if len(model_names) != correlation_matrix_of_rdms.shape[0]:
            for i, model_name in enumerate(model_names):
                if 'SNA' in model_name and '(' not in model_name:
                    index_segnet = i
            if not index_segnet != -1:
                break
            model_names_new = [model_names[index_segnet] + ' (Task 1)', model_names[index_segnet] + ' (Task 2)',
                               model_names[index_segnet] + ' (Task 3)']
            model_names[index_segnet] = model_names[index_segnet] + ' (All Tasks)'
            model_names = model_names[:index_segnet + 1] + model_names_new + model_names[index_segnet + 1:]
        else:
            break
        if reduce:
            break

    # Generate a mask for the upper triangle
    mask = np.zeros_like(correlation_matrix_of_rdms, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.diag_indices_from(mask)] = False

    # Set up the matplotlib figure
    if reduce:
        f, ax = plt.subplots(figsize=(5.5, 4.5))
    else:
        f, ax = plt.subplots(figsize=(20, 15))

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Create df
    correlation_matrix_of_rdms_df = pd.DataFrame(data=correlation_matrix_of_rdms, columns=model_names, index=model_names)

    vmin, vmax = correlation_matrix_of_rdms[mask].min(), correlation_matrix_of_rdms[mask].max()

    axins1 = inset_axes(ax,
                        width="5%",  # width of parent_bbox width
                        height="80%",  # height
                        loc='upper right')
    cbar_ax = axins1
    # cbar_ax = f.add_axes([.92, .3, .02, .4])
    # Draw the heatmap with the mask and correct aspect ratio
    h = sns.heatmap(correlation_matrix_of_rdms_df, mask=mask, ax=ax,
                    # cmap=cmap,
                    # vmin=vmin, vmax=vmax,
                    # robust=True,
                    # center=0.5,
                    square=True, linewidths=.2, cbar_kws={"shrink": .8},
                    cbar_ax=cbar_ax
                    )
    h.set_xticklabels(h.get_xticklabels(), rotation=30, ha='right',
                      # fontsize=MEDIUM_SIZE
                      )
    h.set_yticklabels(h.get_yticklabels(),
                      # fontsize=MEDIUM_SIZE
                      )

    # h.set_title('MDM')
    plt.subplots_adjust(left=0.15, right=0.99, top=0.96, bottom=0.2)
    plt.savefig('./plots/mdm.png', dpi=300)
    plt.show()


def plot_correlation_matrix_of_rdms_embedding(correlation_matrix_of_rdms, model_names, reduce=True, metrics=None,
                                              all_blocks=False, over_time=False, embedding='mds'):

    if isinstance(model_names[0], list) and reduce:
        model_names = [model_name_list[0] for model_name_list in model_names]

    if isinstance(model_names[0], list) and not reduce:
        model_names = [item for sublist in model_names for item in sublist]

    model_names = [create_label(name_model) for name_model in model_names]

    if all_blocks:
        prefixes = ['B{} '.format(i+1) if i < 10 else '' for i in range(0, 10)]
    else:
        prefixes = ['']

    if over_time:
        times = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        prefixes = [pre + 'E{} '.format(t) if t != -1 else pre for (pre, t) in itertools.product(prefixes, times)]

    model_names = [pre + mn for mn in model_names for pre in prefixes]

    save_index_segnet_ls = []
    for pre in prefixes:
        save_index_segnet = -1
        while True:
            index_segnet = -1
            if len(model_names) != correlation_matrix_of_rdms.shape[0]:
                for i, model_name in enumerate(model_names):
                    if '{}SNA'.format(pre) in model_name and '(' not in model_name and index_segnet == -1:
                        index_segnet = i
                if not index_segnet != -1:
                    break
                if save_index_segnet == -1:
                    save_index_segnet = index_segnet
                model_names_new = [model_names[index_segnet] + ' (Task 1)', model_names[index_segnet] + ' (Task 2)',
                                   model_names[index_segnet] + ' (Task 3)']
                model_names[index_segnet] = model_names[index_segnet] + ' (All Tasks)'
                model_names = model_names[:index_segnet + 1] + model_names_new + model_names[index_segnet + 1:]
            else:
                break
            if reduce:
                break
        save_index_segnet_ls.append(save_index_segnet)

    append_name = ''
    if embedding == 'mds':
        embedding = sklearn.manifold.MDS(n_components=2, dissimilarity='precomputed', random_state=1)
        mdm_transformed = embedding.fit_transform(correlation_matrix_of_rdms)

        print("Relative error of embedding")
        # http://www.mbfys.ru.nl/~robvdw/CNP04/LAB_ASSIGMENTS/LAB05_CN05/MATLAB2007b/stats/html/cmdscaledemo.html
        # maxrelerr = max(abs(D - pdist(Y(:,1:2)))) / max(D)
        from sklearn.metrics.pairwise import euclidean_distances
        edists = euclidean_distances(mdm_transformed, mdm_transformed)
        maxrelerr = np.max(np.abs(correlation_matrix_of_rdms - edists)) / correlation_matrix_of_rdms.max()
        print("maxrelerr:", maxrelerr)
        meanabserr = np.mean(np.abs(correlation_matrix_of_rdms - edists))
        print("meanabserr:", meanabserr)
        meanrelerr = np.mean(np.abs((correlation_matrix_of_rdms - edists) / (correlation_matrix_of_rdms + 1e-9)))
        print("meanrelerr:", meanrelerr)

    elif embedding == 'tsne':
        append_name = '_'+embedding
        mdm_transformed = TSNE(n_components=2, metric='precomputed').fit_transform(correlation_matrix_of_rdms)
        pass
    elif embedding == 'spectral':
        append_name = '_'+embedding
        mdm_transformed = SpectralEmbedding(n_components=2, affinity='precomputed').\
            fit_transform(1-correlation_matrix_of_rdms)
        pass
    elif embedding == 'umap':
        append_name = '_'+embedding
        mdm_transformed = UMAP(n_neighbors=50, random_state=42, metric='precomputed').fit_transform(
            correlation_matrix_of_rdms)
    elif embedding == 'som':
        import sompy
        # Not working and not that easy (no pairwise implementation available)
        
        append_name = '_'+embedding
        som = sompy.SOMFactory.build(correlation_matrix_of_rdms, [35, 35])
        som.train(n_job=1, verbose='info')
        mdm_transformed = som.project_data(correlation_matrix_of_rdms)
    else:
        raise ValueError()

    # TODO number of rdms is correct, something with the model names
    df = pd.DataFrame(mdm_transformed, columns=['x', 'y'])
    df['name'] = model_names
    df['style'] = 0

    if metrics is None:
        for (pre, save_index_segnet) in zip(prefixes, save_index_segnet_ls):
            for i, name in enumerate(model_names):
                if pre == '' and 'B' == name[0]:
                    continue
                if '{}Single Task'.format(pre) in name:
                    df.loc[i, 'style'] = int(name[-1])
                    df.loc[i, 'name'] = '{}Single Task'.format(pre)
                elif '(Task' in name and pre in name:
                    df.loc[i, 'style'] = int(name[-2])
                    df.loc[i, 'name'] = df.loc[save_index_segnet, 'name']
    else:
        df['join_name'] = df['name'].map(lambda x: x[:x.find('(') - 1] if x.find('(') != -1 else x)
        df = df.join(metrics.set_index('Model'), on='join_name')
        df['style'] = df.index
        df = df.melt(id_vars=['x', 'y', 'name', 'join_name', 'style'], var_name='Metric', value_name='Value')
        df = df.replace('SNA+DWA+A (All Tasks)', 'Attention All')
        df = df.replace('SNA+DWA+A (Task 1)', 'Attention Task 1')
        df = df.replace('SNA+DWA+A (Task 2)', 'Attention Task 2')
        df = df.replace('SNA+DWA+A (Task 3)', 'Attention Task 3')
        df = df.replace('SN+EQ+A', 'Tree-like')
        df = df.replace('SN+DWA+MTAHD', 'Tree-like\n+ Gradient Mixing')

        df['Order'] = df['Metric'].apply(
            lambda x: {'$RMSE$ $\\downarrow$': 1, '$MAD$ $\\downarrow$': 2, '$IoU$ $\\uparrow$': 0}[x])
        df = df.sort_values(by='Order')

        df = df.replace('$IoU$ $\\uparrow$', 'Sem. Segm. $\\uparrow$')
        df = df.replace('$MAD$ $\\downarrow$', 'Surf. Normals $\\downarrow$')
        df = df.replace('$RMSE$ $\\downarrow$', 'Depth $\\downarrow$')

    def name_to_alpha(name, start_alpha=0.2, blocks=10):
        if name[0] == 'B' and name[2] == ' ':
            alpha = start_alpha + (int(name[1])-1) * ((1.0-start_alpha)/(blocks-1))
            new_name = name[3:]
        else:
            alpha = 1.0
            if name[0] == 'B':
                new_name = name[4:]
            else:
                new_name = name
        return new_name, alpha

    def name_to_epoch(name):
        assert name[0] != 'B', 'Run name_to_alpha first'
        assert name[0] == 'E', 'Every name has to contain a epoch'
        splitted = name.split(' ')
        ep = int(splitted[0][1:])
        new_name = ' '.join(splitted[1:])

        return new_name, ep

    vnta = np.vectorize(name_to_alpha)
    new_names, alphas = vnta(df.name.values)
    df.name = new_names
    df['alpha'] = alphas

    plot_epochs = [None]

    if over_time:
        vnte = np.vectorize(name_to_epoch)
        new_names, epochs = vnte(df.name.values)
        df.name = new_names
        df['epoch'] = epochs

        plot_epochs = np.unique(epochs)

    for plot_epoch in plot_epochs:
        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(5.5 * (1 if metrics is None else 3), 4.5))
        if metrics is None:

            if plot_epoch != None:
                df_p = df[df.epoch == plot_epoch]
            else:
                df_p = df

            df_p['use'] = 1.

            markers = {
                0: 'o',
                1: 'X',
                2: 's',
                3: 'P',
                4: 'D',
                5: '^',
                6: 'v',
                7: 'p'
            }

            palette = {
                'Single Task': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                'SN+EQ+A': (1.0, 0.4980392156862745, 0.054901960784313725),
                'SN+DWA+A': (1.0, 0.4980392156862745, 0.054901960784313725),
                'SNA+DWA+A (All Tasks)': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                'SN+DWA+MTAHD': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
            }

            if embedding == 'mds':
                xlim = (-0.56939660443679, 0.7341330811042694)
                xlim = (-0.9, 0.7341330811042694)
                ylim = (-0.7031715957318339, 0.8)
                ylim = (-0.8, 0.85)
            else:
                borders = (mdm_transformed.max(axis=0) - mdm_transformed.min(axis=0)) * 0.05
                mins = mdm_transformed.min(axis=0) - borders
                maxs = mdm_transformed.max(axis=0) + borders
                xlim = (mins[0], maxs[0])
                ylim = (mins[1], maxs[1])


            alphas = df_p.alpha.unique()

            filter = True
            draw_around_last = False
            draw_connecting_lines = False
            if filter:
                # df_p.loc[df_p.name == 'Single Task', 'use'] = 0.
                # df_p.loc[df_p.alpha > 0.6, 'use'] = 0.
                # df_p.loc[df_p.alpha > alphas[1], 'use'] = 0.
                df_p['ns'] = df_p.apply(lambda x: x['name'] + ' ' + str(x['style']), axis=1)
                draw_around_last = True
                draw_connecting_lines = False
                print()

            if filter:
                alpha = 0.
                sc = sns.scatterplot(x='x', y='y', hue='name', style='style', data=df_p[df_p.alpha == alphas[0]], ax=ax,
                                     legend='full', s=50 if not all_blocks else 20, alpha=alpha,
                                     markers=markers, palette=palette)

            sc = sns.scatterplot(x='x', y='y', hue='name', style='style', data=df_p[(df_p.alpha == alphas[0]) & (df_p.use == 1.)],
                                 ax=ax if not filter else sc,
                                 legend='full' if not filter else None, s=50 if not all_blocks else 20, alpha=alphas[0],
                                 markers=markers, palette=palette)

            for alpha in alphas[1:]:
                sns.scatterplot(x='x', y='y', hue='name', style='style', data=df_p[(df_p.alpha == alpha) & (df_p.use == 1.)], ax=sc,
                                legend=None, s=50 if not all_blocks else 20, alpha=alpha,
                                markers=markers, palette=palette)

            if draw_around_last:
                alpha = alphas[-1]
                last_in_df_p = df_p[(df_p.alpha == alpha) & (df_p.use == 1.)]
                circle_rad = 5
                sc.plot(last_in_df_p.x, last_in_df_p.y, 'o',
                        ms=circle_rad * 2, mec='black', mfc='none', mew=1)

            if draw_connecting_lines:
                colors_ = [palette['Single Task']] * 3 + [palette['SN+EQ+A']] + [palette['SNA+DWA+A (All Tasks)']] * 4 + [
                    palette['SN+DWA+MTAHD']]
                for color_, ns in zip(colors_, df_p.ns.unique()):
                    df_p_f = df_p[df_p.ns == ns]
                    sc.plot(df_p_f.x, df_p_f.y, alpha=0.5, linewidth=0.5, color=color_)
                print()

            sc.set_xlim(*xlim)
            sc.set_ylim(*ylim)

            handles, labels = sc.get_legend_handles_labels()

            handle_size = 20
            newhandles = []
            newlabels = []
            for handle, label in zip(handles, labels):
                if label != 'name' and label != 'style':
                    handle._sizes[0] = handle_size
                    newhandles.append(handle)
                    newlabels.append(label)

            changedhandles = []
            changedlabels = []
            for handle, label in zip(newhandles[-3:], newlabels[-3:]):
                changedlabels.append(newlabels[0] + ' ' + label)
                handle_copy = copy(handle)
                handle_copy._edgecolors = newhandles[0]._edgecolors
                handle_copy._facecolors = newhandles[0]._facecolors
                changedhandles.append(handle_copy)

            if save_index_segnet != -1:
                found = False
                for i, label in enumerate(newlabels):
                    if 'SNA' in label:
                        found = True
                        index_segnet = i
                assert found

                changedhandles_segnet = []
                changedlabels_segnet = []
                for handle, label in zip(newhandles[-3:], newlabels[-3:]):
                    changedlabels_segnet.append(newlabels[index_segnet].replace('(All Tasks)', '(Task {})'.format(label)))
                    handle._edgecolors = newhandles[index_segnet]._edgecolors
                    handle._facecolors = newhandles[index_segnet]._facecolors
                    changedhandles_segnet.append(handle)

                newhandles = newhandles[:index_segnet+1] + changedhandles_segnet + newhandles[index_segnet+1:]
                newlabels = newlabels[:index_segnet+1] + changedlabels_segnet + newlabels[index_segnet+1:]

            newhandles = changedhandles + newhandles[1:-4]
            newlabels = changedlabels + newlabels[1:-4]



            sc.legend().remove()
            sc.set_xticks([])
            sc.set_xticklabels([])
            sc.set_xlabel('')
            sc.set_yticks([])
            sc.set_yticklabels([])
            sc.set_ylabel('')
            legend_cols = 4

            if all_blocks:
                newlabels = ['Specialist 1', 'Specialist 2', 'Specialist 3',
                             'Attention All', 'Attention 1', 'Attention 2', 'Attention 3',
                             'Tree-like', 'Tree-like + Gradient Mixing']

            f.legend(handles=flip(newhandles, legend_cols), labels=flip(newlabels, legend_cols),
                     ncol=legend_cols, loc='lower center')
            print()

            plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.15)
            plt.savefig('./plots/mdm_embedding{}{}.png'.format(append_name, '_epoch' + str(
                plot_epoch) if plot_epoch != None else ''), dpi=300)

        else:
            filled_markers = ('o', 'D', 'H', 'p', 'v', '^', '<', '>', 's', '8', 'd', 'P', 'X', '*')
            labels = df_p.name.unique()

            cmap = sns.color_palette("cubehelix", 8)
            only_black = True
            if only_black:
                df_p.Value = 100.

                g = sns.FacetGrid(df_p, col='Metric')
                g.map_dataframe(sns.scatterplot, x='x', y='y', style='style', markers=filled_markers,
                                cmap=cmap, s=50, legend='full').set_titles(
                    "{col_name}").add_legend(label_order=['0', '1', '2', '3', '4', '5', '6', '7', '8'])
            else:
                g = sns.FacetGrid(df_p, col='Metric')
                g.map_dataframe(sns.scatterplot, x='x', y='y', style='style', hue='Value', markers=filled_markers,
                                cmap=cmap, s=50, legend='full').set_titles(
                    "{col_name}").add_legend(label_order=['0', '1', '2', '3', '4', '5', '6', '7', '8'])

            for i in range(len(g._legend.texts)):
                g._legend.texts[i].set_text(labels[i])

            for ax in g.axes.flat:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_xlabel('')
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_ylabel('')

            plt.savefig(
                './plots/mdm_embedding_metrics{}.png'.format('_epoch' + str(plot_epoch) if plot_epoch != None else ''),
                dpi=300)

        # sc.set_title('Models embedded by MDS based on MDM')
        plt.show()


def plot_cluster_tree(mdm, model_names):
    from matplotlib import pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([model.children_, model.distances_,
                                        counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    # setting distance_threshold=0 ensures we compute the full tree.
    model = myAggClustering.AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='average')

    mdm = mdm[:10, :10]

    model = model.fit(mdm)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(model, truncate_mode=None)
    plt.xlabel("Number of points in node (or index of point if no parenthesis.")
    plt.show()


def plot_input_rdm(rdm):
    f = plt.figure()
    plt.matshow(rdm[:100, :100])
    plt.savefig('./plots/sample_input_rdm.png', dpi=300)
    plt.show()

def plot_metrics_bar(metrics_df):
    # metrics_df_ = metrics_df[metrics_df['Metric'] == '$PA$ $uparrow$']
    plt.figure(figsize=(7, 5))
    metric_task = {'$RMSE$ $\\downarrow$': 'Depth', '$SRD$ $\\downarrow$': 'Depth',
                   '$MAD$ $\\downarrow$': 'Surf. Normals', '$MMAD$ $\\downarrow$': 'Surf. Normals',
                   '$PA$ $\\uparrow$': 'Sem. Segm.',
                   '$IoU$ $\\uparrow$': 'Sem. Segm.'}
    # metrics_df['Task'] = metrics_df['Metric'].apply(
    #     lambda x: {'$RMSE$ $\\downarrow$': 'Depth', '$SRD$ $\\downarrow$': 'Depth',
    #                '$MAD$ $\\downarrow$': 'Surf. Normals', '$MMAD$ $\\downarrow$': 'Surf. Normals',
    #                '$PA$ $\\uparrow$': 'Sem. Segm.',
    #                '$IoU$ $\\uparrow$': 'Sem. Segm.'}[x])
    col_len = 3 if len(metrics_df.Metric.unique()) == 9 else 2
    g = sns.FacetGrid(metrics_df, col="Metric", col_wrap=col_len, height=1.5, sharey=False, margin_titles=True)
    hue_order = ['SN ST', 'SN+EQ+A', 'SNA+DWA+A', 'SN+DWA+MTAHD']
    g.map_dataframe(sns.barplot, x='Model', y='Observation', hue='Model', palette='bright', hue_order=hue_order, ci='sd',
                    dodge=False).set_titles("{col_name}").add_legend(ncol=2)

    # g = sns.catplot(x="Model", y="Observation",
    #                 hue="Model", col="Metric",
    #                 data=metrics_df, kind="bar",
    #                 height=4, aspect=.7,
    #                 dodge=False,
    #                 col_wrap=3)

    for ax in g.axes.ravel():
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
        ax.set_xticklabels([])
        ax.set_ylabel(ax.title._text)
        ax.set_title(metric_task[ax.title._text])

    plt.subplots_adjust(0.11, 0.15, 0.95, 0.95)

    bb = g._legend.get_bbox_to_anchor().inverse_transformed(ax.transAxes)
    yOffset = -2.1
    xOffset = -0.8
    bb.y0 += yOffset
    bb.y1 += yOffset
    bb.x0 += xOffset
    bb.x1 += xOffset
    g._legend.set_bbox_to_anchor(bb, transform=ax.transAxes)
    g._legend.get_texts()[0].set_text('Specialist')
    g._legend.get_texts()[1].set_text('Tree-like')
    g._legend.get_texts()[2].set_text('Attention')
    g._legend.get_texts()[3].set_text('Tree-like + Gradient Mixing')

    plt.savefig('./plots/metrics_bars.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    loss_str = 'LOSS FORMAT: SEMANTIC_LOSS | MEAN_IOU PIX_ACC | DEPTH_LOSS | ABS_ERR REL_ERR | NORMAL_LOSS | MEAN MED <11.25 <22.5 <30\n'

    name_model_run = 'mtan_segnet_dwa_adam_run_2'
    model_class = None

    # TODO pay attention to gpu
    global device
    device = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    print('RUN ON GPU: ' + str(device))
    device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    CHECKPOINT_PATH = Path('./logs/{}/model_checkpoints/checkpoint.chk'.format(name_model_run))

    metrics = [
        'segmentation_loss',
        'segmentation_pix_acc',
        'segmentation_miou',

        'depth_loss',
        'depth_abs_err',
        'depth_rel_err',

        'normals_loss',
        'normals_angle_dist_mean',
        'normals_angle_dist_median'
    ]

    model_names = [
        ['mtan_segnet_without_attention_equal_adam_single_task_0_run_1',
         'mtan_segnet_without_attention_equal_adam_single_task_0_run_2',
         'mtan_segnet_without_attention_equal_adam_single_task_0_run_3'],

        ['mtan_segnet_without_attention_equal_adam_single_task_1_run_0',
         'mtan_segnet_without_attention_equal_adam_single_task_1_run_1',
         'mtan_segnet_without_attention_equal_adam_single_task_1_run_2'],

        ['mtan_segnet_without_attention_equal_adam_single_task_2_run_0',
         'mtan_segnet_without_attention_equal_adam_single_task_2_run_1',
         'mtan_segnet_without_attention_equal_adam_single_task_2_run_2'],

        ['mtan_segnet_without_attention_equal_adam_run_0',
         'mtan_segnet_without_attention_equal_adam_run_1',
         'mtan_segnet_without_attention_equal_adam_run_2'],

        # ---------------------------------------------------------------------

        # ['mtan_segnet_without_attention_dwa_adam_run_0',
        #  'mtan_segnet_without_attention_dwa_adam_run_2',
        #  'mtan_segnet_without_attention_dwa_adam_run_3'],
        #
        # ['mtan_segnet_without_attention_gradnorm_adam_run_0',
        #  'mtan_segnet_without_attention_gradnorm_adam_run_1',
        #  'mtan_segnet_without_attention_gradnorm_adam_run_2'],

        ['mtan_segnet_dwa_adam_run_2',
         'mtan_segnet_dwa_adam_run_3',
         'mtan_segnet_dwa_adam_run_4'],

        # ['mtan_segnet_without_attention_dwa_multitask_adam_run_0',
        #  'mtan_segnet_without_attention_dwa_multitask_adam_run_1',
        #  'mtan_segnet_without_attention_dwa_multitask_adam_run_2'],

        ['mtan_segnet_without_attention_dwa_multitask_adam_hd_run_0',
         'mtan_segnet_without_attention_dwa_multitask_adam_hd_run_1',
         'mtan_segnet_without_attention_dwa_multitask_adam_hd_run_2'],

        # ['mtan_segnet_without_attention_dwa_multitask_adam_linear_combination_hd_run_0',
        #  'mtan_segnet_without_attention_dwa_multitask_adam_linear_combination_hd_run_1',
        #  'mtan_segnet_without_attention_dwa_multitask_adam_linear_combination_hd_run_2'],
    ]

    model_names_subsampled = [
        ['mtan_segnet_without_attention_equal_adam_single_task_0_tr_subsampling_0.25run_0',
         'mtan_segnet_without_attention_equal_adam_single_task_0_tr_subsampling_0.5run_0',
         'mtan_segnet_without_attention_equal_adam_single_task_0_tr_subsampling_0.75run_0',
         'mtan_segnet_without_attention_equal_adam_single_task_0_run_3'],

        ['mtan_segnet_without_attention_equal_adam_single_task_1_tr_subsampling_0.25run_0',
         'mtan_segnet_without_attention_equal_adam_single_task_1_tr_subsampling_0.5run_0',
         'mtan_segnet_without_attention_equal_adam_single_task_1_tr_subsampling_0.75run_0',
         'mtan_segnet_without_attention_equal_adam_single_task_1_run_2'],

        ['mtan_segnet_without_attention_equal_adam_single_task_2_tr_subsampling_0.25run_0',
         'mtan_segnet_without_attention_equal_adam_single_task_2_tr_subsampling_0.5run_0',
         'mtan_segnet_without_attention_equal_adam_single_task_2_tr_subsampling_0.75run_0',
         'mtan_segnet_without_attention_equal_adam_single_task_2_run_2'],

        None,

        ['mtan_segnet_without_attention_dwa_adam_tr_subsampling_0.25run_0',
         'mtan_segnet_without_attention_dwa_adam_tr_subsampling_0.5run_0',
         'mtan_segnet_without_attention_dwa_adam_tr_subsampling_0.75run_0',
         'mtan_segnet_without_attention_dwa_adam_run_3'],

        None,

        ['mtan_segnet_dwa_adam_tr_subsampling_0.25run_0',
         'mtan_segnet_dwa_adam_tr_subsampling_0.5run_0',
         'mtan_segnet_dwa_adam_tr_subsampling_0.75run_0',
         'mtan_segnet_dwa_adam_run_4'],

        None,

        ['mtan_segnet_without_attention_dwa_multitask_adam_hd_tr_subsampling_0.25run_0',
         'mtan_segnet_without_attention_dwa_multitask_adam_hd_tr_subsampling_0.5run_0',
         'mtan_segnet_without_attention_dwa_multitask_adam_hd_tr_subsampling_0.75run_0',
         'mtan_segnet_without_attention_dwa_multitask_adam_hd_run_2'],

        None,
    ]

    model_names_finetuning = [
        None,

        None,

        None,

        None,

        ['mtan_segnet_without_attention_dwa_multitask_adam_hd_finetuning_run_0',
         'mtan_segnet_without_attention_dwa_multitask_adam_hd_finetuning_run_1',
         'mtan_segnet_without_attention_dwa_multitask_adam_hd_finetuning_run_2'],

        None,

        ['mtan_segnet_dwa_multitask_adam_hd_finetuning_run_0',
         'mtan_segnet_dwa_multitask_adam_hd_finetuning_run_1',
         'mtan_segnet_dwa_multitask_adam_hd_finetuning_run_2'],

        None,

        None,

        None,
    ]

    # plot_metrics(model_names, metrics, ncols=3, start_epoch=20)
    retrained_heads = [
        ['mtan_segnet_without_attention_equal_adam_single_task_0_retrain_head_run_0'],    # 1 -> 0
        ['mtan_segnet_without_attention_equal_adam_single_task_0_retrain_head_run_1'],    # 2 -> 0
        ['mtan_segnet_without_attention_equal_adam_single_task_1_retrain_head_run_0'],    # 0 -> 1
        ['mtan_segnet_without_attention_equal_adam_single_task_1_retrain_head_run_1'],    # 2 -> 1
        ['mtan_segnet_without_attention_equal_adam_single_task_2_retrain_head_run_0'],    # 0 -> 2
        ['mtan_segnet_without_attention_equal_adam_single_task_2_retrain_head_run_1'],    # 1 -> 2
        ['mtan_segnet_without_attention_dwa_adam_retrain_head_run_0'],                    # sn+dwa+a -> sn+dwa+a
    ]

    # models with checkpoints for RSA
    model_checkpointed_rsa = [
        'mtan_segnet_without_attention_equal_adam_single_task_2_run_3',
        'mtan_segnet_without_attention_equal_adam_single_task_1_run_3',
        'mtan_segnet_dwa_adam_run_5',
        'mtan_segnet_without_attention_equal_adam_single_task_0_run_4',
        'mtan_segnet_without_attention_dwa_adam_run_5',
        'mtan_segnet_without_attention_dwa_multitask_adam_hd_run_4',
    ]

    rsa_during_learning = True
    if rsa_during_learning:
        model_names = model_checkpointed_rsa

    # plot_metrics(retrained_heads, metrics + ['normals_within_11.5', 'normals_within_22.5', 'normals_within_30'],
    #              ncols=3, start_epoch=0, just_values=True)
    bar = False
    if bar:
        metrics_df = plot_metrics(model_names,
                                  metrics,
                                  ncols=3, start_epoch=20, just_values=True, bar=bar)
    else: pass

    # plot_metrics(['mtan_segnet_without_attention_dwa_multitask_adam_hd_run_5'], metrics, ncols=3)

    # TODO
    # plot_metrics(model_names_subsampled, metrics, ncols=3, subsampled=True, best_subsampled=True)
    # plot_metrics(model_names_subsampled, metrics, ncols=3, subsampled=True)
    # plot_metrics(model_names_finetuning, metrics, ncols=3, finetuning=True)

    just_one_model = False
    if just_one_model:
        model_names = [mnl[0] for mnl in model_names]

    reduce = False
    all_blocks = True

    intrinsic_dimension = True
    if intrinsic_dimension:
        intrinsic_dimensions = intrinsic_dimension_from_names(model_names, reduce=reduce, all_blocks=all_blocks,
                                                over_time=rsa_during_learning)

    testing_plot = False
    if testing_plot:
        mdm = np.load('./test_mdm.npy')
    else:
        mdm = correlation_matrix_of_rdms_from_names(model_names, reduce=reduce, all_blocks=all_blocks,
                                                    over_time=rsa_during_learning)
    # plot_input_rdm(rdms_from_names(model_names, reduce=reduce)[0])
    # plot_correlation_matrix_of_rdms(mdm, model_names, reduce=reduce)

    cluster_tree = False
    if cluster_tree:
        plot_cluster_tree(mdm, model_names)

    if bar:
        metrics = metrics_df[metrics_df['Metric'].isin(['$IoU$ $\\uparrow$', '$RMSE$ $\\downarrow$', '$MAD$ $\\downarrow$'])]

        metrics = metrics.groupby(['Model', 'Metric']).mean().reset_index()

        metrics.loc[
            (metrics['Model'] == 'SN ST') & (metrics['Metric'] == '$IoU$ $\\uparrow$'), 'Model'] = 'Single Task 1'
        metrics.loc[len(metrics)] = ['Single Task 1', '$RMSE$ $\\downarrow$', 1.183]
        metrics.loc[len(metrics)] = ['Single Task 1', '$MAD$ $\\downarrow$', 40.5]
        metrics.loc[
            (metrics['Model'] == 'SN ST') & (metrics['Metric'] == '$RMSE$ $\\downarrow$'), 'Model'] = 'Single Task 2'
        metrics.loc[len(metrics)] = ['Single Task 2', '$IoU$ $\\uparrow$', 0.196]
        metrics.loc[len(metrics)] = ['Single Task 2', '$MAD$ $\\downarrow$', 45.3]
        metrics.loc[
            (metrics['Model'] == 'SN ST') & (metrics['Metric'] == '$MAD$ $\\downarrow$'), 'Model'] = 'Single Task 3'
        metrics.loc[len(metrics)] = ['Single Task 3', '$IoU$ $\\uparrow$', 0.196]
        metrics.loc[len(metrics)] = ['Single Task 3', '$RMSE$ $\\downarrow$', 1.203]
        metrics = metrics.pivot_table('Observation', 'Model', 'Metric')
        metrics.reset_index(drop=False, inplace=True)
        # todo add retrained performances
        print()
    else:
        metrics = None

    plot_correlation_matrix_of_rdms_embedding(mdm, model_names, reduce=reduce, metrics=metrics, all_blocks=all_blocks,
                                              over_time=rsa_during_learning,
                                              embedding='umap')

    input_dissimilarity = False
    # test_multitask_adam_mixing_hd_grads(CHECKPOINT_PATH, device, ModelClass=architectures.SegNetWithoutAttention)
    # sys.exit(0)

    one_head = True
    artificial_css, blocks, loss_vs_names = calculate_artificial_css(CHECKPOINT_PATH, device, one_head=one_head)
    plot_artificial_css(artificial_css, blocks, one_head, loss_vs_names)

    # sys.exit(0)

    # plot_input_dissimilarity_from_file(Path('./logs/{}/input_dissimilarity.npy'.format(name_model_run)))

    # corr_matrix = correlation_matrix_of_rdms_from_names([name_model_run for _ in range(10)])
    # plot_correlation_matrix_of_rdms(corr_matrix)

    # plot_mds_embedding_input_rdms(rdms_from_names([name_model_run for _ in range(10)]))

    plot_sample = True

    # TODO pay attention to model classp
    test = False
    if test:
        avg_cost, performance, metrics_per_batch, original_batch_costs, places_challenge_metrics = \
            evaluate_saved_model(CHECKPOINT_PATH, device, ModelClass=model_class, test=test,
                                 input_dissimilarity=input_dissimilarity)
    else:
        ret = \
            evaluate_saved_model(CHECKPOINT_PATH, device, ModelClass=model_class,
                                 input_dissimilarity=input_dissimilarity, plot_sample=plot_sample)
        if input_dissimilarity:
            avg_cost, performance, input_dissimilarity = ret
            write_input_dissimilarity(name_model_run, input_dissimilarity)
        else:
            avg_cost, performance = ret

    write_performance(name_model_run, performance, loss_str)
