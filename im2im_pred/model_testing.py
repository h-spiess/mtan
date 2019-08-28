import os

from matplotlib.lines import Line2D

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
from matplotlib import rc

rc('text', usetex=True)
# charter as first font
plt.rcParams['text.latex.preamble'] = [r'\usepackage[libertine]{newtxmath}']
plt.rcParams['font.serif'][0], plt.rcParams['font.serif'][-2] = plt.rcParams['font.serif'][-2], \
                                                                plt.rcParams['font.serif'][0]
plt.rcParams['font.family'] = 'serif'

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
    for i in range(output.size()[0]):
        l.append(output[i, :, :, :].detach().clone().cpu().numpy())


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


def evaluate_model(model, test_loader, device, index, avg_cost, cost, pixel_acc_mean_over_classes=True, test=False, input_dissimilarity=False):
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
        model.last_shared_layer().register_forward_hook(appender)

    # evaluating test data
    avg_cost.fill(0.), cost.fill(0.)
    with torch.no_grad():  # operations inside don't track history

        for test_data, test_label, test_depth, test_normal in tqdm(test_loader, desc='Testing'):
            test_data, test_label = test_data.to(device), test_label.type(torch.LongTensor).to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)
            test_labels = [test_label, test_depth, test_normal]

            test_pred = model(test_data)
            test_loss = model.model_fit(test_pred[0], test_label, test_pred[1], test_depth, test_pred[2], test_normal)

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
            activations = np.array(l)
            del l
            activations = activations.reshape(activations.shape[0], -1)
            res = correlation_matrix(activations, SPLITROWS=100)
            res = 1 - res  # dissimilarity

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


def calculate_artificial_css(CHECKPOINT_PATH, device, ModelClass=None, one_head=True, segmentation=True, **kwargs):

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
        one_hot = torch.zeros(n, nClasses, h, w).to(tensor.device).scatter_(1, tensor.view(n, 1, h, w), 1)
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

    nrepeat = 12
    for _ in range(nrepeat):
        input = torch.randn(1, 3, 100, 100).to(device)
        output = model.forward(input)
        if one_head:
            output = output[0]  # is log softmax
            # target = torch.rand(1).to(device) * torch.ones_like(output)
            target = torch.randint_like(output, 0, 13)
            if segmentation:
                loss_dice = dice_loss(output, target)
                loss_jaccard = jaccard_loss(output, target)
                target_max = target.argmax(dim=1)
                loss_cross = F.nll_loss(output, target_max.long())
                losses = (loss_dice, loss_jaccard, loss_cross)
                # print(losses)
                loss_vs_names = ('Dice, Jaccard', 'Dice, NLL', 'Jaccard, NLL')
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
    for i, gradient_logger in enumerate(model.gradient_loggers[::-1]):
        artificial_css[i][0] += gradient_logger.grad_metrics['cosine_similarity_grad_weights_task1_task2']
        artificial_css[i][1] += gradient_logger.grad_metrics['cosine_similarity_grad_weights_task1_task3']
        artificial_css[i][2] += gradient_logger.grad_metrics['cosine_similarity_grad_weights_task2_task3']
    artificial_css = np.array(artificial_css)
    return artificial_css, loss_vs_names


def plot_artificial_css(artificial_css, one_head, loss_vs_names):
    x = xr.DataArray(artificial_css, dims=('Depth', 'Losses', 'Observations'),
                     coords={'Depth': [i for i in range(artificial_css.shape[0])],
                             'Losses': list(loss_vs_names),
                             'Observations': [i for i in
                                              range(artificial_css.shape[-1])]})
    df = x.to_dataframe('Cosine similarity').reset_index()
    plt.figure()
    ax = sns.barplot(x="Depth", y="Cosine similarity", hue='Losses', data=df)
    ax.set_title('One Head' if one_head else 'Separate Heads')
    plt.show()
    plt.savefig(Path('./logs/artificial_css/{}.pdf'.format('one_head' if one_head else 'separate_heads')))


def evaluate_saved_model(CHECKPOINT_PATH, device, ModelClass=None, test=False, input_dissimilarity=False, **kwargs):
    dataset_path = 'data/nyuv2'
    #TODO change to train=False again
    nyuv2_test_set = NYUv2(root=dataset_path, train=False)
    batch_size = 2
    num_workers = 2

    nyuv2_test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)

    test_avg_cost = np.zeros(12, dtype=np.float32)
    test_cost = np.zeros(12, dtype=np.float32)

    model = load_model(CHECKPOINT_PATH, device, ModelClass=ModelClass, **kwargs)
    return evaluate_model(model, nyuv2_test_loader, device, -1, test_avg_cost, test_cost, test=test,
                          input_dissimilarity=input_dissimilarity)


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


def write_input_dissimilarity(name_model_run, input_dissimilarity):
    SAVE_PATH = Path('./logs/{}/'.format(name_model_run))
    os.makedirs(SAVE_PATH, exist_ok=True)
    np.save(SAVE_PATH / 'input_dissimilarity.npy', input_dissimilarity)


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


def plot_metrics(name_models, metrics):
    def create_label(name_model):
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

    metric_name_to_ind = {
        'segmentation_loss': 0,
        'segmentation_miou': 1,
        'segmentation_pix_acc': 2,

        'depth_loss': 3,
        'depth_abs_err': 4,
        'depth_rel_err': 5,

        'normals_loss': 6,
        'normals_angle_dist_mean': 7,
        'normals_angle_dist_median': 8,
        'normals_within_11.5': 9,
        'normals_within_22.5': 10,
        'normals_within_30': 11
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

    CHECKPOINT_PATHS = [Path('./logs/{}/model_checkpoints/checkpoint.chk'.format(name_model)) for name_model in name_models]
    metrics_datas = [torch.load(CHECKPOINT_PATH)['avg_cost'] for CHECKPOINT_PATH in CHECKPOINT_PATHS]
    metrics_datas = [metrics_data[~np.all(metrics_data == 0, axis=1)] for metrics_data in metrics_datas]

    ncols = 2
    fig, axs = plt.subplots(math.ceil(len(metrics) / ncols), ncols)
    axs = axs.reshape(-1)
    fig.suptitle('Metrics')

    nepochs = [metrics_data.shape[0] for metrics_data in metrics_datas]
    assert len(set(nepochs)) == 1
    nepochs = nepochs[0]

    xtrain = np.arange(0.5, 0.5 + nepochs)
    xtest = np.arange(1, 1 + nepochs)

    name_models = [create_label(name_model) for name_model in name_models]

    for i, m in enumerate(metrics):
        for j, metrics_data in enumerate(metrics_datas):
            colors = sns.color_palette()

            axs[i].plot(xtrain, metrics_data[:, metric_name_to_ind[m]], '--', color=colors[j % len(colors)])
            axs[i].plot(xtest, metrics_data[:, metric_name_to_ind[m] + 12], color=colors[j % len(colors)],
                        label=name_models[j].replace('_', ' '))
            opt_ind = metric_name_to_opt_funcs[m][0](metrics_data[:, metric_name_to_ind[m] + 12])
            opt = metric_name_to_opt_funcs[m][1](metrics_data[:, metric_name_to_ind[m] + 12])
            axs[i].scatter([xtest[opt_ind]], [opt], facecolors='none', marker='o', color=colors[j % len(colors)])
            axs[i].axvline(x=xtest[opt_ind], color=colors[j % len(colors)], alpha=0.5)
            axs[i].axhline(y=opt, color=colors[j % len(colors)], alpha=0.5)
        axs[i].legend()
        axs[i].set_title(m.replace('_', ' '))

    labels = ['Train', 'Test']
    lines = [Line2D([0], [0], color='black', linewidth=1, linestyle='--' if l == 'Train' else None) for l in labels]
    fig.legend(lines, labels, ncol=2, loc='lower center')
    plt.show()


def correlation_matrix_of_rdms_from_names(model_names):
    return correlation_matrix_of_rdms(rdms_from_names(model_names))


def rdms_from_names(model_names):
    paths = [Path('./logs/{}/input_dissimilarity.npy'.format(name_model_run)) for name_model_run in model_names]
    return rdms_from_paths(paths)


def rdms_from_paths(paths):
    rdms = []
    for p in paths:
        rdms.append(np.load(p))
    return rdms


def correlation_matrix_of_rdms_from_paths(paths):
    return correlation_matrix_of_rdms(rdms_from_paths(paths))


def correlation_matrix_of_rdms(rdms):
    rdms = np.array(rdms)
    rdms = rdms.reshape(rdms.shape[0], -1)
    correlation_matrix_of_rdms, _ = scipy.stats.spearmanr(rdms.T)
    return correlation_matrix_of_rdms


def plot_correlation_matrix_of_rdms(correlation_matrix_of_rdms):
    f = plt.figure()
    plt.imshow(correlation_matrix_of_rdms)
    plt.colorbar()
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()


def plot_mds_embedding_input_rdms(rdms):
    rdms = np.array(rdms)
    rdms = rdms.reshape(rdms.shape[0], -1)
    embedding = sklearn.manifold.MDS(n_components=2)
    rdms_transformed = embedding.fit_transform(rdms)
    f = plt.figure()
    plt.scatter(rdms_transformed[:, 0], rdms_transformed[:, 1])
    plt.show()


if __name__ == '__main__':
    loss_str = 'LOSS FORMAT: SEMANTIC_LOSS | MEAN_IOU PIX_ACC | DEPTH_LOSS | ABS_ERR REL_ERR | NORMAL_LOSS | MEAN MED <11.25 <22.5 <30\n'

    name_model_run = 'mtan_segnet_without_attention_equal_adam_run_1'
    model_class = None

    # TODO pay attention to gpu
    device = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    print('RUN ON GPU: ' + str(device))
    device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    CHECKPOINT_PATH = Path('./logs/{}/model_checkpoints/checkpoint.chk'.format(name_model_run))

    metrics = [
        'segmentation_loss',
        'segmentation_pix_acc',

        'depth_loss',
        'depth_abs_err',

        'normals_loss',
        'normals_angle_dist_mean'
    ]
    plot_metrics([
        'mtan_segnet_without_attention_equal_adam_run_1',
        'mtan_segnet_without_attention_dwa_adam_run_3',
        'mtan_segnet_dwa_adam_run_6'
    ], metrics)

    input_dissimilarity = False

    # test_multitask_adam_mixing_hd_grads(CHECKPOINT_PATH, device, ModelClass=architectures.SegNetWithoutAttention)
    # sys.exit(0)

    # one_head = True
    # artificial_css, loss_vs_names = calculate_artificial_css(CHECKPOINT_PATH, device, ModelClass=model_class,
    #                                           one_head=one_head)
    # plot_artificial_css(artificial_css, one_head, loss_vs_names)
    #
    # sys.exit(0)

    # plot_input_dissimilarity_from_file(Path('./logs/{}/input_dissimilarity.npy'.format(name_model_run)))

    # corr_matrix = correlation_matrix_of_rdms_from_names([name_model_run for _ in range(10)])
    # plot_correlation_matrix_of_rdms(corr_matrix)

    # plot_mds_embedding_input_rdms(rdms_from_names([name_model_run for _ in range(10)]))

    # TODO pay attention to model class
    test = False
    if test:
        avg_cost, performance, metrics_per_batch, original_batch_costs, places_challenge_metrics = \
            evaluate_saved_model(CHECKPOINT_PATH, device, ModelClass=model_class, test=test,
                                 input_dissimilarity=input_dissimilarity)
    else:
        ret = \
            evaluate_saved_model(CHECKPOINT_PATH, device, ModelClass=model_class,
                                 input_dissimilarity=input_dissimilarity)
        if input_dissimilarity:
            avg_cost, performance, input_dissimilarity = ret
            write_input_dissimilarity(name_model_run, input_dissimilarity)
        else:
            avg_cost, performance = ret

    write_performance(name_model_run, performance, loss_str)
