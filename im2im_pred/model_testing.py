import os
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


def evaluate_model(model, test_loader, device, index, avg_cost, cost, test=False, input_dissimilarity=False):
    model.eval()

    miou_metric = IntersectionOverUnion(-1, model.class_nb)
    iou_metric = PixelAccuracy(-1, model.class_nb)
    depth_errors = DepthErrors(rmse=True)
    metric_callbacks = [(miou_metric, 0), (iou_metric, 0), (depth_errors, 1)]

    area_intersections = []
    area_unions = []
    pixel_accuracies = []
    pixel_corrects = []
    pixel_labeled = []

    if test:
        original_batch_costs = np.zeros(4, np.float32)

        metric_batches = {miou_metric: [],
                          iou_metric: []}

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
                    if cb is miou_metric or cb is iou_metric:
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

            # TODO probably this averaging is not okay for these metrics
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
            pixel_accuracies = np.array(pixel_accuracies)
            pixel_corrects = np.array(pixel_corrects)
            pixel_labeled = np.array(pixel_labeled)

            IoU = 1.0 * np.sum(area_intersections, axis=0) / np.sum(np.spacing(1) + area_unions, axis=0)
            mean_pixel_accuracy = 1.0 * np.sum(pixel_corrects) / (np.spacing(1) + np.sum(pixel_labeled))

        avg_cost[1] = miou_metric.metric
        avg_cost[2] = iou_metric.metric
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
    optimizer = MultitaskAdamMixingHD(model.parameters(), lr=1e-4*math.sqrt(2), test_grad_eps=test_grad_eps,
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
    optimizer = MultitaskAdamMixingHD(model.parameters(), lr=1e-4*math.sqrt(2), test_grad_eps=-test_grad_eps,
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


def calculate_artificial_css(CHECKPOINT_PATH, device, ModelClass=None, one_head=True, **kwargs):
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
            output = output[0]
            target = torch.rand(1).to(device) * torch.ones_like(output)
            loss_1 = F.mse_loss(output, target)
            loss_2 = F.l1_loss(output, target)
            loss_1_square = F.mse_loss(output, target)**2
        else:
            rand_scalar = torch.rand(1).to(device)
            loss_1 = F.mse_loss(output[0], rand_scalar * torch.ones_like(output[0]))
            loss_2 = F.l1_loss(output[1], rand_scalar * torch.ones_like(output[1]))
            loss_1_square = F.mse_loss(output[2], rand_scalar * torch.ones_like(output[2])) ** 2
        loss_1.backward(retain_graph=True)
        loss_2.backward(retain_graph=True)
        loss_1_square.backward()
        model.calculate_metrics_gradient_loggers()
    artificial_css = [[[] for _ in range(3)] for _ in range(len(model.gradient_loggers))]
    for i, gradient_logger in enumerate(model.gradient_loggers):
        artificial_css[i][0] += gradient_logger.grad_metrics['cosine_similarity_grad_weights_task1_task2']
        artificial_css[i][1] += gradient_logger.grad_metrics['cosine_similarity_grad_weights_task1_task3']
        artificial_css[i][2] += gradient_logger.grad_metrics['cosine_similarity_grad_weights_task2_task3']
    artificial_css = np.array(artificial_css)
    return artificial_css


def plot_artificial_css(artificial_css, one_head):
    x = xr.DataArray(artificial_css, dims=('layers', 'loss_kinds', 'observations'),
                     coords={'layers': [i for i in range(artificial_css.shape[0])],
                             'loss_kinds': ['mse_vs_l1',
                                            'mse_vs_mse_sq',
                                            'l1_vs_mse_sq'],
                             'observations': [i for i in
                                              range(artificial_css.shape[-1])]})
    df = x.to_dataframe('artificial_css').reset_index()
    plt.figure()
    ax = sns.barplot(x="layers", y="artificial_css", hue='loss_kinds', data=df)
    ax.set_title('One Head' if one_head else 'Separate Heads')
    plt.show()
    plt.savefig(Path('./logs/artificial_css/{}.pdf'.format('one_head' if one_head else 'separate_heads')))


def evaluate_saved_model(CHECKPOINT_PATH, device, ModelClass=None, test=False, input_dissimilarity=False, **kwargs):
    dataset_path = 'data/nyuv2'
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
    with open(PERFORMANCE_PATH / 'final_performance.txt', 'w') as handle:
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


import matplotlib.pyplot as plt

plt.interactive(False)
import math


def plot_metrics(CHECKPOINT_PATH, metrics):
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

    metrics_data = torch.load(CHECKPOINT_PATH)['avg_cost']
    metrics_data = metrics_data[~np.all(metrics_data == 0, axis=1)]

    ncols = 2
    fig, axs = plt.subplots(math.ceil(len(metrics) / ncols), ncols)
    axs = axs.reshape(-1)
    fig.suptitle('Metrics')

    nepochs = metrics_data.shape[0]

    xtrain = np.arange(0.5, 0.5 + nepochs)
    xtest = np.arange(1, 1 + nepochs)

    for i, m in enumerate(metrics):
        colors = sns.color_palette()

        axs[i].plot(xtrain, metrics_data[:, metric_name_to_ind[m]], '--', color=colors[i % len(colors)],
                    label=m + '_train')
        axs[i].plot(xtest, metrics_data[:, metric_name_to_ind[m] + 12], color=colors[i % len(colors)],
                    label=m + '_test')
        opt_ind = metric_name_to_opt_funcs[m][0](metrics_data[:, metric_name_to_ind[m] + 12])
        opt = metric_name_to_opt_funcs[m][1](metrics_data[:, metric_name_to_ind[m] + 12])
        axs[i].scatter([xtest[opt_ind]], [opt], facecolors='none', marker='o', color=colors[i % len(colors)])
        axs[i].axvline(x=xtest[opt_ind], color=colors[i % len(colors)], alpha=0.5)
        axs[i].axhline(y=opt, color=colors[i % len(colors)], alpha=0.5)
        axs[i].legend()

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

    name_model_run = 'mtan_segnet_without_attention_dwa_multitask_adam_hd_run_4'
    # TODO pay attention to gpu
    device = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    print('RUN ON GPU: ' + str(device))
    device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")

    CHECKPOINT_PATH = Path('./logs/{}/model_checkpoints/checkpoint.chk'.format(name_model_run))

    metrics = [
        'segmentation_loss',
        'segmentation_pix_acc',

        'depth_loss',
        'depth_abs_err',

        'normals_loss',
        'normals_angle_dist_mean'
    ]
    # plot_metrics(CHECKPOINT_PATH, metrics)

    input_dissimilarity = False

    test_multitask_adam_mixing_hd_grads(CHECKPOINT_PATH, device, ModelClass=architectures.SegNetWithoutAttention)
    sys.exit(0)

    one_head = False
    artificial_css = calculate_artificial_css(CHECKPOINT_PATH, device, ModelClass=architectures.SegNetWithoutAttention,
                                              one_head=one_head)
    plot_artificial_css(artificial_css, one_head)

    sys.exit(0)

    # plot_input_dissimilarity_from_file(Path('./logs/{}/input_dissimilarity.npy'.format(name_model_run)))

    corr_matrix = correlation_matrix_of_rdms_from_names([name_model_run for _ in range(10)])
    plot_correlation_matrix_of_rdms(corr_matrix)

    plot_mds_embedding_input_rdms(rdms_from_names([name_model_run for _ in range(10)]))

    # TODO pay attention to model class
    test = False
    if test:
        avg_cost, performance, metrics_per_batch, original_batch_costs, places_challenge_metrics = \
            evaluate_saved_model(CHECKPOINT_PATH, device, ModelClass=architectures.SegNetWithoutAttention, test=test,
                                 input_dissimilarity=input_dissimilarity)
    else:
        ret = \
            evaluate_saved_model(CHECKPOINT_PATH, device, ModelClass=architectures.SegNetWithoutAttention,
                                 input_dissimilarity=input_dissimilarity)
        if input_dissimilarity:
            avg_cost, performance, input_dissimilarity = ret
            write_input_dissimilarity(name_model_run, input_dissimilarity)
        else:
            avg_cost, performance = ret

    write_performance(name_model_run, performance, loss_str)
