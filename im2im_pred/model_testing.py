import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import architectures
from create_dataset import NYUv2
from metrics import IntersectionOverUnion, PixelAccuracy, DepthErrors


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
    imPred = imPred * ((imLab != missing)*2 - 1)

    # Compute area intersection:
    intersection = imPred * ((imPred == imLab)*2 - 1)
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


def evaluate_model(model, test_loader, device, index, avg_cost, cost, test=False):
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
            return avg_cost, performance, [np.array(l).mean() for _, l in metric_batches.items()], original_batch_costs, \
                   [mean_pixel_accuracy, IoU]
        else:
            return avg_cost, performance


def load_model(CHECKPOINT_PATH, ModelClass, device, **kwargs):
    model = ModelClass(device, **kwargs)

    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_saved_model(CHECKPOINT_PATH, ModelClass, device, test=False, **kwargs):
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

    model = load_model(CHECKPOINT_PATH, ModelClass, device, **kwargs)
    return evaluate_model(model, nyuv2_test_loader, device, -1, test_avg_cost, test_cost, test=test)


def write_performance(name_model_run, performance, loss_str):
    PERFORMANCE_PATH = Path('./logs/{}/'.format(name_model_run))
    os.makedirs(PERFORMANCE_PATH, exist_ok=True)
    with open(PERFORMANCE_PATH / 'final_performance.txt', 'w') as handle:
        handle.write(loss_str)
        handle.write(performance)


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

    for m in metrics:
        assert m in metric_name_to_ind, 'Unknown metric: ' + str(m)

    metrics_data = torch.load(CHECKPOINT_PATH)['avg_cost']
    metrics_data = metrics_data[~np.all(metrics_data == 0, axis=1)]

    ncols = 2
    fig, axs = plt.subplots(math.ceil(len(metrics) / ncols), ncols)
    axs = axs.reshape(-1)
    fig.suptitle('Metrics')

    nepochs = metrics_data.shape[0]
    xtrain = np.linspace(0.5, 0.5 + nepochs, nepochs)
    xtest = np.linspace(1, 1 + nepochs, nepochs)

    for i, m in enumerate(metrics):
        axs[i].plot(xtrain, metrics_data[:, metric_name_to_ind[m]], label=m + '_train')
        axs[i].plot(xtest, metrics_data[:, metric_name_to_ind[m] + 12], label=m + '_test')
        axs[i].legend()

    plt.show()


if __name__ == '__main__':
    loss_str = 'LOSS FORMAT: SEMANTIC_LOSS | MEAN_IOU PIX_ACC | DEPTH_LOSS | ABS_ERR REL_ERR | NORMAL_LOSS | MEAN MED <11.25 <22.5 <30\n'

    name_model_run = 'mtan_segnet_without_attention_dwa_multitask_adam_hd_run_4'
    # TODO pay attention to gpu
    device = torch.device("cuda:{}".format(1) if torch.cuda.is_available() else "cpu")

    CHECKPOINT_PATH = Path('./logs/{}/model_checkpoints/checkpoint.chk'.format(name_model_run))

    metrics = [
        'segmentation_loss',
        'segmentation_pix_acc',

        'depth_loss',
        'depth_abs_err',

        'normals_loss',
        'normals_angle_dist_mean'
    ]
    plot_metrics(CHECKPOINT_PATH, metrics)

    # TODO pay attention to model class
    avg_cost, performance, metrics_per_batch, original_batch_costs, places_challenge_metrics = evaluate_saved_model(CHECKPOINT_PATH, architectures.SegNetWithoutAttention, device, test=True)

    # TODO Pixel accuracy from places is okay

    write_performance(name_model_run, performance, loss_str)
