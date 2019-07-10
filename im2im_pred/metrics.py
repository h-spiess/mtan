import numpy as np
import torch
from fastai.callback import Callback, Any, Tensor


class PixelAccuracy(Callback):
    def __init__(self, missing_code, num_classes, mean_over_classes=True):
        super().__init__()
        self.missing_code = missing_code
        self.num_classes = num_classes
        self.name = 'PixelAccuracy'
        self.mean_over_classes = mean_over_classes

    def on_epoch_begin(self, **kwargs: Any):
        self.pixel_corrects = []
        self.pixel_labeled = []

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs:Any):
        for sample in range(last_output.size()[0]):
            _, p_c, p_l = pixelAccuracy(last_output[sample, :, :, :].argmax(dim=0).cpu().numpy(),
                                          last_target[sample, :, :].cpu().numpy())
            self.pixel_corrects.append(p_c)
            self.pixel_labeled.append(p_l)

    def on_epoch_end(self, **kwargs:Any):
        pixel_corrects = np.array(self.pixel_corrects)
        pixel_labeled = np.array(self.pixel_labeled)

        self.metric = 1.0 * np.sum(pixel_corrects) / (np.spacing(1) + np.sum(pixel_labeled))
        if not self.mean_over_classes:
            raise NotImplementedError('PixelAccuracy only available as mean.')


class IntersectionOverUnion(Callback):
    def __init__(self, missing_code, num_classes, mean_over_classes=True):
        super().__init__()
        self.missing_code = missing_code
        self.num_classes = num_classes
        self.mean_over_classes = mean_over_classes
        self.name = 'IoU'

    def on_epoch_begin(self, **kwargs: Any):
        self.area_intersections = []
        self.area_unions = []

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs:Any):
        for sample in range(last_output.size()[0]):
            a_i, a_u = intersectionAndUnion(last_output[sample, :, :, :].argmax(dim=0).cpu().numpy(),
                                            last_target[sample, :, :].cpu().numpy(),
                                            self.num_classes)
            self.area_intersections.append(a_i)
            self.area_unions.append(a_u)

    def on_epoch_end(self, **kwargs:Any):
        area_intersections = np.array(self.area_intersections)
        area_unions = np.array(self.area_unions)

        self.metric = 1.0 * np.sum(area_intersections, axis=0) / np.sum(np.spacing(1) + area_unions, axis=0)
        mask = np.sum(area_unions, axis=0) != 0
        if self.mean_over_classes:
            self.metric_per_class = self.metric.copy()
            self.metric = self.metric[mask].mean()


class DepthErrors(Callback):
    def __init__(self, rmse=True):
        super().__init__()
        self.rmse = rmse
        self.name = 'Depth error{}'.format(' (RMSE)' if self.rmse else '')

    def on_epoch_begin(self, **kwargs: Any):
        self.sum_abs_error = 0
        self.sum_rel_error = 0
        self.sum_non_zero = 0

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs:Any):
        abs_error, rel_error, non_zero = depth_error(last_output, last_target, rmse=self.rmse)
        self.sum_abs_error += abs_error
        self.sum_rel_error += rel_error
        self.sum_non_zero += non_zero

    def on_epoch_end(self, **kwargs:Any):
        self.metric = (self.sum_abs_error / self.sum_non_zero,
                       self.sum_rel_error / self.sum_non_zero)


def depth_error(x_pred, x_output, rmse=True):
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)

    if not rmse:
        abs_err = torch.abs(x_pred_true - x_output_true)
    else:
        abs_err = torch.sqrt((x_pred_true - x_output_true) ** 2)
    rel_error = abs_err/x_output_true

    _ = x_output.to('cpu')
    return torch.sum(abs_err), torch.sum(rel_error), torch.nonzero(binary_mask).size(0)


def intersection_union(imPred, imLab, missing_code, numClass):
    """
    This function takes the prediction and label of a single image, returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > missing_code).long()

    # Compute area intersection:
    intersection = imPred * (imPred == imLab).long()
    area_intersection = torch.histc(intersection.float().cpu(), bins=numClass, min=1, max=numClass)

    # Compute area union:
    area_pred = torch.histc(imPred.float().cpu(), bins=numClass, min=1, max=numClass)
    area_lab = torch.histc(imLab.float().cpu(), bins=numClass, min=1, max=numClass)
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)


# # from places challenge
# def intersectionAndUnion(imPred, imLab, missing_code, numClass):
#
#     """
#     This function takes the prediction and label of a single image, returns intersection and union areas for each class
#     To compute over many images do:
#     for i in range(Nimages):
#         (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
#     IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
#     """
#
#     imPred = np.asarray(imPred)
#     imLab = np.asarray(imLab)
#
#     # Remove classes from unlabeled pixels in gt image.
#     # We should not penalize detections in unlabeled portions of the image.
#     imPred = imPred * (imLab > missing_code)
#
#     # Compute area intersection:
#     intersection = imPred * (imPred == imLab)
#     (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))
#
#     # Compute area union:
#     (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
#     (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
#     area_union = area_pred + area_lab - area_intersection
#
#     return (area_intersection, area_union)

def intersectionAndUnion(imPred, imLab, numClass, missing=-1):
    """
        This function takes the prediction and label of a single image, returns intersection and union areas for each class
        To compute over many images do:
        for i in range(Nimages):
            (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
        IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
        """
    # imPred = np.asarray(imPred)
    # imLab = np.asarray(imLab)

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

    # imPred = np.asarray(imPred)
    # imLab = np.asarray(imLab)

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab != missing)
    pixel_correct = np.sum((imPred == imLab) * (imLab != missing))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled

    return (pixel_accuracy, pixel_correct, pixel_labeled)

def test_iou_fastai_vs_challenge():
    np.random.seed(42)

    nimages = 10

    imPred_output = np.random.randn(nimages, 13, 100, 100)
    imPred = imPred_output.argmax(axis=1)
    imLab = np.random.randint(0, 13, (nimages, 100, 100))

    area_intersection = []
    area_union = []
    for i in range(nimages):
        area_i, area_u = intersectionAndUnion(imPred[i], imLab[i], 12, 12)
        area_intersection.append(area_i)
        area_union.append(area_u)
    area_intersection = np.array(area_intersection).T
    area_union = np.array(area_union).T
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)

    IoU_fastai = IntersectionOverUnion(12, 13)
    IoU_fastai.on_epoch_begin()
    split_batch_at = 5
    IoU_fastai.on_batch_end(torch.FloatTensor(imPred_output[0:split_batch_at]),
                            torch.LongTensor(imLab[0:split_batch_at]))
    IoU_fastai.on_batch_end(torch.FloatTensor(imPred_output[split_batch_at:]),
                            torch.LongTensor(imLab[split_batch_at:]))
    IoU_fastai.on_epoch_end()

    assert np.all(torch.stack(IoU_fastai.area_union).long().numpy().T == area_union), \
        'IuO: Area unions are not the same'
    assert np.all(torch.stack(IoU_fastai.area_intersection).long().numpy().T == area_intersection), \
        'IuO: Area unions are not the same'
    assert np.all(IoU_fastai.metric_per_class.numpy() == IoU), 'IuO for fastai not the same as in the places challenge'


def test_pixel_acc_fastai():
    np.random.seed(42)

    nimages = 10

    imPred_output = np.random.randn(nimages, 13, 100, 100)
    imPred = imPred_output.argmax(axis=1)
    imLab = np.random.randint(0, 13, (nimages, 100, 100))

    Pixel_fastai = PixelAccuracy(12)
    Pixel_fastai.on_epoch_begin()
    split_batch_at = 5
    Pixel_fastai.on_batch_end(torch.FloatTensor(imPred_output[0:split_batch_at]),
                            torch.LongTensor(imLab[0:split_batch_at]))
    Pixel_fastai.on_batch_end(torch.FloatTensor(imPred_output[split_batch_at:]),
                            torch.LongTensor(imLab[split_batch_at:]))
    Pixel_fastai.on_epoch_end()
    pass