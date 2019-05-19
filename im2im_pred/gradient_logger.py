import os
import pickle
from pathlib import Path

import torch.nn.functional as F
from torch import nn


class GradientLogger:
    created_save_path_dir = False

    def __init__(self, n_tasks, layer_name, save_path):
        self.n_tasks = n_tasks
        self.layer_name = layer_name
        self.save_path = Path(save_path)
        self.save_path = self.save_path/'grad_metrics_{}.pickle'.format(layer_name)
        if not GradientLogger.created_save_path_dir:
            os.makedirs(self.save_path.parent)
            GradientLogger.created_save_path_dir = True

        self.grad_list_weights = []
        self.grad_list_biases = []

        self.grad_metrics_save = {}
        self.init_grad_metrics()

    def init_grad_metrics(self):
        self.grad_metrics = {}

        for name in ['weights', 'biases']:
            for task_ind in range(self.n_tasks):
                self.grad_metrics['norm_grad_{}_task{}'.format(name, task_ind + 1)] = []
                for task_ind_other in range(task_ind + 1, self.n_tasks):
                    self.grad_metrics[
                        'cosine_similarity_grad_{}_task{}_task{}'.format(name, task_ind + 1, task_ind_other + 1)] = []

    def update_grad_list(self, module, grad_input, grad_output):
        self.grad_list_weights.append(grad_input[1].clone().detach().flatten().to('cpu'))
        if module.bias is not None:
            self.grad_list_biases.append(grad_input[2].clone().detach().to('cpu'))
        if len(self.grad_list_weights) == self.n_tasks:
            self.add_grad_metrics()
            self.grad_list_weights = []
            self.grad_list_biases = []

    def add_grad_metrics(self):
        for name, grad_list in [('weights', self.grad_list_weights), ('biases', self.grad_list_biases)]:
            for task_ind in range(len(grad_list)):
                self.grad_metrics['norm_grad_{}_task{}'.format(name, task_ind+1)].append(
                    grad_list[task_ind].norm().item())
                for task_ind_other in range(task_ind+1, len(grad_list)):
                    self.grad_metrics['cosine_similarity_grad_{}_task{}_task{}'.format(name, task_ind + 1,
                                                                                       task_ind_other + 1)].append(
                        F.cosine_similarity(grad_list[task_ind], grad_list[task_ind_other], dim=0).item()
                    )

    def update_grad_metrics(self, epoch):
        self.grad_metrics_save['epoch{:03d}'.format(epoch)] = self.grad_metrics
        self.init_grad_metrics()

    def write_grad_metrics(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as handle:
                grad_metrics = pickle.load(handle)
        else:
            grad_metrics = {}
        if self.layer_name not in grad_metrics:
            grad_metrics[self.layer_name] = {}

        with open(self.save_path, 'wb+') as handle:
            grad_metrics[self.layer_name].update(self.grad_metrics_save)
            pickle.dump(grad_metrics, handle)

        self.grad_metrics_save = {}


def last_conv(module):
    if len(list(module.children())) == 0:
        if not isinstance(module, nn.Conv2d):
            raise ValueError('Hook on non-sequential non-conv layer.')
        else:
            return module
    for layer in reversed(list(module.children())):
        if isinstance(layer, nn.Sequential):  # if sequential layer, apply recursively to layers in sequential layer
            try:
                return last_conv(layer)
            except:  # may not contain a conv layer
                pass
        if isinstance(layer, nn.Conv2d):  # if leaf node, add it to list
            return layer
        try:
            return last_conv(layer)
        except:
            pass
    raise ValueError('Hook on sequential with recursive inner non-conv layers.')


def create_last_conv_hook_at(module, n_tasks, name, grad_save_path, gradient_loggers):
    last_conv(module).register_backward_hook(
        append_and_return(gradient_loggers, GradientLogger(n_tasks, name, grad_save_path)).update_grad_list)

def append_and_return(gradient_loggers, gradient_logger):
    gradient_loggers.append(gradient_logger)
    return gradient_loggers[-1]

# _ = _.register_hook(GradientLogger(3, 'test_layer', './grad_metrics/').add_grad_metrics)
