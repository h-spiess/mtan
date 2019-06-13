import os
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from sklearn.decomposition import IncrementalPCA


def css_whole_gradient(grad1, grad2):
    return F.cosine_similarity(grad1.flatten(), grad2.flatten(), dim=0).item()


def css_separate_filter(grad1, grad2):
    dim = 0
    if len(grad1.size()) != 1 and len(grad2.size()) != 1:
        grad1 = grad1.flatten(start_dim=1, end_dim=-1)
        grad2 = grad2.flatten(start_dim=1, end_dim=-1)
        dim = 1
    return F.cosine_similarity(grad1, grad2, dim=dim).mean().item()


def euclidean_distance(grad1, grad2):
    return (grad1-grad2).norm().item()


def chebyshev_distance(grad1, grad2):
    return (grad1-grad2).abs().max().item()


def kohonen_similarity(grad1, grad2):
    inner = (grad1.flatten() @ grad2.flatten())
    return (inner / (inner + (grad1 - grad2).norm()**2)).item()


class GradientLogger:
    created_save_path_dir = False

    def __init__(self, n_tasks, layer_name, save_path, use_incremental_pca=True):
        self.n_tasks = n_tasks
        self.layer_name = layer_name
        self.save_path = Path(save_path)
        self.save_path = self.save_path / 'grad_metrics_{}.pickle'.format(layer_name)
        if not GradientLogger.created_save_path_dir:
            os.makedirs(self.save_path.parent)
            GradientLogger.created_save_path_dir = True

        self.grad_list_weights = []
        self.grad_list_biases = []

        self.grad_metrics_save = {}

        self.similarity_metrics = [
            ('cosine_similarity_grad_{}_task{}_task{}', css_whole_gradient),
            ('cosine_similarity_separate_filter_grad_{}_task{}_task{}', css_separate_filter),
            # ('euclidean_distance_grad_{}_task{}_task{}', euclidean_distance),
            # ('chebyshev_distance_grad_{}_task{}_task{}', chebyshev_distance),
            ('kohonen_similarity_grad_{}_task{}_task{}', kohonen_similarity)
        ]

        self.use_incremental_pca = use_incremental_pca

        self.init_grad_metrics()

    def init_grad_metrics(self):
        self.grad_metrics = {}

        for name in ['weights', 'biases']:
            for task_ind in range(self.n_tasks):
                self.grad_metrics['norm_grad_{}_task{}'.format(name, task_ind + 1)] = []
                for task_ind_other in range(task_ind + 1, self.n_tasks):
                    for similarity_metric, _ in self.similarity_metrics:
                        self.grad_metrics[
                            similarity_metric.format(name, task_ind + 1, task_ind_other + 1)] = []

        if self.use_incremental_pca:
            self.grad_metrics['incremental_pca_embedding'] = []

    def update_grad_list(self, module, grad_input, grad_output):
        # weights not 3x3, bc. grads from backward pass, not the ones applied to conv params
        if len(grad_input) == 1:
            ind = 0
        else:
            ind = 1
        self.grad_list_weights.append(grad_input[ind].clone().detach().to('cpu'))
        if hasattr(module, 'bias') and module.bias is not None:
            self.grad_list_biases.append(grad_input[2].clone().detach().to('cpu'))
        if len(self.grad_list_weights) == self.n_tasks:
            self.add_grad_metrics('weights')
            self.grad_list_weights = []
            if hasattr(module, 'bias') and module.bias is not None:
                self.add_grad_metrics('biases')
                self.grad_list_biases = []

    def update_grad_list_param_weights(self, grad_input):
        self.grad_list_weights.append(grad_input.clone().detach().to('cpu'))
        if len(self.grad_list_weights) == self.n_tasks:
            self.add_grad_metrics('weights')
            self.grad_list_weights = []

    def update_grad_list_param_biases(self, grad_input):
        self.grad_list_biases.append(grad_input.clone().detach().to('cpu'))
        if len(self.grad_list_biases) == self.n_tasks:
            self.add_grad_metrics('biases')
            self.grad_list_biases = []

    def add_grad_metrics(self, param_kind):
        param_kinds = {'weights': ('weights', self.grad_list_weights),
                       'biases': ('biases', self.grad_list_biases)}
        assert param_kind in param_kinds, 'Not valid param_kind: ' + param_kind

        for name, grad_list in [param_kinds[param_kind]]:
            for task_ind in range(len(grad_list)):
                self.grad_metrics['norm_grad_{}_task{}'.format(name, task_ind + 1)].append(
                    grad_list[task_ind].flatten().norm().item())
                for task_ind_other in range(task_ind + 1, self.n_tasks):
                    for similarity_metric, similarity_func in self.similarity_metrics:
                        self.grad_metrics[similarity_metric.format(name, task_ind + 1,
                                                                   task_ind_other + 1)].append(
                            similarity_func(grad_list[task_ind], grad_list[task_ind_other])
                        )

        if param_kind == 'weights' and self.use_incremental_pca:
            self.grad_metrics['incremental_pca_embedding'].append(self.incremental_pca())

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

    def incremental_pca(self):
        if not hasattr(self, 'ipca'):
            self.ipca = IncrementalPCA(n_components=2, batch_size=self.n_tasks)
        X = torch.stack(self.grad_list_weights).flatten(start_dim=1).numpy()
        self.ipca.partial_fit(X)
        return self.ipca.transform(X)


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
    add_grad_hook_conv_params(last_conv(module), n_tasks, name, grad_save_path, gradient_loggers)


def add_grad_hook_conv_params(module, n_tasks, name, grad_save_path, gradient_loggers):
    assert hasattr(module, 'weight') and hasattr(module, 'bias'), \
        'Conv hook can only be applied to Conv layer with weight and bias.'

    grad_logger = append_and_return(gradient_loggers, GradientLogger(n_tasks, name, grad_save_path))

    module.weight.register_hook(
        grad_logger.update_grad_list_param_weights)

    module.bias.register_hook(
        grad_logger.update_grad_list_param_biases)


def add_grad_hook(module, n_tasks, name, grad_save_path, gradient_loggers):
    module.register_backward_hook(
        append_and_return(gradient_loggers, GradientLogger(n_tasks, name, grad_save_path)).update_grad_list)


def append_and_return(gradient_loggers, gradient_logger):
    gradient_loggers.append(gradient_logger)
    return gradient_loggers[-1]

# _ = _.register_hook(GradientLogger(3, 'test_layer', './grad_metrics/').add_grad_metrics)
