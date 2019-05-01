import os
import pickle
from pathlib import Path

import torch.nn.functional as F


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
        self.grad_list_biases.append(grad_input[2].clone().detach().to('cpu'))
        if len(self.grad_list_weights) == self.n_tasks:
            self.add_grad_metrics()
            self.grad_list_weights = []
            self.grad_list_biases = []

    def add_grad_metrics(self):
        for name, grad_list in [('weights', self.grad_list_weights), ('biases', self.grad_list_biases)]:
            for task_ind in range(self.n_tasks):
                self.grad_metrics['norm_grad_{}_task{}'.format(name, task_ind+1)].append(
                    grad_list[task_ind].norm().item())
                for task_ind_other in range(task_ind+1, self.n_tasks):
                    self.grad_metrics['cosine_similarity_grad_{}_task{}_task{}'.format(name, task_ind + 1,
                                                                                       task_ind_other + 1)].append(
                        F.cosine_similarity(grad_list[task_ind], grad_list[task_ind_other], dim=0).item()
                    )

    def write_grad_metrics(self, epoch):
        if os.path.exists(self.save_path):
            with open(self.save_path, 'rb') as handle:
                grad_metrics = pickle.load(handle)
        else:
            grad_metrics = {}
        if self.layer_name not in grad_metrics:
            grad_metrics[self.layer_name] = {'epoch{:03d}'.format(epoch): {}}

        with open(self.save_path, 'wb+') as handle:
            grad_metrics[self.layer_name]['epoch{:03d}'.format(epoch)] = self.grad_metrics
            pickle.dump(grad_metrics, handle)

        self.init_grad_metrics()


# _ = _.register_hook(GradientLogger(3, 'test_layer', './grad_metrics/').add_grad_metrics)
