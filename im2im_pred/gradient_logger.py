import os
import pickle
from pathlib import Path

import torch.nn.functional as F
from functools import partial


class GradientLogger:
    def __init__(self, n_tasks, layer_name, save_path):
        self.n_tasks = n_tasks
        self.layer_name = layer_name
        self.save_path = Path(save_path)
        self.save_path = self.save_path/'grad_metrics_{}.pickle'.format(layer_name)
        os.makedirs(self.save_path.parent)

        self.grad_list = []
        self.grad_metrics = {}

        for task_ind in range(self.n_tasks):
            self.grad_metrics['norm_grad_task{}'.format(task_ind+1)] = []
            for task_ind_other in range(task_ind+1, self.n_tasks):
                self.grad_metrics['cosine_similarity_grad_task{}_task{}'.format(task_ind+1, task_ind_other+1)] = []

    def update_grad_list(self, grad_tensor):
        self.grad_list.append(grad_tensor.clone().detach().to('cpu'))
        if len(self.grad_list) == self.n_tasks:
            self.add_grad_metrics()
            self.grad_list = []

    def add_grad_metrics(self):
        for task_ind in range(self.n_tasks):
            self.grad_metrics['norm_grad_task{}'.format(task_ind+1)].append(self.grad_list[task_ind].norm())
            for task_ind_other in range(task_ind+1, self.n_tasks):
                self.grad_metrics['cosine_similarity_grad_task{}_task{}'.format(task_ind+1, task_ind_other+1)].append(
                    F.cosine_similarity(self.grad_list[task_ind], self.grad_list[task_ind_other])
                )

    def write_grad_metrics(self, epoch):
        first_write = False
        if not os.path.exists(self.save_path): first_write = True
        with open(self.save_path, 'rb+') as handle:
            if not first_write:
                grad_metrics = pickle.load(handle)
            else:
                grad_metrics = {self.layer_name: {'epoch{}'.format(epoch): {}}}
            grad_metrics[self.layer_name]['epoch{}'.format(epoch)] = self.grad_metrics
            pickle.dump(grad_metrics, handle)
        self.grad_metrics = {}


_ = _.register_hook(GradientLogger(3, 'test_layer', './grad_metrics/').add_grad_metrics)
