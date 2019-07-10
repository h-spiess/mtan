import math
import os
import pickle
from functools import partial
from pathlib import Path

import torch
from torch.optim import Optimizer


def reduce_add(l, factors=None):
    sum = l[0] if not factors else factors[0] * l[0]
    for i, e in enumerate(l[1:]):
        sum += e if not factors else factors[i+1] * l[0]
    return sum


class MultitaskOptimizer(Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)
        self.add_hooks()

    def add_hooks(self):
        # TODO: activate hooks for all params, that track the successive backward passes
        # - when called: received one backward pass -> for each received backward pass: add or update one set of metrics

        # TODO: call zero_grad per hand in train loop -> not possible with hook
        self.state['task_grads'] = {}

        for group in self.param_groups:
            for p in group['params']:   #  this is ordered and remains in that order
                self.state['task_grads'][p] = []
                p.register_hook(partial(self.append_grad_to_state_list, p))

    def append_grad_to_state_list(self, p, grad_input):
        self.state['task_grads'][p].append(grad_input.detach().clone())

    def clear_task_grads(self, p):
        self.state['task_grads'][p].clear()

    def remove_first_task_grad(self, p):
        self.state['task_grads'][p].pop(0)

    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        raise NotImplementedError

    def update_optimizer_data_logs(self, named_parameters, epoch):
        if not self.parameter_name_map:
            self.parameter_name_map = {p: n for n, p in named_parameters}
        optimizer_logs_with_names = {}
        for key, value in self.optimizer_logs.items():
            optimizer_logs_with_names[self.parameter_name_map[key]] = value
        self.optimizer_logs_save['epoch{:03d}'.format(epoch)] = optimizer_logs_with_names
        self.optimizer_logs = {}

    def log_optimizer_data(self, save_path):
        save_path = Path(save_path)/'optimizer_logs.pickle'

        if not os.path.exists(save_path.parent):
            os.makedirs(save_path.parent)

        if os.path.exists(save_path):
            with open(save_path, 'rb') as handle:
                optimizer_logs = pickle.load(handle)
        else:
            optimizer_logs = {}

        with open(save_path, 'wb+') as handle:
            optimizer_logs.update(self.optimizer_logs_save)
            pickle.dump(optimizer_logs, handle)

        self.optimizer_logs_save = {}


class MultitaskAdam(MultitaskOptimizer):
    r"""Implements Multitask Adam algorithm.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, hypergrad_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if hypergrad_lr and not 0.0 <= hypergrad_lr:
            raise ValueError("Invalid hypergradient learning rate: {}".format(hypergrad_lr))
        if hypergrad_lr and amsgrad:
            raise ValueError("Hypergradient learning descent not compatible with amsgrad for now")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, hypergrad_lr=hypergrad_lr)
        super().__init__(params, defaults)
        if hypergrad_lr:
            self.logging = True
            self.parameter_name_map = None
            self.optimizer_logs = {}
            self.optimizer_logs_save = {}

    def __setstate__(self, state):
        # TODO
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                task_grads = self.state['task_grads'][p]

                if task_grads[0].is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                hypergrad_lr = group['hypergrad_lr']

                state = self.state[p]
                n_tasks = len(task_grads)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values per task
                    state['exp_avg'] = [torch.zeros_like(p.data)] * n_tasks
                    # Exponential moving average of squared gradient values per task
                    state['exp_avg_sq'] = [torch.zeros_like(p.data)] * n_tasks
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values per task
                        state['max_exp_avg_sq'] = [torch.zeros_like(p.data)] * n_tasks

                    if hypergrad_lr:
                        # per parameter learning rate
                        state['lr'] = [group['lr']] * n_tasks

                beta1, beta2 = group['betas']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction = math.sqrt(bias_correction2) / bias_correction1

                step_size = group['lr']

                for i, task_grad in enumerate(task_grads):
                    exp_avg, exp_avg_sq = state['exp_avg'][i], state['exp_avg_sq'][i]
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq'][i]

                    if group['weight_decay'] != 0:
                        task_grad.add_(group['weight_decay']/n_tasks, p.data)

                    if hypergrad_lr and state['step'] > 1:
                        prev_bias_correction1 = 1 - beta1 ** (state['step'] - 1)
                        prev_bias_correction2 = 1 - beta2 ** (state['step'] - 1)
                        # Hypergradient for Adam:
                        h = torch.dot(task_grad.view(-1), torch.div(exp_avg, exp_avg_sq.sqrt().add_(group['eps'])).view(-1)) * math.sqrt(prev_bias_correction2) / prev_bias_correction1
                        # Hypergradient descent of the learning rate:
                        state['lr'][i] += group['hypergrad_lr'] * h
                        step_size = state['lr'][i]

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(1 - beta1, task_grad)
                    exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, task_grad, task_grad)
                    self.remove_first_task_grad(p)
                    if amsgrad:
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                    else:
                        denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p.data.addcdiv_(-step_size*bias_correction, exp_avg, denom)

                if hypergrad_lr:
                    if p in self.optimizer_logs:
                        self.optimizer_logs[p].append(scalar_tensor_list_to_item_list(state['lr']))
                    else:
                        self.optimizer_logs[p] = [scalar_tensor_list_to_item_list(state['lr'])]

        return loss


class MultitaskAdamHD(MultitaskAdam):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hypergrad_lr (float, optional): hypergradient learning rate for the online
        tuning of the learning rate, introduced in the paper
        `Online Learning Rate Adaptation with Hypergradient Descent`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Online Learning Rate Adaptation with Hypergradient Descent:
        https://openreview.net/forum?id=BkrsAzWAb
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, hypergrad_lr=1e-7):
        super(MultitaskAdamHD, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                              amsgrad=False, hypergrad_lr=hypergrad_lr)


class MultitaskAdamMixingHD(MultitaskOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, hypergrad_lr=1e-7, normalize_mixing_weights=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= hypergrad_lr:
            raise ValueError("Invalid hypergradient learning rate: {}".format(hypergrad_lr))

        defaults = dict(lr=lr, betas=betas, eps=eps, hypergrad_lr=hypergrad_lr,
                        normalize_mixing_weights=normalize_mixing_weights)
        super().__init__(params, defaults)
        if hypergrad_lr:
            self.logging = True
            self.parameter_name_map = None
            self.optimizer_logs = {}
            self.optimizer_logs_save = {}

    def __setstate__(self, state):
        # TODO
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                task_grads = self.state['task_grads'][p]

                if task_grads[0].is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                hypergrad_lr = group['hypergrad_lr']

                state = self.state[p]
                n_tasks = len(task_grads)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['mixing_factors'] = [1.] * n_tasks

                beta1, beta2 = group['betas']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                mixed_grad = state['mixing_factors'][0] * task_grads[0]
                unmixed_grad = reduce_add(task_grads)

                if hypergrad_lr and n_tasks > 1:
                    state['prev_task_grads'] = self.state['task_grads'][p]

                if hypergrad_lr and state['step'] > 1 and n_tasks > 1:
                    # from previous time step
                    prev_task_grads = state['prev_task_grads']
                    prev_bias_correction1 = 1 - beta1 ** (state['step'] - 1)
                    prev_bias_correction2 = 1 - beta2 ** (state['step'] - 1)
                    prev_mixed_grad = reduce_add(prev_task_grads, state['mixing_factors'])

                    prev_exp_avg_corrected = exp_avg / prev_bias_correction1
                    prev_exp_avg_sq_corrected_sqrt = (exp_avg_sq / prev_bias_correction2).sqrt_()

                    # Hypergradient for mixing_factor:

                    # easier to test and read but slower

                    # beta_1_ratio = (1-beta1)/prev_bias_correction1
                    # beta_2_ratio = (1-beta2)/prev_bias_correction2
                    # h_mixing_const_2 = -group['lr'] * ((beta_1_ratio * (prev_exp_avg_sq_corrected_sqrt+group['eps'])-prev_exp_avg_corrected*beta_2_ratio*(1/prev_exp_avg_sq_corrected_sqrt)*prev_mixed_grad)/(prev_exp_avg_sq_corrected_sqrt + group['eps'])**2) * unmixed_grad

                    h_mixing_const = prev_exp_avg_corrected.mul_((1 - beta2) / prev_bias_correction2).div_(
                        prev_exp_avg_sq_corrected_sqrt + group['eps']).mul_(prev_mixed_grad).sub_(  # epsilon in this row is for preventing zero division
                        ((1 - beta1) / prev_bias_correction1) * (
                                prev_exp_avg_sq_corrected_sqrt + group['eps'])).div_(
                        (prev_exp_avg_sq_corrected_sqrt + group['eps']) ** 2)
                    h_mixing_const.mul_(group['lr']).mul_(unmixed_grad)

                    del prev_exp_avg_corrected
                    del prev_exp_avg_sq_corrected_sqrt
                    del prev_mixed_grad

                    for i, task_grad in enumerate(task_grads):
                        h_mixing = torch.dot(h_mixing_const.view(-1), prev_task_grads[i].view(-1)).item()
                        state['mixing_factors'][i] -= group['hypergrad_lr'] * h_mixing

                if group['normalize_mixing_weights']:
                    sum_mixing_weights = 0
                    for i in range(len(task_grads)):
                        sum_mixing_weights += abs(state['mixing_factors'][i])
                    for i in range(len(task_grads)):
                        state['mixing_factors'][i] *= len(task_grads) / sum_mixing_weights

                for i, task_grad in enumerate(task_grads):
                    if i > 0:
                        mixed_grad += state['mixing_factors'][i] * task_grads[i]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, mixed_grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, mixed_grad, mixed_grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                p.data.addcdiv_(-step_size, exp_avg, denom)

                self.state['task_grads'][p] = []

                if hypergrad_lr:
                    if p in self.optimizer_logs:
                        self.optimizer_logs[p].append(scalar_tensor_list_to_item_list(state['mixing_factors']))
                    else:
                        self.optimizer_logs[p] = [scalar_tensor_list_to_item_list(state['mixing_factors'])]

        return loss


def scalar_tensor_list_to_item_list(tensor_list):
    return [tensor.item() if type(tensor) is not float else tensor for tensor in tensor_list]