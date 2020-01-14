import math
import os
import pickle
from functools import partial
from pathlib import Path

import torch
from torch import sigmoid
from torch.optim import Optimizer
import numpy as np


def reduce_add(l, factors=None):
    if not factors or isinstance(factors[0], float):
        sum = l[0] if not factors else factors[0] * l[0]
    elif len(l[0].size()) == 1:
        sum = l[0] if not factors else factors[0] * l[0]
    else:
        sum = l[0] if not factors else factors[0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * l[0]
    for i, e in enumerate(l[1:]):
        if not factors or isinstance(factors[i+1], float):
            sum += e if not factors else factors[i+1] * e
        elif len(l[i+1].size()) == 1:
            sum = e if not factors else factors[i+1] * e
        else:
            sum += e if not factors else factors[i + 1].unsqueeze(1).unsqueeze(2).unsqueeze(3) * e
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

        # needed for multitask_adam_mixing_hd
        self.state['prev_task_grads'] = {}

        for group in self.param_groups:
            for p in group['params']:   #  this is ordered and remains in that order
                self.state['task_grads'][p] = []
                self.state['prev_task_grads'][p] = []
                p.register_hook(partial(self.append_grad_to_state_list, p))

    def append_grad_to_state_list(self, p, grad_input):
        self.state['task_grads'][p].append(grad_input.detach().clone())

    def clear_task_grads(self, p):
        self.state['task_grads'][p].clear()

    def remove_first_task_grad(self, p):
        self.state['task_grads'][p][0] = None

    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Arguments:
            closure (callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        raise NotImplementedError

    def update_optimizer_data_logs(self, named_parameters, epoch):
        # TODO use this function alter half an epoch to start new average

        def recurse_list_to_np_array_and_divide(l, divider):
            for e in l:
                if not isinstance(e, list):
                    e /= divider
                else:
                    recurse_list_to_np_array_and_divide(e, divider)

        # TODO alternative to dict: sqlitedict
        for group in self.param_groups:
            if not group['parameter_name_map']:
                group['parameter_name_map'] = {p: n for n, p in named_parameters if
                                               group['param_to_log_map'].get(p) and any(
                                                   p is p_g for p_g in group['params'])}
            optimizer_logs_with_names = {}
            for key, value in group['optimizer_logs'].items():
                recurse_list_to_np_array_and_divide(value, group['optimizer_logs_counts'][key])
                optimizer_logs_with_names[group['parameter_name_map'][key]] = value
            if 'epoch{:03d}'.format(epoch) not in group['optimizer_logs_save']:
                group['optimizer_logs_save']['epoch{:03d}'.format(epoch)] = optimizer_logs_with_names
            else:
                for name in group['optimizer_logs_save']['epoch{:03d}'.format(epoch)].keys():
                    group['optimizer_logs_save']['epoch{:03d}'.format(epoch)][name].append(
                        optimizer_logs_with_names[name])

            group['optimizer_logs'] = {}
            group['optimizer_logs_counts'] = {}

    def log_optimizer_data(self, save_path):
        for i, group in enumerate(self.param_groups):
            save_path = Path(save_path)/'optimizer_logs_group_{}.pickle'.format(i)

            if not os.path.exists(save_path.parent):
                os.makedirs(save_path.parent)

            if os.path.exists(save_path):
                with open(save_path, 'rb') as handle:
                    optimizer_logs = pickle.load(handle)
            else:
                optimizer_logs = {}

            with open(save_path, 'wb+') as handle:
                optimizer_logs.update(group['optimizer_logs_save'])
                pickle.dump(optimizer_logs, handle)

            group['optimizer_logs_save'] = {}

    def log_task_weights(self, p, task_weights, group):
        if group['param_to_log_map'].get(p):
            if p in group['optimizer_logs']:
                if len(task_weights) > 0 and (isinstance(task_weights[0], float) or len(task_weights[0].size()) == 0):
                    group['optimizer_logs'][p].append(scalar_tensor_list_to_item_list(task_weights))
                else:
                    # sum up to calculate the mean
                    task_weights_list = scalar_tensor_list_to_item_list(task_weights)
                    for i in range(len(group['optimizer_logs'][p][0])):
                        group['optimizer_logs'][p][0][i] += task_weights_list[i]
                    group['optimizer_logs_counts'][p] += 1
            else:
                group['optimizer_logs'][p] = [scalar_tensor_list_to_item_list(task_weights)]
                group['optimizer_logs_counts'][p] = 1

    def multiplicative_update_step(self, group, h, multiplicative_rule_normalization):
        invalid_normalization_indices = torch.isclose(multiplicative_rule_normalization,
                                                      torch.zeros_like(
                                                          multiplicative_rule_normalization))
        multiplicative_update = group['hypergrad_lr'] * h.div_(multiplicative_rule_normalization)
        # change NaN to 0 to not change the mixing weight if 0 in normalization
        # number_nans = (multiplicative_update != multiplicative_update).sum()
        # if number_nans > 0:
        #     print('{} filters have NaN in multiplicative rule normalization.'.format(number_nans))
        # multiplicative_update[multiplicative_update != multiplicative_update] = 0.
        multiplicative_update[invalid_normalization_indices] = 0.
        # update can not be bigger than 1 (scalar product / norms = -1 <= cosine <= 1)
        multiplicative_update.clamp_(min=-1., max=1.)
        return multiplicative_update


class MultitaskAdam(MultitaskOptimizer):
    r"""Implements Multitask Adam algorithm.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, hypergrad_lr=None, multiplicative_rule=True, per_filter=True,
                 logging=True, decoupled=False, model=None):
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
        if hypergrad_lr is None:
            per_filter = False

        self.model = model

        if hypergrad_lr:
            logging = logging
            parameter_name_map = None
            optimizer_logs = {}
            optimizer_logs_counts = {}
            optimizer_logs_save = {}
        else:
            assert not decoupled, "Decoupled parameter only for hypergradient-based method."
            assert not decoupled and not multiplicative_rule, 'Multiplicative rule is not working with decoupled probably.'

            logging = False
            parameter_name_map = None
            optimizer_logs = None
            optimizer_logs_counts = None
            optimizer_logs_save = None

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, hypergrad_lr=hypergrad_lr,
                        multiplicative_rule=multiplicative_rule,
                        per_filter=per_filter,
                        logging=logging, parameter_name_map=parameter_name_map,
                        decoupled=decoupled,
                        optimizer_logs=optimizer_logs, optimizer_logs_counts=optimizer_logs_counts,
                        optimizer_logs_save=optimizer_logs_save)
        super().__init__(params, defaults)

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
                multiplicative_rule = group['multiplicative_rule']
                decoupled = group['decoupled']

                state = self.state[p]
                n_tasks = len(task_grads)

                # State initialization

                # doing per filter for moving averages makes no sense -> they are already means per filter
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values per task
                    state['exp_avg'] = [torch.zeros_like(p.data) for _ in range(n_tasks)]
                    # Exponential moving average of squared gradient values per task
                    state['exp_avg_sq'] = [torch.zeros_like(p.data) for _ in range(n_tasks)]
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values per task
                        state['max_exp_avg_sq'] = [torch.zeros_like(p.data) for _ in range(n_tasks)]

                    if hypergrad_lr:
                        initial_value = group['lr'] if not decoupled else 1.

                        # per parameter learning rate
                        if not group['per_filter']:
                            state['lr'] = [initial_value] * n_tasks
                        elif len(p.data.size()) == 1:
                            state['lr'] = [torch.full(p.data.size(), initial_value).to(p.data.device)
                                           for _ in range(n_tasks)]
                        elif len(p.data.size()) == 4:
                            state['lr'] = [torch.full(p.data.size()[0:1], initial_value).to(p.data.device)
                                           for _ in range(n_tasks)]
                        else:
                            raise ValueError("Param has not 4 (conv) or 1 (bias) dimension. Size: ", p.data.size())

                beta1, beta2 = group['betas']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction = math.sqrt(bias_correction2) / bias_correction1

                step_size = group['lr']

                multiplicative_rule_normalization = 1.

                loop_run = False
                for i, task_grad in enumerate(task_grads):
                    loop_run = True
                    exp_avg, exp_avg_sq = state['exp_avg'][i], state['exp_avg_sq'][i]
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq'][i]

                    if group['weight_decay'] != 0:
                        task_grad.add_(group['weight_decay']/n_tasks, p.data)

                    if hypergrad_lr and state['step'] > 1:
                        prev_bias_correction1 = 1 - beta1 ** (state['step'] - 1)
                        prev_bias_correction2 = 1 - beta2 ** (state['step'] - 1)
                        # Hypergradient for Adam:
                        if not group['per_filter']:
                            grad_h = -torch.div(exp_avg, exp_avg_sq.sqrt().add_(group['eps'])).mul_(
                                *math.sqrt(prev_bias_correction2) / prev_bias_correction1)

                            if decoupled:
                                grad_h *= sigmoid(state['lr'][i]) * (1 - sigmoid(state['lr'][i]))

                            h = torch.dot(task_grad.view(-1), grad_h.view(-1))

                            if multiplicative_rule:
                                multiplicative_rule_normalization = task_grad.norm() * grad_h.norm()

                        elif len(task_grad.size()) == 1:
                            grad_h = -torch.div(exp_avg, exp_avg_sq.sqrt().add_(group['eps'])).mul_(
                                math.sqrt(prev_bias_correction2) / prev_bias_correction1)

                            if decoupled:
                                grad_h *= sigmoid(state['lr'][i]) * (1 - sigmoid(state['lr'][i]))

                            h = task_grad * grad_h

                            if multiplicative_rule:
                                multiplicative_rule_normalization = task_grad.norm() * grad_h

                            # to compensate the smaller learning rate because norm per filter is smaller
                            # h *= h.numel()
                        elif len(task_grad.size()) == 4:
                            # this multiplication should be okay -> matrixcalculus gave back an diag matrix
                            # which is the same as element-wise
                            grad_h = -torch.div(exp_avg, exp_avg_sq.sqrt().add_(group['eps'])).mul_(
                                math.sqrt(prev_bias_correction2) / prev_bias_correction1)

                            if decoupled:
                                sigm = sigmoid(state['lr'][i])[..., None, None, None]
                                grad_h *= sigm * (1 - sigm)

                            h = (task_grad * grad_h).sum(dim=(1, 2, 3))

                            # TODO
                            if multiplicative_rule:
                                multiplicative_rule_normalization = task_grad.norm() * grad_h.sum(dim=(1, 2, 3))

                            # to compensate the smaller learning rate because norm per filter is smaller
                            # h *= h.numel()
                        else:
                            raise ValueError("Param has not 4 (conv) or 1 (bias) dimension. Size: ", task_grad.size())
                        # Hypergradient descent of the learning rate:
                        if not multiplicative_rule:
                            state['lr'][i] -= group['hypergrad_lr'] * h
                        else:
                            multiplicative_update = self.multiplicative_update_step(group, h,
                                                                                    multiplicative_rule_normalization)

                            state['lr'][i] *= (1 - multiplicative_update)

                            # prevent it from getting zero -> if it does then no more change possible
                            if isinstance(state['lr'][i], float):
                                state['lr'][i] = max(state['lr'][i], group['eps'])
                            else:
                                state['lr'][i] = torch.max(state['lr'][i], torch.full_like(state['lr'][i], group['eps']))

                        step_size = state['lr'][i] if not decoupled else group['lr'] * sigmoid(state['lr'][i])

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

                    if not group['per_filter'] or isinstance(step_size, float) or len(step_size.size()) == 0:
                        p.data.addcdiv_(-step_size*bias_correction, exp_avg, denom)
                    elif len(p.size()) == 1:
                        p.data.addcdiv_(-bias_correction, exp_avg * step_size, denom)
                    elif len(p.size()) == 4:
                        p.data.addcdiv_(-bias_correction, exp_avg * step_size.unsqueeze(1).unsqueeze(2).unsqueeze(3),
                                        denom)
                    else:
                        raise ValueError("Param has not 4 (conv) or 1 (bias) dimension. Size: ", p.size())

                if loop_run:
                    self.clear_task_grads(p)

                if group['logging']:
                    self.log_task_weights(p, state['lr'], group)

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
                 weight_decay=0, hypergrad_lr=1e-2, multiplicative_rule=True, per_filter=True, logging=True, model=None):
        super(MultitaskAdamHD, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                              amsgrad=False, hypergrad_lr=hypergrad_lr,
                                              multiplicative_rule=multiplicative_rule, per_filter=per_filter,
                                              logging=logging, model=model)


class MultitaskAdamDecoupledHD(MultitaskAdam):
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
                 weight_decay=0, hypergrad_lr=1e-2, multiplicative_rule=False, per_filter=True, logging=True, model=None):
        super(MultitaskAdamDecoupledHD, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                                       amsgrad=False, hypergrad_lr=hypergrad_lr,
                                                       multiplicative_rule=multiplicative_rule, per_filter=per_filter,
                                                       logging=logging, decoupled=True, model=model)


class MultitaskAdamMixingHD(MultitaskOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, hypergrad_lr=1e-2, multiplicative_rule=True,
                 normalize_and_clamp_mixing_weights=True, per_filter=True, test_grad_eps=None, logging=True,
                 model=None):
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

        self.model = model

        if hypergrad_lr:    # additive in paper: 1e-7, mulitplicative in paper: 1e-2
            logging = logging
            parameter_name_map = None
            optimizer_logs = {}
            optimizer_logs_counts = {}
            optimizer_logs_save = {}
        else:
            logging = False
            parameter_name_map = None
            optimizer_logs = None
            optimizer_logs_counts = None
            optimizer_logs_save = None

        defaults = dict(lr=lr, betas=betas, eps=eps, hypergrad_lr=hypergrad_lr, multiplicative_rule=multiplicative_rule,
                        normalize_and_clamp_mixing_weights=normalize_and_clamp_mixing_weights, per_filter=per_filter,
                        test_grad_eps=test_grad_eps,
                        logging=logging, parameter_name_map=parameter_name_map,
                        optimizer_logs=optimizer_logs, optimizer_logs_counts=optimizer_logs_counts,
                        optimizer_logs_save=optimizer_logs_save)
        super().__init__(params, defaults)

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

        test_grad_exit = False
        if any([group['test_grad_eps'] for group in self.param_groups]):
            return_h = None

        for group in self.param_groups:
            for p in group['params']:
                task_grads = self.state['task_grads'][p]

                if task_grads[0].is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                hypergrad_lr = group['hypergrad_lr']
                multiplicative_rule = group['multiplicative_rule']

                state = self.state[p]
                n_tasks = len(task_grads)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    if not group['per_filter']:
                        state['mixing_factors'] = [1. for _ in range(n_tasks)]
                    elif len(p.data.size()) == 1:
                        state['mixing_factors'] = [torch.full(p.data.size(), 1.0).to(p.data.device)
                                                   for _ in range(n_tasks)]
                    elif len(p.data.size()) == 4:
                        state['mixing_factors'] = [torch.full(p.data.size()[0:1], 1.0).to(p.data.device)
                                                   for _ in range(n_tasks)]
                    else:
                        raise ValueError("Param has not 4 (conv) or 1 (bias) dimension. Size: ", p.data.size())

                beta1, beta2 = group['betas']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # these are from previous time step
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # this is loss at t w.r.t. weights -> no mixing here
                unmixed_grad = reduce_add(task_grads)

                if hypergrad_lr and state['step'] > 1 and n_tasks > 1:
                    # from previous time step
                    prev_task_grads = self.state['prev_task_grads'][p]
                    prev_bias_correction1 = 1 - beta1 ** (state['step'] - 1)
                    prev_bias_correction2 = 1 - beta2 ** (state['step'] - 1)
                    prev_mixed_grad = reduce_add(prev_task_grads, state['mixing_factors'])

                    prev_exp_avg_corrected = exp_avg.clone()
                    prev_exp_avg_sq_corrected_sqrt = exp_avg_sq.clone().sqrt_() + group['eps']  # epsilon in this row is for preventing zero division

                    # Hypergradient for mixing_factor:

                    # easier to test and read but slower
                    beta_1_ratio = (1 - beta1) / prev_bias_correction1
                    beta_2_ratio = (1 - beta2) / prev_bias_correction2

                    # numpy version (checked)-> this is the gradients, therefore not exactly same bc. below is const. of update step
                    # import numpy as np
                    # alpha = group['lr']
                    # v_t_corr = exp_avg_sq.clone().cpu().numpy()
                    # m_t_corr = exp_avg.clone().cpu().numpy()
                    # prev_mixed_grad_np = prev_mixed_grad.clone().cpu().numpy()
                    # unmixed_grad_np = unmixed_grad.clone().cpu().numpy()
                    # eps = group['eps']

                    # h_mixing_const_np = -alpha * ((beta_1_ratio * (np.sqrt(v_t_corr) + eps)
                    #                                - m_t_corr * beta_2_ratio * 1 / (
                    #                                            np.sqrt(v_t_corr) + eps) * prev_mixed_grad_np)
                    #                               /
                    #                               (np.sqrt(v_t_corr) + eps) ** 2) * unmixed_grad_np

                    # h_mixing_const_2 = -group['lr'] * ((beta_1_ratio * (exp_avg_sq.sqrt() + group['eps'])-exp_avg*beta_2_ratio*(1/(exp_avg_sq.sqrt()+group['eps']))*prev_mixed_grad) / (exp_avg_sq.sqrt() + group['eps'])**2) * unmixed_grad

                    h_mixing_const = prev_exp_avg_corrected.mul_(beta_2_ratio).div_(
                        prev_exp_avg_sq_corrected_sqrt).mul_(prev_mixed_grad).sub_(
                        beta_1_ratio * prev_exp_avg_sq_corrected_sqrt).div_(prev_exp_avg_sq_corrected_sqrt ** 2)

                    # this is multiplied here because every gradient w.r.t. to mixing factor has to be multiplied by g_t
                    h_mixing_const.mul_(group['lr'])
                    if not multiplicative_rule:
                        h_mixing_const.mul_(unmixed_grad)

                    del prev_exp_avg_corrected
                    del prev_exp_avg_sq_corrected_sqrt
                    del prev_mixed_grad
                    self.state['prev_task_grads'][p] = []

                    multiplicative_rule_normalization = 1

                    for i, task_grad in enumerate(task_grads):
                        if test_grad_exit:
                            continue

                        if not group['per_filter']:
                            if multiplicative_rule:
                                grad_h_mixing = h_mixing_const * prev_task_grads[i]
                                multiplicative_rule_normalization = unmixed_grad.norm() * grad_h_mixing.norm()
                                h_mixing = torch.dot(grad_h_mixing.view(-1), unmixed_grad.view(-1)).item()
                            else:
                                # mixing const is multiplied with prev. grad
                                # (switched prev. grad and current grad to have more constant)
                                h_mixing = torch.dot(h_mixing_const.view(-1), prev_task_grads[i].view(-1)).item()

                            if group['test_grad_eps'] and return_h is None:
                                return_h = h_mixing
                                state['mixing_factors'][i] += group['test_grad_eps']
                                test_grad_exit = True
                                continue

                        elif len(prev_task_grads[i].size()) == 1:
                            if multiplicative_rule:
                                grad_h_mixing = h_mixing_const * prev_task_grads[i]
                                multiplicative_rule_normalization = unmixed_grad.norm() * grad_h_mixing
                                h_mixing = grad_h_mixing.mul_(unmixed_grad)
                            else:
                                h_mixing = h_mixing_const * prev_task_grads[i]

                            if group['test_grad_eps'] and return_h is None:
                                return_h = h_mixing.clone()
                                state['mixing_factors'][i] += group['test_grad_eps']
                                test_grad_exit = True
                                continue

                            # to compensate the smaller learning rate because norm per filter is smaller
                            # -> but the changes to the mixing factors sum up at the end
                            # h_mixing *= h_mixing.numel()
                        elif len(prev_task_grads[i].size()) == 4:
                            if multiplicative_rule:
                                grad_h_mixing = h_mixing_const * prev_task_grads[i]
                                multiplicative_rule_normalization = unmixed_grad.norm() * grad_h_mixing.sum(dim=(1, 2, 3))
                                h_mixing = grad_h_mixing.mul_(unmixed_grad).sum(dim=(1, 2, 3))
                            else:
                                # this multiplication should be okay -> matrixcalculus gave back an diag matrix
                                # which is the same as element-wise
                                h_mixing = (h_mixing_const * prev_task_grads[i]).sum(dim=(1, 2, 3))

                            if group['test_grad_eps'] and return_h is None:
                                return_h = h_mixing.clone()
                                # TODO only update this one weight
                                state['mixing_factors'][i] += group['test_grad_eps']
                                test_grad_exit = True
                                continue

                            # to compensate the smaller learning rate because norm per filter is smaller
                            # -> but the changes to the mixing factors sum up at the end
                            # h_mixing *= h_mixing.numel()
                        else:
                            raise ValueError("Param has not 4 (conv) or 1 (bias) dimension. Size: ",
                                             prev_task_grads[i].size())

                        if not multiplicative_rule:
                            state['mixing_factors'][i] -= group['hypergrad_lr'] * h_mixing
                        else:
                            multiplicative_update = self.multiplicative_update_step(group, h_mixing,
                                                                                    multiplicative_rule_normalization)

                            state['mixing_factors'][i] *= (1 - multiplicative_update)

                            # prevent it from getting zero -> if it does then no more change possible
                            if isinstance(state['mixing_factors'][i], float):
                                state['mixing_factors'][i] = max(state['mixing_factors'][i], group['eps'])
                            else:
                                state['mixing_factors'][i] = torch.max(state['mixing_factors'][i],
                                                                       torch.full_like(state['mixing_factors'][i], group['eps']))

                if not test_grad_exit and group['normalize_and_clamp_mixing_weights']:
                    sum_mixing_weights = 0
                    for i in range(len(task_grads)):
                        # tried removing the abs here
                        # sum_mixing_weights += abs(state['mixing_factors'][i])

                        # clamping to ensure mixing factors are >= 0 (projected gradient descent)
                        if isinstance(state['mixing_factors'][i], float):
                            state['mixing_factors'][i] = max(0., state['mixing_factors'][i])
                        else:
                            state['mixing_factors'][i] = state['mixing_factors'][i].clamp_(min=0.)

                        sum_mixing_weights += state['mixing_factors'][i]
                    if (isinstance(sum_mixing_weights, float) and np.allclose(sum_mixing_weights, 0.)) or (
                            not isinstance(sum_mixing_weights, float) and torch.allclose(
                            sum_mixing_weights, torch.zeros_like(sum_mixing_weights))):
                        eps = 1e-9
                        for i in range(len(task_grads)):
                            state['mixing_factors'][i] += eps / len(task_grads)
                        sum_mixing_weights = eps
                    for i in range(len(task_grads)):
                        state['mixing_factors'][i] *= len(task_grads) / sum_mixing_weights

                if not group['per_filter'] or len(task_grads[0].size()) == 1:
                    mixed_grad = state['mixing_factors'][0] * task_grads[0]
                elif len(task_grads[0].size()) == 4:
                    mixed_grad = state['mixing_factors'][0].unsqueeze(1).unsqueeze(2).unsqueeze(3) * task_grads[0]
                else:
                    raise ValueError("Param has not 4 (conv) or 1 (bias) dimension. Size: ", task_grads[0].size())

                for i, task_grad in enumerate(task_grads):
                    if i > 0:
                        if not group['per_filter'] or len(task_grads[i].size()) == 1:
                            mixed_grad += state['mixing_factors'][i] * task_grads[i]
                        elif len(task_grads[i].size()) == 4:
                            mixed_grad += state['mixing_factors'][i].unsqueeze(1).unsqueeze(2).unsqueeze(3) * task_grads[i]
                        else:
                            raise ValueError("Param has not 4 (conv) or 1 (bias) dimension. Size: ",
                                             task_grads[i].size())

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, mixed_grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, mixed_grad, mixed_grad)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                p.data.addcdiv_(-step_size, exp_avg, denom)

                if hypergrad_lr and n_tasks > 1:
                    self.state['prev_task_grads'][p] = self.state['task_grads'][p]

                self.state['task_grads'][p] = []

                if test_grad_exit:
                    return return_h

                # only the conv layers where the gradients are logged
                # log moving average twice per epoch
                if not group['test_grad_eps'] and group['logging']:
                    self.log_task_weights(p, state['mixing_factors'], group)

        return loss


class MultitaskAdamLinearCombinationHD(MultitaskAdamMixingHD):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, hypergrad_lr=1e-2, multiplicative_rule=True,
                 per_filter=True, test_grad_eps=None, logging=True,
                 model=None):
        super().__init__(params, normalize_and_clamp_mixing_weights=False, lr=lr, betas=betas, eps=eps,
                         hypergrad_lr=hypergrad_lr, multiplicative_rule=multiplicative_rule,
                         per_filter=per_filter, test_grad_eps=test_grad_eps, logging=logging, model=model)


def scalar_tensor_list_to_item_list(tensor_list):
    if len(tensor_list) > 0 and (isinstance(tensor_list[0], float) or len(tensor_list[0].size()) == 0):
        return [tensor.item() if type(tensor) is not float else tensor for tensor in tensor_list]
    else:
        return [tensor.cpu().numpy() for tensor in tensor_list]
