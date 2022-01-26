import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required


class FedNova(Optimizer):
    r"""Implements federated normalized averaging (FedNova).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        ratio (float): relative sample size of client
        gmf (float): global/server/slow momentum factor
        mu (float): parameter for proximal local SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=0.05, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=False, variance=0, mu=0):
        self.momentum = momentum
        self.mu = mu
        self.ai_l1_norm = 0
        self.local_counter = 0
        self.local_steps = 0


        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedNova, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FedNova, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        loss = None
        if closure is not None:
            loss = closure()

        # scale = 1**self.itr

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']          

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                
                param_state = self.state[p]
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()

                local_lr = group['lr']

                # apply momentum updates
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # apply proximal updates
                if self.mu != 0:
                    d_p.add_(p.data - param_state['old_init'], alpha=self.mu)

                # update accumulated local updates
                if 'cum_grad' not in param_state:
                    param_state['cum_grad'] = torch.clone(d_p).detach()
                    param_state['cum_grad'].mul_(local_lr)
                else:
                    param_state['cum_grad'].add_(d_p, alpha=local_lr)

                p.data.add_(d_p, alpha=-local_lr)

        # compute local normalizing vector a_i ... but it's a scalar?
        # should't a_i be applied to cum_grad?
        # so this must be the l1 norm? -> this seems correct. a_i is not computed directly, only it's l1 norm
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.ai_l1_norm += self.local_counter
        
        self.etamu = local_lr * self.mu
        if self.etamu != 0:
            self.ai_l1_norm *= (1 - self.etamu)
            self.ai_l1_norm += 1

        if self.momentum == 0 and self.etamu == 0:
            self.ai_l1_norm += 1
        
        self.local_steps += 1

        return loss

    def set_tau_eff(self, tau_eff):
        self.tau_eff = tau_eff

    def pre_communicate(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]

                # apply fednova update rule
                # learning rate has already been applied
                cum_grad = param_state['cum_grad']
                p.data.sub_(cum_grad)   # get back to old_init
                p.data.add_(cum_grad, alpha=self.tau_eff/self.ai_l1_norm)   # rescale changes
                
                # delete stuff for next round
                del param_state['old_init']
                param_state['cum_grad'].zero_()
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()
        
        self.local_counter = 0
        self.ai_l1_norm = 0
        self.local_steps = 0
