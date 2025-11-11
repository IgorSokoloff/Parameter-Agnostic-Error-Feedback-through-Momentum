import torch
from torch.optim.optimizer import Optimizer


class SGDGen(Optimizer):
    r"""
        based on torch.optim.SGD implementation
    """

    def __init__(self, params, lr, eta, n_workers, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, comp=None, master_comp=None,
                 error_feedback=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eta is not None:
            if eta < 0.0: 
                raise ValueError("Invalid eta: {}".format(eta))
        
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, eta=eta, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDGen, self).__init__(params, defaults)

        self.comp = comp
        self.error_feedback = error_feedback
        if self.error_feedback and self.comp is None:
            raise ValueError("For Error-Feedback, compression can't be None")

        self.master_comp = master_comp  # should be unbiased, Error-Feedback is not supported at the moment

        self.n_workers = n_workers
        self.grads_received = 0

    def __setstate__(self, state):
        super(SGDGen, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step_local_global(self, w_id, hvp_dict=None, mvr2b_gprev=None):
        """Performs a single optimization step.
        Arguments:
            w_id: integer, id of the worker
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None

        self.grads_received += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]

                d_p = p.grad.data

                if self.error_feedback == "EF":
                    error_key = 'error_' + str(w_id)
                    if error_key not in param_state:
                        #loc_grad = d_p.mul(group['lr'])
                        loc_grad = torch.clone(d_p).detach()
                    else:
                        loc_grad = torch.clone(d_p).detach() + param_state[error_key].mul(group['eta'])
                        #loc_grad = d_p.mul(group['lr']) + param_state[error_key]

                    d_p = self.comp(loc_grad)
                    param_state[error_key] = loc_grad - d_p
                    
                elif self.error_feedback == "ECONTROL":
                    error_key = 'error_' + str(w_id)
                    if error_key not in param_state:
                        param_state[error_key] = torch.zeros_like(d_p)
                        
                    h_name = 'h_' + str(w_id)
                    if h_name not in param_state:
                        param_state[h_name] = d_p
                        
                    Delta = self.comp(param_state[error_key].mul(group['eta']) + d_p - param_state[h_name])
                    
                    param_state[error_key] = d_p + param_state[error_key] - param_state[h_name] - Delta
                    param_state[h_name] = param_state[h_name] + Delta
       
                    d_p = param_state[h_name]
                
                elif self.error_feedback == "EF21":
                    error_key = 'error_' + str(w_id)
                    if error_key not in param_state:
                        #g0 = self.comp(d_p)                   # <- compress once
                        g0 = d_p                   # <- uncompressed init
                        param_state[error_key] = g0          # store compressed g_i^0
                        d_p = g0                              # send compressed
                    else:
                        param_state[error_key] += self.comp(d_p - param_state[error_key])
                        d_p = param_state[error_key]
                
                elif self.error_feedback == "EF21_SGDM":
                    g_key = 'g_' + str(w_id)
                    v_key = 'v_' + str(w_id)
                    if g_key not in param_state or v_key not in param_state:
                        #g0 = self.comp(d_p)
                        g0 = d_p            # <- uncompressed init
                        param_state[g_key] = g0              # g_i^{-1}  = C(∇f_i)
                        param_state[v_key] = g0              # v_i^{-1}  = C(∇f_i)
                        d_p = g0
                    else:
                        param_state[v_key] = d_p.mul(group['eta']) + param_state[v_key].mul(1-group['eta'])
                        param_state[g_key] += self.comp(param_state[v_key] - param_state[g_key])
                    
                        d_p = param_state[g_key]
                
                elif self.error_feedback == "EF21_SGDM_NORM":
                    g_key = 'g_' + str(w_id)
                    v_key = 'v_' + str(w_id)
                    # --- initialise states on first call
                    if g_key not in param_state or v_key not in param_state:
                        #g0 = self.comp(d_p)
                        g0 = d_p                               # <- uncompressed init
                        param_state[g_key] = g0              # g_i^{-1}
                        param_state[v_key] = g0              # v_i^{-1}
                        d_p = g0
                    else:
                        # v_iᵏ  ←  (1-ηₖ)v_i^{k-1} + ηₖ ∇f_i
                        param_state[v_key] = d_p.mul(group['eta'])+ param_state[v_key].mul(1 - group['eta'])
                        # g_iᵏ  ←  g_i^{k-1} + C(v_iᵏ – g_i^{k-1})
                        param_state[g_key] += self.comp(param_state[v_key] - param_state[g_key])
                        d_p = param_state[g_key]   # send g_iᵏ       (will be summed later)
                
                elif self.error_feedback == "EF21_HM_NORM":
                    g_key, v_key = f"g_{w_id}", f"v_{w_id}"
                    prev_x       = f"x_prev_{w_id}"
                    eta, comp    = group["eta"], self.comp
                    # ---------- first visit ----------------------------------------
                    if g_key not in param_state:
                        param_state[g_key]  = d_p.detach().clone()
                        param_state[v_key]  = d_p.detach().clone()
                        param_state[prev_x] = p.data.detach().clone()
                        d_p = param_state[g_key]                      # message
                    # ---------- main iteration -------------------------------------
                    else:
                        if eta == 1:
                            # exact EF21 path – drop the HVP term entirely
                            param_state[v_key] = d_p.detach().clone()          # <-  clone!
                        else:
                            # • g_curr already in d_p
                            # • hvp_term supplied once from the loop
                            hvp_term  = hvp_dict[p]
                            v_prev = param_state[v_key]
                            param_state[v_key] = ((1 - eta) * (v_prev + hvp_term) + eta * d_p).detach()# good practice
                        
                        param_state[g_key] += comp(param_state[v_key] - param_state[g_key])
                        param_state[prev_x] = p.data.detach().clone()
                        d_p = param_state[g_key]                      # message
                
                # NOTE: new methods goes below
                elif self.error_feedback == "EF21_RHM_NORM":
                    g_key, v_key = f"g_{w_id}", f"v_{w_id}"
                    prev_x       = f"x_prev_{w_id}"
                    eta, comp    = group["eta"], self.comp

                    if g_key not in param_state or v_key not in param_state:
                        g0 = d_p.detach().clone()
                        param_state[g_key]  = g0
                        param_state[v_key]  = g0
                        # initialize snapshot
                        param_state[prev_x] = p.data.detach().clone()
                        d_p = g0
                    else:
                        # d_p is ∇f_i(x^k) from the first backward
                        hvp_term = hvp_dict[p]  # ∇²f_i(x_hat)·(x^k − x^{k−1})
                        param_state[v_key] = (1 - eta) * (param_state[v_key] + hvp_term) + eta * d_p
                        param_state[g_key] += comp(param_state[v_key] - param_state[g_key])
                        d_p = param_state[g_key]
                        # roll snapshot to current x^k (before master update)
                        param_state[prev_x] = p.data.detach().clone()

                
                
                elif self.error_feedback == "EF21_IGT_NORM":
                    # States (per worker)
                    g_key, v_key = f"g_{w_id}", f"v_{w_id}"
                    prev_x       = f"x_prev_{w_id}"
                    eta, comp    = group["eta"], self.comp

                    if g_key not in param_state or v_key not in param_state:
                        # First call: initialize with current (uncompressed) gradient
                        g0 = d_p.detach().clone()
                        param_state[g_key] = g0        # g_i^{-1}
                        param_state[v_key] = g0        # v_i^{-1}
                        param_state[prev_x] = p.data.detach().clone()   # <-- initialize snapshot
                        d_p = g0
                    else:
                        # d_p is ∇f_i evaluated at the **extrapolated point** (computed in train.py)
                        # v_i^{t+1} = (1-η_t) v_i^t + η_t ∇f_i(x_extr)
                        param_state[v_key] = (1 - eta) * param_state[v_key] + eta * d_p
                        # g_i^{t+1} = g_i^t + C(v_i^{t+1} - g_i^t)
                        param_state[g_key] += comp(param_state[v_key] - param_state[g_key])
                        d_p = param_state[g_key]
                        
                        # keep the rolling "previous x" snapshot up to date
                        param_state[prev_x] = p.data.detach().clone()   # <-- advance snapshot

                elif self.error_feedback == "EF21_MVR_1b":
                    g_key, v_key = f"g_{w_id}", f"v_{w_id}"
                    prevg_key    = f"prev_grad_{w_id}"
                    eta, comp    = group["eta"], self.comp

                    if g_key not in param_state or v_key not in param_state or prevg_key not in param_state:
                        g0 = d_p.detach().clone()      # current grad ~ ∇f_i(x^{t+1})
                        param_state[g_key]   = g0
                        param_state[v_key]   = g0
                        param_state[prevg_key] = g0    # bootstrap prev grad
                        d_p = g0
                    else:
                        g_new  = d_p.detach().clone()                # ∇f_i(x^{t+1})
                        g_old  = param_state[prevg_key]              # (cached) ≈ ∇f_i(x^t)
                        v_prev = param_state[v_key]
                        # v^{t+1} = (1-η_t)(v^t + g_new - g_old) + η_t g_new
                        v_next = (1 - eta) * (v_prev + (g_new - g_old)) + eta * g_new
                        param_state[v_key] = v_next
                        param_state[g_key] += comp(v_next - param_state[g_key])
                        param_state[prevg_key] = g_new               # cache for next iter
                        d_p = param_state[g_key]

                elif self.error_feedback == "EF21_MVR_NORM":
                    # Two-backprop, same minibatch
                    # d_p carries g_new = ∇f_i(x^k; ξ) from loss.backward()
                    # mvr2b_gprev[p] carries g_old = ∇f_i(x^{k-1}; ξ) computed via autograd.grad
                    g_key, v_key = f"g_{w_id}", f"v_{w_id}"
                    prev_x       = f"x_prev_{w_id}"
                    eta, comp    = group["eta"], self.comp

                    if g_key not in param_state or v_key not in param_state or mvr2b_gprev is None:
                        # Bootstrap (first step): fall back to using current grad only
                        g0 = d_p.detach().clone()
                        param_state[g_key]  = g0
                        param_state[v_key]  = g0
                        param_state[prev_x] = p.data.detach().clone()
                        d_p = g0
                    else:
                        g_new  = d_p.detach().clone()
                        g_old  = mvr2b_gprev[p].detach()
                        v_prev = param_state[v_key]

                        # v^{t+1} = (1-η)(v^t + g_new - g_old) + η g_new
                        v_next = (1 - eta) * (v_prev + (g_new - g_old)) + eta * g_new
                        param_state[v_key] = v_next

                        # EF21 aggregation state update: g ← g + C(v_next − g)
                        param_state[g_key] += comp(v_next - param_state[g_key])
                        d_p = param_state[g_key]

                        # roll snapshot of x to "current" for next step's previous
                        param_state[prev_x] = p.data.detach().clone()

                else:
                    raise ValueError("Unknown error feedback type: {}".format(self.error_feedback))

                if 'full_grad' not in param_state or self.grads_received == 1:
                    param_state['full_grad'] = torch.clone(d_p).detach()
                else:
                    param_state['full_grad'] += torch.clone(d_p).detach()

                if self.grads_received == self.n_workers:
                    grad = param_state['full_grad'] / self.n_workers
                    

                    if self.error_feedback in ("EF21_SGDM_NORM", "EF21_HM_NORM", "EF21_IGT_NORM", "EF21_RHM_NORM", "EF21_MVR_1b", "EF21_MVR_NORM"):
                        norm = grad.norm()
                        if norm > 0:
                            grad = grad / norm
                    # ----------------------------------------------------------

                    if self.master_comp is not None:
                        grad = self.master_comp(grad)

                    if weight_decay != 0:
                        grad.add(p, alpha=weight_decay)
                    if momentum != 0:
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                        if nesterov:
                            grad = grad.add(buf, alpha=momentum)
                        else:
                            grad = buf

                    #with torch.no_grad():
                    p.data.add_(-group['lr'], grad)
                    

        if self.grads_received == self.n_workers:
            self.grads_received = 0

        return loss
