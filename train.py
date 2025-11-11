import torch
import numpy as np
import time

from utils import create_run, update_run, save_run, seed_everything, log_series
from prep_data import create_loaders
from gen_sgd import SGDGen
import wandb


RUNS = 3
#torch.set_default_dtype(torch.float64)

# def train_workers(suffix, model, optimizer, criterion, epochs, train_loader_workers,
#                   val_loader, test_loader, n_workers, hpo=False, scheduler=None):

def train_workers(suffix, model, optimizer, criterion, epochs, train_loader_workers,
                   val_loader, test_loader, n_workers, device=None, hpo=False, lr_scheduler=None, eta_schedule=None, q_exp=None):
    
    ##################################################
    #new 
    run = create_run()

    best_val_loss = np.inf
    best_val_acc  = 0.0
    test_loss = np.inf
    test_acc  = 0.0

    cum_bp_eq = 0.0
    cum_ex_bp = 0.0

    start_time = torch.cuda.Event(enable_timing=True) if (device and "cuda" in str(device)) else None
    end_time   = torch.cuda.Event(enable_timing=True) if (device and "cuda" in str(device)) else None

    # =========================
    # BASELINE SNAPSHOT (epoch 0)
    # =========================
    # CHANGED: compute both val & test at initial weights first
    val_loss, val_acc   = accuracy_and_loss(model, val_loader,  criterion, device)
    test_loss, test_acc = accuracy_and_loss(model, test_loader, criterion, device)
    best_val_loss = val_loss
    best_val_acc  = val_acc

    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        n_batches  = 0
        for w in range(n_workers):
            for data, labels in train_loader_workers[w]:
                data, labels = data.to(device), labels.to(device)
                out   = model(data)
                total_loss += criterion(out, labels).item()
                n_batches  += 1
    train_loss0 = total_loss / max(1, n_batches)

    update_run(train_loss0, test_loss, test_acc, run)
    log_series(
        run,
        epoch=0,
        lr=optimizer.param_groups[0]["lr"],
        eta=optimizer.param_groups[0]["eta"],
        val_loss=val_loss,
        val_acc=val_acc,
        cum_bp_eq_total=0.0,
        cum_ex_bp_total=0.0,
        wall_seconds=0.0,
        gpu_seconds=None,
    )

    wandb.log({
        "epoch": 0,
        "lr": optimizer.param_groups[0]["lr"],
        "eta": optimizer.param_groups[0]["eta"],
        "train_loss": train_loss0,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "cum_bp_eq": 0.0,
        "cum_ex_bp": 0.0,
        "wall_seconds": 0.0,
    })
    
    #model.train()                  # (put this back right after the snapshot)
    wall_start = time.time()       # <<< reset so epoch 1 wall_seconds starts at ~0


    

    
    # =========================
    #        EPOCH LOOP
    # =========================
    for e in range(epochs):
        model.train()
        
        # start per-epoch GPU timing (if CUDA)
        if start_time is not None:
            start_time.record()
        
        running_loss = 0
        train_loader_iter = [iter(train_loader_workers[w]) for w in range(n_workers)]
        iter_steps = len(train_loader_workers[0])
        for _ in range(iter_steps):
            for w_id in range(n_workers):
                
                data, labels = next(train_loader_iter[w_id])
                data, labels = data.to(device), labels.to(device)

                optimizer.zero_grad(set_to_none=True)
                bp_eq = 1                                        # default 1 backprop
                ef = optimizer.error_feedback
                eta_now = optimizer.param_groups[0]['eta']

                loss_item = None
                
                # -------------------- EF21-IGT (single backprop at extrapolated x) --------------------
                if ef == "EF21_IGT_NORM":
                    bp_eq = 1
                    # CHANGED: no pre-branch forward; compute only at x_extr
                    disp = []
                    for p in model.parameters():
                        s  = optimizer.state[p]
                        kp = f"x_prev_{w_id}"
                        if kp not in s:
                            s[kp] = p.data.detach().clone()
                        disp.append((p.data - s[kp]).detach())

                    scale = (1 - eta_now) / max(eta_now, 1e-12)
                    with torch.no_grad():
                        saved_params = [p.data.detach().clone() for p in model.parameters()]
                        for p, d in zip(model.parameters(), disp):
                            p.data = (p.data + scale * d).detach()

                    out_ex = model(data); loss_ex = criterion(out_ex, labels)
                    loss_ex.backward()
                    loss_item = loss_ex.item()                     # CHANGED

                    with torch.no_grad():
                        for p, sdata in zip(model.parameters(), saved_params):
                            p.data = sdata

                    optimizer.step_local_global(w_id)
                
                # -------------------- EF21-RHM (3 bp-equiv: 1 grad at x^k + HVP at x_hat) ------------
                elif ef == "EF21_RHM_NORM":
                    bp_eq = 3
                    out = model(data); loss = criterion(out, labels)   # CHANGED: compute here
                    loss.backward(create_graph=True)
                    loss_item = loss.item()

                    disp = []
                    for p in model.parameters():
                        s  = optimizer.state[p]
                        kp = f"x_prev_{w_id}"
                        if kp not in s:
                            s[kp] = p.data.detach().clone()
                        disp.append((p.data - s[kp]).detach())

                    q_t = torch.rand(1).item()
                    with torch.no_grad():
                        saved_params = [p.data.detach().clone() for p in model.parameters()]
                        for p in model.parameters():
                            s  = optimizer.state[p]
                            kp = f"x_prev_{w_id}"
                            xk, xkm1 = p.data, s[kp]
                            p.data = (q_t * xk + (1 - q_t) * xkm1).detach()

                    out_hat = model(data); loss_hat = criterion(out_hat, labels)
                    grads_hat = torch.autograd.grad(loss_hat, list(model.parameters()), create_graph=True)
                    hvp = torch.autograd.grad(
                        outputs=grads_hat,
                        inputs=list(model.parameters()),
                        grad_outputs=disp,
                        retain_graph=False,
                        only_inputs=True
                    )
                    hvp_dict = {p: h for p, h in zip(model.parameters(), hvp)}

                    with torch.no_grad():
                        for p, sdata in zip(model.parameters(), saved_params):
                            p.data = sdata

                    optimizer.step_local_global(w_id, hvp_dict=hvp_dict)

                 # -------------------- EF21-MVR_2b (two grads on same minibatch: x^{k-1}, x^k) --------
                elif ef == "EF21_MVR_NORM":
                    bp_eq = 2
                    prev_points = []
                    for p in model.parameters():
                        s  = optimizer.state[p]
                        kp = f"x_prev_{w_id}"
                        if kp not in s:
                            s[kp] = p.data.detach().clone()
                        prev_points.append(s[kp])

                    with torch.no_grad():
                        saved_params = [p.data.detach().clone() for p in model.parameters()]
                        for p, xkm1 in zip(model.parameters(), prev_points):
                            p.data = xkm1.detach()

                    out_prev = model(data); loss_prev = criterion(out_prev, labels)
                    grads_prev = torch.autograd.grad(loss_prev, list(model.parameters()), create_graph=False)
                    gprev_dict = {p: g for p, g in zip(model.parameters(), grads_prev)}

                    with torch.no_grad():
                        for p, sdata in zip(model.parameters(), saved_params):
                            p.data = sdata

                    optimizer.zero_grad(set_to_none=True)
                    out = model(data); loss = criterion(out, labels)
                    loss.backward()
                    loss_item = loss.item()

                    optimizer.step_local_global(w_id, mvr2b_gprev=gprev_dict)

                elif ef == "EF21_HM_NORM":
                    bp_eq = 2
                    out = model(data); loss = criterion(out, labels)   # CHANGED: compute here
                    loss.backward(create_graph=True)
                    loss_item = loss.item()

                    disp = []
                    for p in model.parameters():
                        s = optimizer.state[p]
                        key = f"x_prev_{w_id}"
                        if key not in s:
                            s[key] = p.data.detach().clone()
                        disp.append((p.data - s[key]).detach())

                    hvp = torch.autograd.grad(
                        outputs=[p.grad for p in model.parameters()],
                        inputs=list(model.parameters()),
                        grad_outputs=disp,
                        retain_graph=False,
                        only_inputs=True
                    )
                    hvp_dict = {p: h for p, h in zip(model.parameters(), hvp)}
                    optimizer.step_local_global(w_id, hvp_dict=hvp_dict)
                else:
                    out = model(data); loss = criterion(out, labels)   # CHANGED: compute here
                    loss.backward()
                    loss_item = loss.item()
                    optimizer.step_local_global(w_id)
                
                # ---- accounting ----
                bsz = data.size(0)
                cum_bp_eq += bp_eq / n_workers
                cum_ex_bp += bp_eq * bsz / n_workers
                running_loss += float(loss_item)                       # CHANGED: accumulate the right loss
   
                ##################################################
                
                #optimizer.zero_grad()
                
        # ---------------------------------------------------------------------------
        if lr_scheduler is not None:
            lr_scheduler.step()

        # ---------- ETA SCHEDULER ----------
        if eta_schedule == "poly" and q_exp is not None:
            eta_init = optimizer.param_groups[0]['eta_init']
            cur_eta  = eta_init * ((2/(e + 2)) ** q_exp)
            for g in optimizer.param_groups:
                g['eta'] = cur_eta
        else:
            cur_eta = optimizer.param_groups[0]['eta']
        # -----------------------------------

        train_loss = running_loss/(iter_steps*n_workers)

        val_loss, val_acc = accuracy_and_loss(model, val_loader, criterion, device)

        test_loss, test_acc = accuracy_and_loss(model, test_loader, criterion, device)
        best_val_acc = val_acc
        best_val_loss = val_loss

        # end per-epoch GPU timing
        gpu_ms = None
        if end_time is not None:
            end_time.record()
            torch.cuda.synchronize()
            gpu_ms = start_time.elapsed_time(end_time)  # milliseconds since record()

        wall_elapsed = time.time() - wall_start         # cumulative wall seconds since training start


        update_run(train_loss, test_loss, test_acc, run)
        cur_lr = optimizer.param_groups[0]["lr"]
                 
        print('Epoch: {}/{} | LR: {:g} | ETA: {} | Train Loss: {:.5f} | Test Loss: {:.5f} | Test Acc: {:.2f} | BP-Eq: {} | Ex-BP: {}'
      .format(e + 1, epochs, cur_lr, cur_eta, train_loss, test_loss, test_acc, cum_bp_eq, cum_ex_bp))
        
        log_series(
            run,
            epoch=e + 1,
            lr=cur_lr,
            eta=cur_eta,
            val_loss=val_loss,
            val_acc=val_acc,
            cum_bp_eq_total=cum_bp_eq,
            cum_ex_bp_total=cum_ex_bp,
            wall_seconds=wall_elapsed,
            gpu_seconds=(gpu_ms / 1000.0) if gpu_ms is not None else None,
        )
        
        
        log_dict = {
            "epoch": e + 1,
            "lr": cur_lr,
            "eta": cur_eta,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "test_loss": test_loss,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "cum_bp_eq": cum_bp_eq,            # ← primary x-axis
            "cum_ex_bp": cum_ex_bp,            # ← batch-aware variant
            "wall_seconds": wall_elapsed}
        if gpu_ms is not None:
            log_dict["gpu_seconds"] = gpu_ms / 1000.0
        wandb.log(log_dict)
        
        
    print('')
    if not hpo:
        save_run(suffix, run)

    return best_val_loss, best_val_acc


def accuracy_and_loss(model, loader, criterion, device):
    correct = 0
    total_loss = 0

    model.eval()
    for data, labels in loader:
        #data, labels = data.to(device, dtype=torch.float64), labels.to(device)
        data, labels = data.to(device), labels.to(device)
        
        output = model(data)
        loss = criterion(output, labels)
        total_loss += loss.item()

        #preds = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        _, preds = torch.max(output.data, 1)
        correct += (preds == labels).sum().item()

    accuracy = 100. * correct / len(loader.dataset)
    total_loss = total_loss / len(loader)

    return total_loss, accuracy


def tune_step_size(exp, suffix=None, schedule=None, device=None):
    best_val_loss = np.inf
    best_lr = 0
    best_val_acc = 0
    best_acc_lr = 0
    
    seed = exp['seed']
    seed_everything(seed)
    
    #torch.use_deterministic_algorithms(True)   # raises if a non-det op is hit
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

    hpo = False
    
    exp['val_losses'] = []
    exp['val_accs'] = []
    for idx, lr in enumerate(exp['lrs']):
        for id, eta in enumerate(exp['etas']):
            #print('Learning rate {:2.4f}:'.format(lr), 'Eta {:2.4f}:'.format(eta))
            
            val_loss, val_acc = run_workers(lr, eta, exp, suffix=suffix, hpo=hpo, schedule=schedule, device=device)
            
            exp['val_losses'].append(val_loss)
            exp['val_accs'].append(val_acc)
            if val_loss < best_val_loss:
                best_lr = lr
                best_val_loss = val_loss
                
            if val_acc > best_val_acc:
                best_acc_lr = lr
                best_val_acc = val_acc
            
    return best_lr, best_acc_lr

def run_workers(lr, eta, exp, suffix=None, hpo=False, schedule=None, device=None):
    dataset_name = exp['dataset_name']
    n_workers = exp['n_workers']
    batch_size = exp['batch_size']
    epochs = exp['epochs']
    criterion = exp['criterion']
    error_feedback = exp['error_feedback']
    momentum = exp['momentum']
    weight_decay = exp['weight_decay']
    compression = get_compression(**exp['compression'])
    master_compression = exp['master_compression']

    seed_everything(exp['seed'])    

    net = exp['net']
    model = net()
    if device is None:                                  # caller did not specify
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:                                               # caller passed e.g. "cuda:3"
       device = torch.device(device)
    print(f"Using device: {device}")
    
    #model.to(device).double()
    
    model.to(device) 
    train_loader_workers, val_loader, test_loader = create_loaders(dataset_name, n_workers, batch_size, seed=exp['seed'])

    optimizer = SGDGen(model.parameters(), lr=lr, eta=eta, n_workers=n_workers, error_feedback=error_feedback,
                       comp=compression, momentum=momentum, weight_decay=weight_decay, master_comp=master_compression)
    
    for g in optimizer.param_groups:
        g['eta_init'] = g['eta']
    
    if schedule == "poly":
        p = exp["p"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1.0 / ((epoch + 1) ** p)))
    elif schedule is not None:   # keep the old StepLR option
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = None

    val_loss, val_acc = train_workers(suffix, model, optimizer, criterion, epochs, train_loader_workers,
                                 val_loader, test_loader, n_workers, device=device, hpo=hpo, lr_scheduler=scheduler, eta_schedule=exp["eta_schedule"], q_exp=exp["q"])
                             
    return val_loss, val_acc


def run_tuned_exp(exp, runs=RUNS, suffix=None):
    if suffix is None:
        suffix = exp['name']

    lr = exp['lr']

    if lr is None:
        raise ValueError("Tune step size first")

    seed = exp['seed']
    seed_everything(seed)

    for i in range(runs):
        print('Run {:3d}/{:3d}, Name {}:'.format(i+1, runs, suffix))
        suffix_run = suffix + '_' + str(i+1)
        run_workers(lr, exp, suffix_run)


def get_single_compression(wrapper, compression, **kwargs):
    if wrapper:
        return compression(**kwargs)
    else:
        return compression


def get_compression(combine=None, **kwargs):
    if combine is None:
        return get_single_compression(**kwargs)
    else:
        compression_1 = get_single_compression(**combine['comp_1'])
        compression_2 = get_single_compression(**combine['comp_2'])
        return combine['func'](compression_1, compression_2)
