import torch
import tqdm
import os
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_curve
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, save_model, load_model
from models import GINe, PNA, GATe, RGCN, ScheduledZeroRoundNorm
from torch_geometric.nn import summary
from torch_geometric.utils import degree
import wandb
import logging
from torch.optim import Optimizer


class NewtonRaphsonAdam(Optimizer):
    """
    Custom Adam optimizer using Newton-Raphson for the inverse square root 
    of the second moment. Simulates the exact mathematical consequences 
    of the MPC protocol on the model's convergence and F1 score.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, nr_iters=4):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0 or 1: {betas}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, nr_iters=nr_iters)
        super(NewtonRaphsonAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias corrections
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                v_hat = (exp_avg_sq / bias_correction2) + group['eps']

                # Inside NewtonRaphsonAdam.step(), right after v_hat is computed:
                if not hasattr(self, "_stats"):
                    self._stats = {"min": float("inf"), "max": 0.0, "log10_samples": []}

                v_flat = v_hat.detach().flatten()
                v_pos = v_flat[v_flat > 0]          # ignore exact zeros
                if v_pos.numel() > 0:
                    self._stats["min"] = min(self._stats["min"], v_pos.min().item())
                    self._stats["max"] = max(self._stats["max"], v_pos.max().item())
                    # subsample so this stays cheap
                    idx = torch.randint(0, v_pos.numel(), (min(1000, v_pos.numel()),))
                    self._stats["log10_samples"].append(torch.log10(v_pos[idx]).cpu())

                

                # Newton-Raphson approximation for 1 / sqrt(v_hat)
                # Static guess x0 = 0.1 to simulate the MPC initialization state
                x = torch.full_like(v_hat, 0.1) 
                for _ in range(group['nr_iters']):
                    x = 0.5 * x * (3.0 - v_hat * (x ** 2))

                m_hat = exp_avg / bias_correction1
                
                # Apply simulated MPC step
                p.add_(m_hat * x, alpha=-group['lr'])

        return loss


class MPCTelemetry:
    """
    Telemetry tracker for a Secure GNN pipeline using:
    - Poly Activation (~1 round per layer)
    - ScheduledZeroRoundNorm (0 rounds per layer)
    - Newton-Raphson Adam
    """
    def __init__(self, num_gnn_layers, n_hidden, num_features, edge_dim, 
                 edge_updates=False, relu_rounds=8):
        self.num_gnn_layers = num_gnn_layers
        self.n_hidden = n_hidden
        self.num_features = num_features
        self.edge_dim = edge_dim
        self.edge_updates = edge_updates
        
        self.relu_rounds = relu_rounds  
        
        self.total_comm_rounds = 0
        self.total_mult_gates = 0

    def record_forward_pass(self, num_nodes, num_edges):
        rounds = 0
        mult_gates = 0
        
        # 1. Initial Embeddings (node_emb, edge_emb)
        rounds += 1
        mult_gates += (num_nodes * self.num_features * self.n_hidden)
        mult_gates += (num_edges * self.edge_dim * self.n_hidden)
        
        # 2. GNN Layers
        for _ in range(self.num_gnn_layers):
            layer_mlp_rounds = 1 + self.relu_rounds + 1
            rounds += layer_mlp_rounds
            
            mult_gates += (num_nodes * self.n_hidden * self.n_hidden) * 2
            mult_gates += (num_nodes * self.n_hidden)  
            
            # ZeroRoundNorm: 0 MPC rounds! (Mean sub + public scalar division)
            rounds += 0 
            
            # Edge updates MLP
            if self.edge_updates:
                rounds += layer_mlp_rounds
                mult_gates += (num_edges * (3 * self.n_hidden) * self.n_hidden)
                mult_gates += (num_edges * self.n_hidden)  
                mult_gates += (num_edges * self.n_hidden * self.n_hidden)

        # 3. Head MLP
        rounds += (1 + self.relu_rounds + 1)
        mult_gates += (num_edges * (3 * self.n_hidden) * 32)
        mult_gates += (num_edges * 32)  
        mult_gates += (num_edges * 32 * 2)  

        self.total_comm_rounds += rounds
        self.total_mult_gates += mult_gates
        return rounds

    def record_loss_computation(self):
        rounds = 12 
        self.total_comm_rounds += rounds
        return rounds

    def record_backward_pass(self, forward_rounds):
        self.total_comm_rounds += forward_rounds

    def record_adam_step(self, num_parameters, nr_iters=4):
        """Adam optimizer step using exact Newton-Raphson approximation rounds."""
        # 1 round (square gradient) + 3 rounds per NR iter
        rounds = 1 + (3 * nr_iters)
        gates = num_parameters * rounds 
        
        self.total_comm_rounds += rounds
        self.total_mult_gates += gates

    def log_telemetry(self, epoch, wandb_run=None):
        est_bandwidth_mb = (self.total_mult_gates * 8 * 2) / (1024 * 1024)
        
        log_dict = {
            "mpc/comm_rounds": self.total_comm_rounds,
            "mpc/mult_gates_millions": self.total_mult_gates / 1e6,
            "mpc/est_bandwidth_mb": est_bandwidth_mb
        }
        if wandb_run:
            wandb_run.log(log_dict, step=epoch)
        else:
            wandb.log(log_dict, step=epoch)

       
@torch.no_grad() 
def collect_scores(loader, inds, model, device):
    model.eval()
    inds = inds.detach().cpu()
    scores, labels = [], []
    for batch in loader:
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        batch.edge_attr = batch.edge_attr[:, 1:]
        batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)

        mask = mask.to(device)
        s = (out[:, 1] - out[:, 0])[mask]
        scores.append(s.cpu())
        labels.append(batch.y[mask].cpu())
    model.train()
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    i = f[:-1].argmax() 
    return float(t[i]), float(f[i])


def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    best_val_f1 = -1.0
    
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    mpc_telemetry = MPCTelemetry(
        num_gnn_layers=config.n_gnn_layers,
        n_hidden=round(config.n_hidden),
        num_features=tr_loader.data.x.shape[1],
        edge_dim=tr_loader.data.edge_attr.shape[1] - 1,
        edge_updates=args.emlps
    )

    for epoch in range(config.epochs):
        model.train()
        total_loss = total_examples = 0
        preds = []
        ground_truths = []

        # Inject epoch into ScheduledZeroRoundNorm for dynamic scaling
        for module in model.modules():
            if isinstance(module, ScheduledZeroRoundNorm):
                module.current_epoch = epoch
        
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            batch.edge_attr = batch.edge_attr[:, 1:]
            batch.to(device)
            
            num_nodes = batch.x.size(0)
            num_edges = batch.edge_index.size(1)
            fwd_rounds = mpc_telemetry.record_forward_pass(num_nodes, num_edges)

            out = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out[mask]
            ground_truth = batch.y[mask]
            
            mpc_telemetry.record_loss_computation()
            
            loss = loss_fn(pred, ground_truth)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch loss={loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # Use dynamic forward rounds and 4-iter NR Adam step
            mpc_telemetry.record_backward_pass(forward_rounds=fwd_rounds) 
            mpc_telemetry.record_adam_step(num_trainable_params, nr_iters=4)
            
            optimizer.step()

            preds.append(pred.argmax(dim=-1).detach().cpu())
            ground_truths.append(ground_truth.detach().cpu())
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        train_pred = torch.cat(preds).numpy()
        train_gt = torch.cat(ground_truths).numpy()
        train_f1 = f1_score(train_gt, train_pred)
        train_loss = total_loss / max(total_examples, 1)
        
        mpc_telemetry.log_telemetry(epoch)

        wandb.log({"f1/train": train_f1, "train/loss": train_loss}, step=epoch)
        logging.info(f'Epoch {epoch} | loss {train_loss:.6f} | Train F1: {train_f1:.4f}')

        val_f1 = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_homo(te_loader, te_inds, model, te_data, device, args)
        wandb.log({"f1/validation": val_f1, "f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1 (argmax): {val_f1:.4f}')
        logging.info(f'Test F1 (argmax): {te_f1:.4f}')

        val_s, val_y = collect_scores(val_loader, val_inds, model, device)
        te_s, te_y = collect_scores(te_loader, te_inds, model, device)
        val_auc = roc_auc_score(val_y, val_s)
        val_ap = average_precision_score(val_y, val_s)
        te_auc = roc_auc_score(te_y, te_s)
        te_ap = average_precision_score(te_y, te_s)
        thr, val_f1_tuned = best_threshold(val_y, val_s)
        te_f1_at_thr = f1_score(te_y, (te_s > thr).astype(int))

        logging.info(f"Validation AUC: {val_auc:.4f} | AP: {val_ap:.4f}")
        logging.info(f"Validation best threshold {thr:.4f} -> F1 {val_f1_tuned:.4f}")
        logging.info(f"Test @ val threshold | F1 {te_f1_at_thr:.4f} | AUC {te_auc:.4f} | AP {te_ap:.4f}")
        wandb.log({
            "val/auc": val_auc, "val/ap": val_ap, "val/f1_tuned": val_f1_tuned,
            "test/auc": te_auc, "test/ap": te_ap, "test/f1_at_val_thr": te_f1_at_thr,
            "val/threshold": thr,
        }, step=epoch)

        if hasattr(optimizer, "_stats") and optimizer._stats["log10_samples"]:
            s = optimizer._stats
            all_log = torch.cat(s["log10_samples"])
            print(f"[v_hat] min={s['min']:.2e} max={s['max']:.2e} "
                f"median={10**all_log.median().item():.2e} "
                f"p5={10**torch.quantile(all_log, 0.05).item():.2e} "
                f"p95={10**torch.quantile(all_log, 0.95).item():.2e}")
            s["log10_samples"] = []   # reset for next epoch; keep min/max running

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)

    return model

def get_model(sample_batch, config, args):
    n_feats = sample_batch.x.shape[1]
    e_dim = sample_batch.edge_attr.shape[1] - 1
    if args.model == "gin":
        return GINe(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, final_dropout=config.final_dropout,
        )
    if args.model == "gat":
        return GATe(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), n_heads=round(config.n_heads),
            edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, final_dropout=config.final_dropout,
        )
    if args.model == "pna":
        d = degree(sample_batch.edge_index[1], dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        return PNA(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, deg=deg, final_dropout=config.final_dropout,
        )
    if args.model == "rgcn":
        return RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8, num_gnn_layers=round(config.n_gnn_layers),
            n_classes=2, n_hidden=round(config.n_hidden),
            edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None,
        )
    raise ValueError(f"unknown model: {args.model}")


def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.reverse_mp:
        raise ValueError("This pipeline supports only homogeneous graphs; run with reverse_mp off.")

    lr = float(os.environ.get("ADAM_LR", extract_param("lr", args)))
    w_ce2 = float(os.environ.get("W_CE2", extract_param("w_ce2", args)))

    wandb.init(
        mode="disabled" if args.testing else "online",
        project="your_proj_name",
        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": lr,
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "optimizer": "newton_raphson_adam",
            "activation": "poly",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": w_ce2,
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "node_features": ["net_flow", "amount_std", "dwell_time"],
            "clip": 10.0,
            "seed": args.seed,
        },
    )
    config = wandb.config

    transform = AddEgoIds() if args.ego else None
    add_arange_ids([tr_data, val_data, te_data])
    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.finetune:
        model, optimizer = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        # Swapped PyTorch Adam for our custom NewtonRaphsonAdam
        optimizer = NewtonRaphsonAdam(model.parameters(), lr=lr, nr_iters=4)

    sample_batch.to(device)
    sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    logging.info(summary(model, sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr))

    class_weights = torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device)
    logging.info(f"class_weights = {class_weights.tolist()}")
    logging.info(f"loss=ce optimizer=newton_raphson_adam activation=poly lr={lr}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)

    wandb.finish()