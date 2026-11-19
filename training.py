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


class MPCTelemetry:
    """
    Telemetry tracker for a Secure GNN pipeline using Matrix Beaver Triples:
    - Linear layers: Matrix Beaver Triples -> 1 round, online comm = (N*D + D*H) elements
      (Backward pass reuses opened input/weight masks E and F -> only N*H elements for dY)
    - ReLU activation: Comparison-based DReLU -> `relu_rounds` rounds, 1 element-wise gate/elem
    - ScheduledZeroRoundNorm: 0 rounds, 0 online elements (mean sub + public scalar div)
    - Cross-entropy loss: `loss_rounds` rounds (Softmax/log approximation)
    - SGD with momentum: 0 rounds, 0 online elements (local public-scalar mults/adds)
    """
    def __init__(self, num_gnn_layers, n_hidden, num_features, edge_dim,
                 edge_updates=False, relu_rounds=8, loss_rounds=12):
        self.num_gnn_layers = num_gnn_layers
        self.n_hidden = n_hidden
        self.num_features = num_features
        self.edge_dim = edge_dim
        self.edge_updates = edge_updates
        self.relu_rounds = relu_rounds
        self.loss_rounds = loss_rounds

        self.total_comm_rounds = 0
        self.total_online_elements = 0      # Secret-shared elements opened over the network (per party)
        self.total_local_macs = 0           # Local plaintext/share multiply-accumulate ops (0 comm)
        self.total_local_linear_ops = 0     # Optimizer local public-scalar ops (0 comm)

    @staticmethod
    def _matrix_triple_cost(n_rows, d_in, h_out):
        """
        Returns (fwd_elements, bwd_elements, macs) for Y = X @ W where X is (N, D) and W is (D, H):
        - Forward: Party opens E = X - A (N*D) and F = W - B (D*H) -> N*D + D*H elements.
        - Backward: dX = dY @ W^T and dW = X^T @ dY reuse opened masks F and E -> only dY (N*H) opened.
        - Local Compute: 1 forward + 2 backward matrix multiplications = 3 * N * D * H MACs.
        """
        fwd_elems = (n_rows * d_in) + (d_in * h_out)
        bwd_elems = (n_rows * h_out)
        macs = 3 * n_rows * d_in * h_out
        return fwd_elems, bwd_elems, macs

    def record_forward_and_backward(self, num_nodes, num_edges):
        """
        Records communication rounds, online Matrix Beaver Triple elements, and local MACs
        for both the forward and backward passes of one batch.
        """
        fwd_rounds = 0
        online_elems = 0
        local_macs = 0

        # 1. Initial embeddings (node_emb, edge_emb) -> 1 parallel round
        fwd_rounds += 1
        for rows, d_in, h_out in [
            (num_nodes, self.num_features, self.n_hidden),
            (num_edges, self.edge_dim, self.n_hidden),
        ]:
            f_el, b_el, macs = self._matrix_triple_cost(rows, d_in, h_out)
            online_elems += (f_el + b_el)
            local_macs += macs

        # 2. GNN Layers
        for _ in range(self.num_gnn_layers):
            # GINEConv MLP: Linear -> ReLU -> Linear
            fwd_rounds += (1 + self.relu_rounds + 1)

            # Two Linear(n_hidden, n_hidden) layers
            for _ in range(2):
                f_el, b_el, macs = self._matrix_triple_cost(num_nodes, self.n_hidden, self.n_hidden)
                online_elems += (f_el + b_el)
                local_macs += macs

            # ReLU element-wise masking gates (fwd + bwd = 2 * N * H)
            online_elems += 2 * (num_nodes * self.n_hidden)

            # ScheduledZeroRoundNorm: 0 rounds, 0 online elements
            fwd_rounds += 0

            if self.edge_updates:
                fwd_rounds += (1 + self.relu_rounds + 1)
                for d_in, h_out in [(3 * self.n_hidden, self.n_hidden), (self.n_hidden, self.n_hidden)]:
                    f_el, b_el, macs = self._matrix_triple_cost(num_edges, d_in, h_out)
                    online_elems += (f_el + b_el)
                    local_macs += macs
                online_elems += 2 * (num_edges * self.n_hidden)

        # 3. Head MLP: Linear(3*H, 32) -> ReLU -> Linear(32, 2)
        fwd_rounds += (1 + self.relu_rounds + 1)
        for d_in, h_out in [(3 * self.n_hidden, 32), (32, 2)]:
            f_el, b_el, macs = self._matrix_triple_cost(num_edges, d_in, h_out)
            online_elems += (f_el + b_el)
            local_macs += macs
        online_elems += 2 * (num_edges * 32)

        # Backward pass mirrors forward round depth
        bwd_rounds = fwd_rounds

        self.total_comm_rounds += (fwd_rounds + bwd_rounds)
        self.total_online_elements += online_elems
        self.total_local_macs += local_macs
        return fwd_rounds

    def record_loss_computation(self):
        """Cross-entropy: softmax/log are approximated; `loss_rounds` is an assumption."""
        self.total_comm_rounds += self.loss_rounds
        return self.loss_rounds

    def record_sgd_step(self, num_parameters):
        """SGD with momentum: buf = mu*buf + g ; p -= lr*buf (0 rounds, 0 online elements)."""
        self.total_local_linear_ops += 4 * num_parameters
        return 0

    def log_telemetry(self, epoch, wandb_run=None):
        # 2 parties * 8 bytes (64-bit shares) per opened element
        est_bandwidth_mb = (self.total_online_elements * 8 * 2) / (1024 * 1024)

        log_dict = {
            "mpc/comm_rounds": self.total_comm_rounds,
            "mpc/online_elements_millions": self.total_online_elements / 1e6,
            "mpc/est_bandwidth_mb": est_bandwidth_mb,
            "mpc/local_macs_billions": self.total_local_macs / 1e9,
            "mpc/local_linear_ops_millions": self.total_local_linear_ops / 1e6,
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
            mpc_telemetry.record_forward_and_backward(num_nodes=batch.x.size(0), num_edges=batch.edge_index.size(1))

            out = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out[mask]
            ground_truth = batch.y[mask]

            

            loss = loss_fn(pred, ground_truth)
            mpc_telemetry.record_loss_computation()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch loss={loss.item()}")

            loss.backward()
            # NOTE: clipping needs a secure norm/sqrt under MPC; not counted in telemetry.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            
            mpc_telemetry.record_sgd_step(num_trainable_params)

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

    # SGD with momentum. NOTE: the lr from extract_param was tuned for Adam;
    # override with the ADAM_LR env var (name kept for compatibility) or tune it for SGD.
    lr = float(os.environ.get("ADAM_LR", extract_param("lr", args)))
    momentum = float(os.environ.get("MOMENTUM", 0.9))
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
            "momentum": momentum,
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "optimizer": "sgd_momentum",
            "activation": "relu",
            "norm": "scheduled_zero_round",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": w_ce2,
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "node_features": ["net_flow", "amount_std", "dwell_time"],
            "clip": None,
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
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    sample_batch.to(device)
    sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    logging.info(summary(model, sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr))

    class_weights = torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device)
    logging.info(f"class_weights = {class_weights.tolist()}")
    logging.info(f"loss=ce optimizer=sgd_momentum(mu={momentum}) activation=relu norm=zero_round lr={lr}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)

    wandb.finish()