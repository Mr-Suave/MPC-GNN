import torch
import tqdm
import os
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_curve
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, save_model, load_model
from models import GINe, PNA, GATe, RGCN, MPCLayerNorm
from torch_geometric.nn import summary
from torch_geometric.utils import degree
import wandb
import logging


class MPCTelemetry:
    """
    Telemetry tracker for a Secure GNN pipeline using:
    - Polynomial Activation (a.x^2 + bx + c): 1 round per layer
    - Newton-Raphson LayerNorm: ~14 rounds per layer
    """
    def __init__(self, num_gnn_layers, n_hidden, num_features, edge_dim, 
                 edge_updates=False, nr_iters=4):
        self.num_gnn_layers = num_gnn_layers
        self.n_hidden = n_hidden
        self.num_features = num_features
        self.edge_dim = edge_dim
        self.edge_updates = edge_updates
        
        # PolyAct requires 1 secure multiplication
        self.poly_rounds = 1  
        
        # LayerNorm: Variance (1) + NR (3 * iters) + Affine Weight (1)
        self.nr_iters = nr_iters
        self.layernorm_rounds = 1 + (3 * self.nr_iters) + 1 
        
        # Cumulative trackers
        self.total_comm_rounds = 0
        self.total_mult_gates = 0

    def record_forward_pass(self, num_nodes, num_edges):
        rounds = 0
        mult_gates = 0
        
        # 1. Initial Embeddings (node_emb, edge_emb) -> 1 MatMul round
        rounds += 1
        mult_gates += (num_nodes * self.num_features * self.n_hidden)
        mult_gates += (num_edges * self.edge_dim * self.n_hidden)
        
        # 2. GNN Layers
        for _ in range(self.num_gnn_layers):
            # GINEConv MLP: Linear (1) -> PolyAct (1) -> Linear (1) -> LayerNorm (14)
            layer_rounds = 1 + self.poly_rounds + 1 + self.layernorm_rounds
            rounds += layer_rounds
            
            # Multiplication gates for 2 linear layers + 1 Poly gate + LN gates
            mult_gates += (num_nodes * self.n_hidden * self.n_hidden) * 2
            mult_gates += (num_nodes * self.n_hidden) # PolyAct gate
            
            # LayerNorm gates (Variance + NR + Affine)
            mult_gates += (num_nodes * self.n_hidden) # Variance square
            mult_gates += (num_nodes * self.n_hidden * self.nr_iters * 3) # NR
            mult_gates += (num_nodes * self.n_hidden) # Affine weight
            
            # Edge updates MLP (if enabled)
            if self.edge_updates:
                rounds += layer_rounds
                mult_gates += (num_edges * (3 * self.n_hidden) * self.n_hidden)
                mult_gates += (num_edges * self.n_hidden)  # PolyAct gate
                mult_gates += (num_edges * self.n_hidden * self.n_hidden)
                # Assuming LayerNorm on edges too
                mult_gates += (num_edges * self.n_hidden * (2 + 3 * self.nr_iters))

        # 3. Head MLP (Linear -> PolyAct -> Linear)
        rounds += (1 + self.poly_rounds + 1)
        mult_gates += (num_edges * (3 * self.n_hidden) * 32)
        mult_gates += (num_edges * 32)  # PolyAct gate
        mult_gates += (num_edges * 32 * 2)  # n_classes = 2

        self.total_comm_rounds += rounds
        self.total_mult_gates += mult_gates
        return rounds

    def record_loss_computation(self):
        """Cross-Entropy / Softmax requires secure exponentiation and division."""
        rounds = 12 
        self.total_comm_rounds += rounds
        return rounds

    def record_backward_pass(self, forward_rounds):
        """Backward pass mirrors the forward pass communication steps."""
        self.total_comm_rounds += forward_rounds

    def record_adam_step(self, num_parameters):
        """Adam optimizer step: 1st/2nd moments + square root division (~17 rounds)."""
        rounds = 17 
        gates = num_parameters * 17 
        
        self.total_comm_rounds += rounds
        self.total_mult_gates += gates

    def log_telemetry(self, epoch, wandb):
        # 64-bit shares (8 bytes) * 2 parties for Beaver Triples
        est_bandwidth_mb = (self.total_mult_gates * 8 * 2) / (1024 * 1024)
        
        wandb.log({
            "mpc/comm_rounds": self.total_comm_rounds,
            "mpc/mult_gates_millions": self.total_mult_gates / 1e6,
            "mpc/est_bandwidth_mb": est_bandwidth_mb
        }, step=epoch)
       
@torch.no_grad() 
def collect_scores(loader, inds, model, device):
    """Scores (logit_1 - logit_0) and labels for the seed edges of a split. Diagnostic only."""
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
    """Threshold maximizing F1 on the given (validation) scores. Diagnostic only."""
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-12)
    i = f[:-1].argmax()  # precision_recall_curve returns one more point than thresholds
    return float(t[i]), float(f[i])


def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    best_val_f1 = -1.0
    
    # --- MPC TELEMETRY INIT ---
    # Count only trainable parameters for the optimizer step
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    mpc_telemetry = MPCTelemetry(
        num_gnn_layers=config.n_gnn_layers,
        n_hidden=round(config.n_hidden),
        num_features=tr_loader.data.x.shape[1],
        edge_dim=tr_loader.data.edge_attr.shape[1] - 1, # -1 because of edge_attr[:, 1:]
        edge_updates=args.emlps
    )
    # --------------------------

    for epoch in range(config.epochs):
        model.train()
        total_loss = total_examples = 0
        preds = []
        ground_truths = []

        
        
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):
            optimizer.zero_grad()
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            batch.edge_attr = batch.edge_attr[:, 1:]
            batch.to(device)
            
            # --- MPC TELEMETRY: FORWARD PASS ---
            num_nodes = batch.x.size(0)
            num_edges = batch.edge_index.size(1)
            fwd_rounds = mpc_telemetry.record_forward_pass(num_nodes, num_edges)
            # -----------------------------------

            out = model(batch.x, batch.edge_index, batch.edge_attr)
            pred = out[mask]
            ground_truth = batch.y[mask]
            
            # --- MPC TELEMETRY: LOSS ---
            mpc_telemetry.record_loss_computation()
            # ---------------------------
            
            loss = loss_fn(pred, ground_truth)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch loss={loss.item()}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            # --- MPC TELEMETRY: BACKWARD PASS & OPTIMIZER ---
            # Using the stored num_gnn_layers from telemetry for backward approximation
            mpc_telemetry.record_backward_pass(forward_rounds=fwd_rounds) 
            mpc_telemetry.record_adam_step(num_trainable_params)
            # ------------------------------------------------
            
            optimizer.step()

            preds.append(pred.argmax(dim=-1).detach().cpu())
            ground_truths.append(ground_truth.detach().cpu())
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()

        train_pred = torch.cat(preds).numpy()
        train_gt = torch.cat(ground_truths).numpy()
        train_f1 = f1_score(train_gt, train_pred)
        train_loss = total_loss / max(total_examples, 1)
        
        # --- MPC TELEMETRY: LOGGING ---
        mpc_telemetry.log_telemetry(epoch, wandb)
        # ------------------------------

        wandb.log({"f1/train": train_f1, "train/loss": train_loss}, step=epoch)
        logging.info(f'Epoch {epoch} | loss {train_loss:.6f} | Train F1: {train_f1:.4f}')

        # IBM protocol: argmax F1, selection by validation F1
        val_f1 = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_homo(te_loader, te_inds, model, te_data, device, args)
        wandb.log({"f1/validation": val_f1, "f1/test": te_f1}, step=epoch)
        logging.info(f'Validation F1 (argmax): {val_f1:.4f}')
        logging.info(f'Test F1 (argmax): {te_f1:.4f}')

        # Diagnostics: threshold-free metrics and threshold-tuned F1
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

    # Config values are the IBM-tuned ones; env vars are optional overrides for sweeps.
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
            "optimizer": "adam",
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
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    sample_batch.to(device)
    sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    logging.info(summary(model, sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr))

    class_weights = torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device)
    logging.info(f"class_weights = {class_weights.tolist()}")
    logging.info(f"loss=ce optimizer=adam activation=poly lr={lr}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)

    wandb.finish()