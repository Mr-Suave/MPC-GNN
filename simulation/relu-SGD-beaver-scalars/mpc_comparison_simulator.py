import numpy as np
import matplotlib.pyplot as plt


class MPCTelemetrySimulator:
    """
    Unified MPC Telemetry Tracker supporting both:
      - mode='optimized': ReLU + ScheduledZeroRoundNorm (0 rounds) + CE + SGD w/ Momentum (0 rounds)
      - mode='baseline':  ReLU + Real BatchNorm (15 rounds/layer) + CE + Plain Adam (17 rounds/step)
    """
    def __init__(self, mode, num_gnn_layers, n_hidden, num_features, edge_dim,
                 num_parameters, edge_updates=False, relu_rounds=8, loss_rounds=12):
        assert mode in ("optimized", "baseline")
        self.mode = mode
        self.num_gnn_layers = num_gnn_layers
        self.n_hidden = n_hidden
        self.num_features = num_features
        self.edge_dim = edge_dim
        self.num_parameters = num_parameters
        self.edge_updates = edge_updates
        self.relu_rounds = relu_rounds
        self.loss_rounds = loss_rounds

        # BatchNorm: 1 (var square) + 13 (4-iter NR inv_sqrt) + 1 (affine gamma) = 15 rounds
        self.bn_rounds = 15 if mode == "baseline" else 0
        # Adam: ~17 rounds; SGD + Momentum: 0 rounds
        self.opt_rounds = 17 if mode == "baseline" else 0

        self.total_comm_rounds = 0
        self.total_mult_gates = 0

    def step_batch(self, num_nodes, num_edges):
        fwd_rounds = 0
        fwd_gates = 0

        # 1. Initial Embeddings (node_emb, edge_emb)
        fwd_rounds += 1
        fwd_gates += (num_nodes * self.num_features * self.n_hidden)
        fwd_gates += (num_edges * self.edge_dim * self.n_hidden)

        # 2. GNN Layers
        for _ in range(self.num_gnn_layers):
            # GINEConv MLP: Linear (1) -> ReLU (8) -> Linear (1)
            layer_mlp_rounds = 1 + self.relu_rounds + 1
            fwd_rounds += layer_mlp_rounds + self.bn_rounds

            fwd_gates += (num_nodes * self.n_hidden * self.n_hidden) * 2
            fwd_gates += (num_nodes * self.n_hidden)  # ReLU gate

            if self.mode == "baseline":
                # BatchNorm secret multiplications: variance (1) + 4-iter NR (12) + affine (1) = 14 gates/elem
                fwd_gates += (num_nodes * self.n_hidden * 14)

            if self.edge_updates:
                fwd_rounds += layer_mlp_rounds
                fwd_gates += (num_edges * (3 * self.n_hidden) * self.n_hidden)
                fwd_gates += (num_edges * self.n_hidden)
                fwd_gates += (num_edges * self.n_hidden * self.n_hidden)

        # 3. Head MLP (Linear -> ReLU -> Linear)
        fwd_rounds += (1 + self.relu_rounds + 1)
        fwd_gates += (num_edges * (3 * self.n_hidden) * 32)
        fwd_gates += (num_edges * 32)
        fwd_gates += (num_edges * 32 * 2)

        # 4. Loss (Cross-Entropy)
        loss_rounds = self.loss_rounds

        # 5. Backward Pass (~1x forward rounds, ~2x forward multiplication gates)
        bwd_rounds = fwd_rounds
        bwd_gates = 2 * fwd_gates

        # 6. Optimizer Step
        opt_rounds = self.opt_rounds
        opt_gates = (self.num_parameters * 17) if self.mode == "baseline" else 0

        self.total_comm_rounds += (fwd_rounds + loss_rounds + bwd_rounds + opt_rounds)
        self.total_mult_gates += (fwd_gates + bwd_gates + opt_gates)

    def get_metrics(self):
        # 64-bit shares (8 bytes) * 2 parties for Beaver triples -> converted to GB
        est_bandwidth_gb = (self.total_mult_gates * 8 * 2) / (1024 ** 3)
        return {
            "rounds": self.total_comm_rounds,
            "gates_millions": self.total_mult_gates / 1e6,
            "bandwidth_gb": est_bandwidth_gb,
        }


def run_simulation(batch_stats, num_epochs=100, num_gnn_layers=2, n_hidden=66,
                   num_features=3, edge_dim=6, num_parameters=33692, edge_updates=False):
    opt_sim = MPCTelemetrySimulator("optimized", num_gnn_layers, n_hidden, num_features, edge_dim, num_parameters, edge_updates)
    base_sim = MPCTelemetrySimulator("baseline", num_gnn_layers, n_hidden, num_features, edge_dim, num_parameters, edge_updates)

    epochs = np.arange(1, num_epochs + 1)
    history = {
        "opt_rounds": [], "base_rounds": [],
        "opt_gates": [],  "base_gates": [],
        "opt_bw": [],     "base_bw": [],
    }

    for _ in epochs:
        for num_nodes, num_edges in batch_stats:
            opt_sim.step_batch(num_nodes, num_edges)
            base_sim.step_batch(num_nodes, num_edges)

        o_m = opt_sim.get_metrics()
        b_m = base_sim.get_metrics()

        history["opt_rounds"].append(o_m["rounds"])
        history["base_rounds"].append(b_m["rounds"])
        history["opt_gates"].append(o_m["gates_millions"])
        history["base_gates"].append(b_m["gates_millions"])
        history["opt_bw"].append(o_m["bandwidth_gb"])
        history["base_bw"].append(b_m["bandwidth_gb"])

    return epochs, history


def plot_comparison(epochs, history, save_path="mpc_100_epoch_comparison.png"):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    c_base = "#D95F02"  # Orange/Rust for Plain Baseline
    c_opt = "#1B9E77"   # Teal/Green for Optimized MPC Pipeline

    # Panel 1: Cumulative Communication Rounds
    axes[0].plot(epochs, np.array(history["base_rounds"]) / 1e6, label="Baseline (Adam + ReLU + BatchNorm)", color=c_base, lw=2.5)
    axes[0].plot(epochs, np.array(history["opt_rounds"]) / 1e6, label="Ours (SGD + ReLU + ZeroRoundNorm)", color=c_opt, lw=2.5)
    axes[0].set_title("Cumulative Communication Rounds", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Communication Rounds (Millions)", fontsize=11)
    axes[0].legend(frameon=True, fontsize=9.5)

    # Panel 2: Cumulative Secret Multiplication Gates
    axes[1].plot(epochs, np.array(history["base_gates"]) / 1e3, label="Baseline (Adam + ReLU + BatchNorm)", color=c_base, lw=2.5)
    axes[1].plot(epochs, np.array(history["opt_gates"]) / 1e3, label="Ours (SGD + ReLU + ZeroRoundNorm)", color=c_opt, lw=2.5)
    axes[1].set_title("Secret Multiplication Gates (Beaver Triples)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Multiplication Gates (Billions)", fontsize=11)
    axes[1].legend(frameon=True, fontsize=9.5)

    # Panel 3: Cumulative Estimated Communication Bandwidth
    axes[2].plot(epochs, history["base_bw"], label="Baseline (Adam + ReLU + BatchNorm)", color=c_base, lw=2.5)
    axes[2].plot(epochs, history["opt_bw"], label="Ours (SGD + ReLU + ZeroRoundNorm)", color=c_opt, lw=2.5)
    axes[2].set_title("Estimated MPC Bandwidth Overhead", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("Cumulative Bandwidth (GB)", fontsize=11)
    axes[2].legend(frameon=True, fontsize=9.5)

    plt.suptitle("100-Epoch Secure GNN Telemetry: Plain Baseline vs. MPC-Optimized Pipeline", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved side-by-side comparison plot to: {save_path}")

    # Print exact 100-epoch summary metrics
    r_red = (1 - history["opt_rounds"][-1] / history["base_rounds"][-1]) * 100
    g_red = (1 - history["opt_gates"][-1] / history["base_gates"][-1]) * 100
    print("\n--- 100-Epoch Simulation Summary ---")
    print(f"Baseline Rounds : {history['base_rounds'][-1]:,.0f} | Ours: {history['opt_rounds'][-1]:,.0f} (-{r_red:.2f}%)")
    print(f"Baseline Gates  : {history['base_gates'][-1]:,.2f}M | Ours: {history['opt_gates'][-1]:,.2f}M (-{g_red:.2f}%)")
    print(f"Baseline BW     : {history['base_bw'][-1]:,.2f} GB | Ours: {history['opt_bw'][-1]:,.2f} GB (-{g_red:.2f}%)")


if __name__ == "__main__":
    # Option A: Fast standalone run using your exact log dimensions (3,173 batches/epoch, ~8,653 nodes & ~48,416 edges/batch)
    # Option B: Replace `batch_stats` below with `[(b.x.size(0), b.edge_index.size(1)) for b in tr_loader]` for exact batch-by-batch counts!
    batches_per_epoch = 3173
    batch_stats = [(8653, 48416)] * batches_per_epoch

    epochs, history = run_simulation(
        batch_stats=batch_stats,
        num_epochs=100,
        num_gnn_layers=2,
        n_hidden=66,
        num_features=3,
        edge_dim=6,
        num_parameters=33692,
        edge_updates=False,
    )
    plot_comparison(epochs, history)