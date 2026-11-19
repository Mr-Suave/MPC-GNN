import numpy as np
import matplotlib.pyplot as plt


class MPCTelemetrySimulator:
    """
    Unified MPC Telemetry Tracker with Matrix Beaver Triples:
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
        self.total_online_elements = 0  # Secret-shared elements opened over the network (per party)
        self.total_local_macs = 0       # Local plaintext/share multiply-accumulate ops (0 comm)

    @staticmethod
    def _matrix_triple_cost(n_rows, d_in, h_out):
        """
        Matrix Beaver Triple cost for Y = X @ W where X is (N, D) and W is (D, H):
          - Forward: Opens E = X - A (N*D) and F = W - B (D*H) -> N*D + D*H elements.
          - Backward: dX = dY @ W^T and dW = X^T @ dY reuse masks F and E -> only dY (N*H) opened.
          - Local Compute: 1 forward + 2 backward MatMuls = 3 * N * D * H MACs.
        """
        fwd_elems = (n_rows * d_in) + (d_in * h_out)
        bwd_elems = (n_rows * h_out)
        macs = 3 * n_rows * d_in * h_out
        return fwd_elems + bwd_elems, macs

    def step_batch(self, num_nodes, num_edges):
        fwd_rounds = 0
        batch_online_elems = 0
        batch_local_macs = 0

        # 1. Initial Embeddings (node_emb, edge_emb) -> 1 parallel round
        fwd_rounds += 1
        for rows, d_in, h_out in [
            (num_nodes, self.num_features, self.n_hidden),
            (num_edges, self.edge_dim, self.n_hidden),
        ]:
            elems, macs = self._matrix_triple_cost(rows, d_in, h_out)
            batch_online_elems += elems
            batch_local_macs += macs

        # 2. GNN Layers
        for _ in range(self.num_gnn_layers):
            # GINEConv MLP: Linear (1) -> ReLU (8) -> Linear (1)
            layer_mlp_rounds = 1 + self.relu_rounds + 1
            fwd_rounds += layer_mlp_rounds + self.bn_rounds

            # Two Linear(n_hidden, n_hidden) layers (Matrix Beaver Triples)
            for _ in range(2):
                elems, macs = self._matrix_triple_cost(num_nodes, self.n_hidden, self.n_hidden)
                batch_online_elems += elems
                batch_local_macs += macs

            # ReLU element-wise masking gates (fwd + bwd = 2 * N * H)
            batch_online_elems += 2 * (num_nodes * self.n_hidden)

            if self.mode == "baseline":
                # BatchNorm element-wise triples (fwd + bwd = 3 * 14 gates/elem)
                batch_online_elems += 3 * (num_nodes * self.n_hidden * 14)

            if self.edge_updates:
                fwd_rounds += layer_mlp_rounds
                for d_in, h_out in [(3 * self.n_hidden, self.n_hidden), (self.n_hidden, self.n_hidden)]:
                    elems, macs = self._matrix_triple_cost(num_edges, d_in, h_out)
                    batch_online_elems += elems
                    batch_local_macs += macs
                batch_online_elems += 2 * (num_edges * self.n_hidden)

        # 3. Head MLP: Linear(3*H, 32) -> ReLU -> Linear(32, 2)
        fwd_rounds += (1 + self.relu_rounds + 1)
        for d_in, h_out in [(3 * self.n_hidden, 32), (32, 2)]:
            elems, macs = self._matrix_triple_cost(num_edges, d_in, h_out)
            batch_online_elems += elems
            batch_local_macs += macs
        batch_online_elems += 2 * (num_edges * 32)

        # 4. Loss (Cross-Entropy)
        loss_rounds = self.loss_rounds

        # 5. Backward Pass rounds mirror forward round depth
        bwd_rounds = fwd_rounds

        # 6. Optimizer Step (element-wise NR triples for baseline Adam; 0 for SGD)
        opt_rounds = self.opt_rounds
        opt_elems = (self.num_parameters * 17) if self.mode == "baseline" else 0

        self.total_comm_rounds += (fwd_rounds + loss_rounds + bwd_rounds + opt_rounds)
        self.total_online_elements += (batch_online_elems + opt_elems)
        self.total_local_macs += batch_local_macs

    def get_metrics(self):
        # 64-bit shares (8 bytes) * 2 parties for online opened elements -> converted to GB
        est_bandwidth_gb = (self.total_online_elements * 8 * 2) / (1024 ** 3)
        return {
            "rounds": self.total_comm_rounds,
            "online_elements_millions": self.total_online_elements / 1e6,
            "bandwidth_gb": est_bandwidth_gb,
            "local_macs_billions": self.total_local_macs / 1e9,
        }


def run_simulation(batch_stats, num_epochs=100, num_gnn_layers=2, n_hidden=66,
                   num_features=3, edge_dim=6, num_parameters=33692, edge_updates=False):
    opt_sim = MPCTelemetrySimulator("optimized", num_gnn_layers, n_hidden, num_features, edge_dim, num_parameters, edge_updates)
    base_sim = MPCTelemetrySimulator("baseline", num_gnn_layers, n_hidden, num_features, edge_dim, num_parameters, edge_updates)

    epochs = np.arange(1, num_epochs + 1)
    history = {
        "opt_rounds": [], "base_rounds": [],
        "opt_elems": [],  "base_elems": [],
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
        history["opt_elems"].append(o_m["online_elements_millions"])
        history["base_elems"].append(b_m["online_elements_millions"])
        history["opt_bw"].append(o_m["bandwidth_gb"])
        history["base_bw"].append(b_m["bandwidth_gb"])

    return epochs, history


def plot_comparison(epochs, history, save_path="mpc_100_epoch_matrix_triples.png"):
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

    # Panel 2: Cumulative Online Opened Elements (Matrix Beaver Triples)
    axes[1].plot(epochs, np.array(history["base_elems"]) / 1e3, label="Baseline (Adam + ReLU + BatchNorm)", color=c_base, lw=2.5)
    axes[1].plot(epochs, np.array(history["opt_elems"]) / 1e3, label="Ours (SGD + ReLU + ZeroRoundNorm)", color=c_opt, lw=2.5)
    axes[1].set_title("Online Opened Secret Elements", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Opened Elements (Billions)", fontsize=11)
    axes[1].legend(frameon=True, fontsize=9.5)

    # Panel 3: Cumulative Online Communication Bandwidth
    axes[2].plot(epochs, history["base_bw"] / np.float64(1024), label="Baseline (Adam + ReLU + BatchNorm)", color=c_base, lw=2.5)
    axes[2].plot(epochs, history["opt_bw"] / np.float64(1024), label="Ours (SGD + ReLU + ZeroRoundNorm)", color=c_opt, lw=2.5)
    axes[2].set_title("Online MPC Bandwidth (Matrix Triples)", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("Cumulative Bandwidth (TB)", fontsize=11)
    axes[2].legend(frameon=True, fontsize=9.5)

    plt.suptitle("100-Epoch Secure GNN Telemetry (Matrix Beaver Triples): Baseline vs. Optimized", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved side-by-side comparison plot to: {save_path}")

    # Print exact 100-epoch summary metrics
    r_red = (1 - history["opt_rounds"][-1] / history["base_rounds"][-1]) * 100
    e_red = (1 - history["opt_elems"][-1] / history["base_elems"][-1]) * 100
    print("\n--- 100-Epoch Matrix Beaver Triple Simulation Summary ---")
    print(f"Baseline Rounds   : {history['base_rounds'][-1]:,.0f} | Ours: {history['opt_rounds'][-1]:,.0f} (-{r_red:.2f}%)")
    print(f"Baseline Elements : {history['base_elems'][-1]:,.2f}M | Ours: {history['opt_elems'][-1]:,.2f}M (-{e_red:.2f}%)")
    print(f"Baseline BW       : {history['base_bw'][-1]/1024:,.2f} TB | Ours: {history['opt_bw'][-1]/1024:,.2f} TB (-{e_red:.2f}%)")


if __name__ == "__main__":
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