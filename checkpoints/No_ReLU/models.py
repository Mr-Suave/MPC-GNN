import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv
import torch.nn.functional as F
import torch
import logging
from torch.nn import Linear
import math


class PolyAct(nn.Module):
    '''Polynomial replacement for ReLU: a*x^2 + b*x + c.
    Fixed coefficients, no comparisons, so it needs only multiplications under MPC.'''
    def __init__(self, a=0.125, b=0.5, c=0.25, min_val=-5.0, max_val=5.0):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        x_clamped = torch.clamp(x, self.min_val, self.max_val)
        return self.a * x_clamped * x_clamped + self.b * x_clamped + self.c


class MPCLayerNorm(nn.Module):
    """MPC-Friendly Layer Normalization using Newton-Raphson."""
    def __init__(self, n_hidden, eps=1e-5, nr_iters=4):
        super().__init__()
        self.n_hidden = n_hidden
        self.eps = eps
        self.nr_iters = nr_iters
        
        # Learnable affine parameters (costs 1 MPC round to multiply)
        self.weight = nn.Parameter(torch.ones(n_hidden))
        self.bias = nn.Parameter(torch.zeros(n_hidden))

    def forward(self, x):
        # 1. Mean subtraction (0 MPC rounds: local sum, public scalar division)
        mean = x.mean(dim=-1, keepdim=True)
        x_centered = x - mean
        
        # 2. Variance (1 MPC round: squaring the secret shared value)
        var = (x_centered ** 2).mean(dim=-1, keepdim=True)
        x_val = var + self.eps
        
        # 3. Newton-Raphson for Inverse Square Root (1/sqrt(x))
        # Initial guess (Bootstrapped for PyTorch simulation stability)
        with torch.no_grad():
            y = torch.rsqrt(x_val) 
            
        # Newton-Raphson Iterations (3 rounds per iteration)
        # Equation: y_{n+1} = 0.5 * y_n * (3 - x * y_n^2)
        for _ in range(self.nr_iters):
            y_sq = y * y               # Round 1
            term = x_val * y_sq        # Round 2
            y = 0.5 * y * (3.0 - term) # Round 3 (0.5 and 3.0 are public constants)
            
        inv_std = y
        
        # 4. Final scaling (1 round: multiplying secret norm by secret weight)
        norm_x = x_centered * inv_std
        return norm_x * self.weight + self.bias



class GINe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2,
                n_hidden=100, edge_updates=False, residual=True,
                edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        
        self.dropout = dropout
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.layer_norms = nn.ModuleList() 
        
        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden),
                PolyAct(),  # Swapped nn.ReLU() -> PolyAct()
                nn.Linear(self.n_hidden, self.n_hidden)
                ), edge_dim=self.n_hidden)
            
            if self.edge_updates:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden),
                    PolyAct(),  # Swapped nn.ReLU() -> PolyAct()
                    nn.Linear(self.n_hidden, self.n_hidden),
                ))
            self.convs.append(conv)
            self.layer_norms.append(MPCLayerNorm(self.n_hidden)) 

        self.mlp = nn.Sequential(
            Linear(n_hidden * 3, 32),
            PolyAct(),  # Swapped nn.ReLU() -> PolyAct()
            Linear(32, n_classes),
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            # 1. 0-round local dropout
            x_drop = F.dropout(x, p=self.dropout, training=self.training)
            
            # 2. Convolution
            h = self.convs[i](x_drop, edge_index, edge_attr)
            
            # 3. Newton-Raphson LayerNorm
            h_norm = self.layer_norms[i](h)
            
            # 4. Residual scaling
            x = (x + h_norm) / 2
            
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden)
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        
        x = F.dropout(x, p=self.final_dropout, training=self.training)
        return self.mlp(x)


    

class GATe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, n_hidden=100, n_heads=4, edge_updates=False, edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        # GAT specific code
        tmp_out = n_hidden // n_heads
        n_hidden = tmp_out * n_heads

        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.dropout = dropout
        self.final_dropout = final_dropout

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            conv = GATConv(self.n_hidden, tmp_out, self.n_heads, concat = True, dropout = self.dropout, add_self_loops = True, edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden),nn.ReLU(),nn.Linear(self.n_hidden, self.n_hidden),))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(25, n_classes))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        logging.debug(f"x.shape = {x.shape}")
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        logging.debug(f"x.shape = {x.shape}")
        out = x

        return self.mlp(out)


class PNA(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2,
                n_hidden=100, edge_updates=True,
                edge_dim=None, dropout=0.0, final_dropout=0.5, deg=None):
        super().__init__()
        n_hidden = int((n_hidden // 5) * 5)
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=n_hidden, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                              Linear(25, n_classes))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        logging.debug(f"x.shape = {x.shape}")
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        logging.debug(f"x.shape = {x.shape}")
        out = x
        return self.mlp(out)


class RGCN(nn.Module):
    def __init__(self, num_features, edge_dim, num_relations, num_gnn_layers, n_classes=2,
                n_hidden=100, edge_update=False,
                residual=True,
                dropout=0.0, final_dropout=0.5, n_bases=-1):
        super(RGCN, self).__init__()

        self.num_features = num_features
        self.num_gnn_layers = num_gnn_layers
        self.n_hidden = n_hidden
        self.residual = residual
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.n_classes = n_classes
        self.edge_update = edge_update
        self.num_relations = num_relations
        self.n_bases = n_bases

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.mlp = nn.ModuleList()

        if self.edge_update:
            self.emlps = nn.ModuleList()
            self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))

        for _ in range(self.num_gnn_layers):
            conv = RGCNConv(self.n_hidden, self.n_hidden, num_relations, num_bases=self.n_bases)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(self.n_hidden))

            if self.edge_update:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden),
                ))

        self.mlp = nn.Sequential(Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                              Linear(25, n_classes))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, RGCNConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        edge_type = edge_attr[:, -1].long()
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x =  (x + F.relu(self.bns[i](self.convs[i](x, edge_index, edge_type)))) / 2
            if self.edge_update:
                edge_attr = (edge_attr + F.relu(self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)))) / 2

        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        x = self.mlp(x)
        out = x

        return x