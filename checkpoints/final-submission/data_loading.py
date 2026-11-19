import pandas as pd
import numpy as np
import torch
import logging
import itertools
from data_util import GraphData, HeteroData, z_norm, create_hetero_obj

NODE_FEATURES = ['net_flow', 'amount_std', 'dwell_time']

FEATURE_CLIP = 10.0


def clip_features(t):
    '''Bound z-normalized values to +/- FEATURE_CLIP so the polynomial activation cannot overflow.'''
    return t.clamp(-FEATURE_CLIP, FEATURE_CLIP)


def compute_node_stats(df_edges_subset, max_n_id):
    '''Node features from the given edges. Nodes with no edges get 0.'''
    out_amounts = df_edges_subset.groupby('from_id')['Amount Received'].agg(['sum', 'std']).rename(columns={'sum': 'out_vol', 'std': 'out_std'})
    in_amounts = df_edges_subset.groupby('to_id')['Amount Received'].agg(['sum', 'std']).rename(columns={'sum': 'in_vol', 'std': 'in_std'})

    node_stats = pd.DataFrame({'NodeID': np.arange(max_n_id)}).set_index('NodeID')
    node_stats = node_stats.join(out_amounts).join(in_amounts).fillna(0)

    # net_flow: relative balance between incoming and outgoing volume
    node_stats['net_flow'] = (node_stats['in_vol'] - node_stats['out_vol']) / (node_stats['in_vol'] + node_stats['out_vol'] + 1e-6)
    # amount_std: spread of transaction amounts touching the account
    node_stats['amount_std'] = (node_stats['out_std'] + node_stats['in_std']) / 2.0

    # dwell_time: mean time between consecutive transactions per node
    from_times = df_edges_subset[['from_id', 'Timestamp']].rename(columns={'from_id': 'NodeID'})
    to_times = df_edges_subset[['to_id', 'Timestamp']].rename(columns={'to_id': 'NodeID'})
    all_node_times = pd.concat([from_times, to_times]).sort_values(['NodeID', 'Timestamp'])
    all_node_times['time_delta'] = all_node_times.groupby('NodeID')['Timestamp'].diff().fillna(0)
    mean_dwell = all_node_times.groupby('NodeID')['time_delta'].mean()
    node_stats['dwell_time'] = mean_dwell.reindex(np.arange(max_n_id)).fillna(0)

    return node_stats[NODE_FEATURES].reset_index(drop=True)


def get_data(args, data_config):
    '''Loads the AML transaction data.

    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''

    transaction_file = f"{data_config['paths']['aml_data']}/{args.data}/formatted_transactions.csv"
    df_edges = pd.read_csv(transaction_file)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

    logging.info(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    logging.info(f"Number of nodes (holdings doing transcations) = {max_n_id}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    node_features = NODE_FEATURES

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features}')
    logging.info(f'Clip after normalization: +/-{FEATURE_CLIP}')

    edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    n_days = int(timestamps.max() / (3600 * 24) + 1)
    n_samples = y.shape[0]
    logging.info(f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

    #data splitting
    daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], []
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_irs.append(y[day_inds].float().mean())
        weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])

    split_per = [0.6, 0.2, 0.2]
    daily_totals = np.array(daily_trans)
    d_ts = daily_totals
    I = list(range(len(d_ts)))
    split_scores = dict()
    for i,j in itertools.combinations(I, 2):
        if j >= i:
            split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v/split_totals_sum for v in split_totals]
            split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
            score = max(split_error)
            split_scores[(i,j)] = score
        else:
            continue

    i,j = min(split_scores, key=split_scores.get)
    split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
    logging.info(f'Calculate split: {split}')

    split_inds = {k: [] for k in range(3)}
    for i in range(3):
        for day in split[i]:
            split_inds[i].append(daily_inds[day])

    tr_inds = torch.cat(split_inds[0])
    val_inds = torch.cat(split_inds[1])
    te_inds = torch.cat(split_inds[2])

    logging.info(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
            f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0][:5]}")
    logging.info(f"Total val samples: {val_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
        f"{y[val_inds].float().mean() * 100:.2f}% || Val days: {split[1][:5]}")
    logging.info(f"Total test samples: {te_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
        f"{y[te_inds].float().mean() * 100:.2f}% || Test days: {split[2][:5]}")

    #Creating the final data objects
    e_tr = tr_inds.numpy()
    e_val = np.concatenate([tr_inds, val_inds])

    # node features per split: train sees train edges, val sees train+val, test sees all
    tr_x = torch.tensor(compute_node_stats(df_edges.iloc[e_tr], max_n_id)[node_features].to_numpy()).float()
    val_x = torch.tensor(compute_node_stats(df_edges.iloc[e_val], max_n_id)[node_features].to_numpy()).float()
    te_x = torch.tensor(compute_node_stats(df_edges, max_n_id)[node_features].to_numpy()).float()

    tr_edge_index,  tr_edge_attr,  tr_y,  tr_edge_times  = edge_index[:,e_tr],  edge_attr[e_tr],  y[e_tr],  timestamps[e_tr]
    val_edge_index, val_edge_attr, val_y, val_edge_times = edge_index[:,e_val], edge_attr[e_val], y[e_val], timestamps[e_val]
    te_edge_index,  te_edge_attr,  te_y,  te_edge_times  = edge_index,          edge_attr,        y,        timestamps

    tr_data = GraphData (x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr,  timestamps=tr_edge_times )
    val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times)
    te_data = GraphData (x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr,  timestamps=te_edge_times )

    #Adding ports and time-deltas if applicable
    if args.ports:
        logging.info(f"Start: adding ports")
        tr_data.add_ports()
        val_data.add_ports()
        te_data.add_ports()
        logging.info(f"Done: adding ports")
    if args.tds:
        logging.info(f"Start: adding time-deltas")
        tr_data.add_time_deltas()
        val_data.add_time_deltas()
        te_data.add_time_deltas()
        logging.info(f"Done: adding time-deltas")

    #Normalize node features with training statistics only, then clip
    x_mean = tr_data.x.mean(0, keepdim=True)
    x_std = tr_data.x.std(0, keepdim=True) + 1e-6
    tr_data.x = clip_features((tr_data.x - x_mean) / x_std)
    val_data.x = clip_features((val_data.x - x_mean) / x_std)
    te_data.x = clip_features((te_data.x - x_mean) / x_std)

    #Normalize edge features (IBM per-split z_norm), then clip
    if not args.model == 'rgcn':
        tr_data.edge_attr = clip_features(z_norm(tr_data.edge_attr))
        val_data.edge_attr = clip_features(z_norm(val_data.edge_attr))
        te_data.edge_attr = clip_features(z_norm(te_data.edge_attr))
    else:
        tr_data.edge_attr[:, :-1] = clip_features(z_norm(tr_data.edge_attr[:, :-1]))
        val_data.edge_attr[:, :-1] = clip_features(z_norm(val_data.edge_attr[:, :-1]))
        te_data.edge_attr[:, :-1] = clip_features(z_norm(te_data.edge_attr[:, :-1]))

    logging.info(f"node x min: {tr_data.x.min(0).values.tolist()} max: {tr_data.x.max(0).values.tolist()}")
    logging.info(f"edge_attr min: {tr_data.edge_attr.min(0).values.tolist()} max: {tr_data.edge_attr.max(0).values.tolist()}")

    #Create heterogenous if reverse MP is enabled
    if args.reverse_mp:
        tr_data = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr, tr_data.timestamps, args)
        val_data = create_hetero_obj(val_data.x,  val_data.y,  val_data.edge_index,  val_data.edge_attr, val_data.timestamps, args)
        te_data = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr, te_data.timestamps, args)

    logging.info(f'train data object: {tr_data}')
    logging.info(f'validation data object: {val_data}')
    logging.info(f'test data object: {te_data}')

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds