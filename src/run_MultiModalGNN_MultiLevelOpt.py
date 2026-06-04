import argparse
import os
import copy
import pickle
import mlflow
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from models import GNN
from my_utils import set_seed, setup_env, move_data_to_device, update_best_model_snapshot \
    , save_metrics, get_edge_index, handle_isolated_nodes, get_gnn_embeddings
from data_loader import create_data_loader
from model_eval import TrainLogMetrics, TestLogMetrics, eval_pred
from plot_utils import plot_losses

DEFAULT_HYPERPARAMETERS = {'train_perc': .6,
                           'val_perc': .2,
                           'test_perc': .2,
                           'num_splits': 5,
                           'aggr_type': 'mean'}
DEFAULT_TRAIN_HYPERPARAMETERS = {'input_embed': 'positional', 'epochs': 1000, 'learning_rate': 1e-3,
                                 'early_stopping_limit': 10, 'check_loss_freq': 5}
DEFAULT_MODEL_HYPERPARAMETERS = {'gnn_type': 'gcn', 'latent_dim': 32, 'dropout': 0.2}
ALL_COUNTRIES = ['china', 'iran', 'UAE', 'cuba', 'russia', 'venezuela']


# =============================================================================
# MODULE B: Focal Loss (Inherited from TSET)
# =============================================================================
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, preds, targets):
        preds = preds.clamp(1e-7, 1 - 1e-7)
        bce = -(targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))
        p_t = preds * targets + (1 - preds) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_weight * focal_weight * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# MODULE C: MMD-based Linguistic Distance Detector (Inherited from TSET)
# =============================================================================
def compute_mmd(x, y, kernel_bandwidths=None):
    if kernel_bandwidths is None:
        kernel_bandwidths = [0.1, 0.5, 1.0, 2.0, 5.0]

    if x.size(0) > 500:
        idx = torch.randperm(x.size(0))[:500]
        x = x[idx]
    if y.size(0) > 500:
        idx = torch.randperm(y.size(0))[:500]
        y = y[idx]

    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    x_sq = (x ** 2).sum(dim=1, keepdim=True)
    y_sq = (y ** 2).sum(dim=1, keepdim=True)

    dist_xx = x_sq + x_sq.t() - 2 * xx
    dist_yy = y_sq + y_sq.t() - 2 * yy
    dist_xy = x_sq + y_sq.t() - 2 * xy

    mmd2 = 0.0
    for bw in kernel_bandwidths:
        K_xx = torch.exp(-dist_xx / (2 * bw ** 2))
        K_yy = torch.exp(-dist_yy / (2 * bw ** 2))
        K_xy = torch.exp(-dist_xy / (2 * bw ** 2))
        mmd2 += K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

    return (mmd2 / len(kernel_bandwidths)).item()


def detect_structure_only_mode(target_text_features, countries_data, mmd_threshold=0.015):
    target_feats = target_text_features.float()
    target_mmds = {}
    all_mmds = []
    country_list = list(countries_data.keys())

    for country in country_list:
        src_feats = countries_data[country]['node_features'].float()
        mmd_val = compute_mmd(target_feats, src_feats)
        target_mmds[country] = mmd_val
        all_mmds.append(mmd_val)
        print(f"  [SOTM] MMD(target, {country}) = {mmd_val:.5f}")

    for i in range(len(country_list)):
        for j in range(i + 1, len(country_list)):
            ci, cj = country_list[i], country_list[j]
            src_i = countries_data[ci]['node_features'].float()
            src_j = countries_data[cj]['node_features'].float()
            ss_mmd = compute_mmd(src_i, src_j)
            all_mmds.append(ss_mmd)

    min_mmd = min(target_mmds.values())
    mean_mmd = sum(all_mmds) / len(all_mmds)
    activated = min_mmd > mean_mmd
    print(f"  [SOTM] min_target_MMD={min_mmd:.5f}, all_pairs_mean={mean_mmd:.5f} -> Structure-Only={activated}")
    return activated, min_mmd


# =============================================================================
# LEVEL 1: Multilingual Semantic Centroid Alignment (MSCA)
# =============================================================================
def apply_msca(target_text_features, countries_data):
    """
    Standardizes target text features and scales/translates them to match the
    global aggregate mean and standard deviation of all source domains combined.
    """
    target_mean = target_text_features.mean(dim=0, keepdim=True)
    target_std = target_text_features.std(dim=0, keepdim=True) + 1e-6

    # Concatenate all source features to get baseline statistics
    source_features_list = [countries_data[c]['node_features'] for c in countries_data]
    all_source_features = torch.cat(source_features_list, dim=0)

    source_mean = all_source_features.mean(dim=0, keepdim=True)
    source_std = all_source_features.std(dim=0, keepdim=True) + 1e-6

    # Perform whitening translation
    aligned_target = (target_text_features - target_mean) / target_std
    aligned_target = aligned_target * source_std + source_mean
    return aligned_target


# =============================================================================
# LEVEL 2: Graph-Regularized Posterior Smoothing (GRPS)
# =============================================================================
def graph_regularized_posterior_smoothing(pred_probs, datasets, alpha=0.5, max_iter=20, tol=1e-4):
    """
    Applies Label Propagation to propagate predictions across target coordination networks.
    """
    num_nodes = datasets['graph'].number_of_nodes()
    coRT_edges = list(datasets['coRT'].edges())
    coURL_edges = list(datasets['coURL'].edges())
    all_edges = coRT_edges + coURL_edges

    if len(all_edges) == 0:
        print("  [GRPS] Warning: No coordination edges found in coRT/coURL. Skipping.")
        return pred_probs

    row = [u for u, v in all_edges] + [v for u, v in all_edges]
    col = [v for u, v in all_edges] + [u for u, v in all_edges]
    data = [1.0] * len(row)

    row_clean = []
    col_clean = []
    data_clean = []
    for r, c, d in zip(row, col, data):
        if r < num_nodes and c < num_nodes:
            row_clean.append(r)
            col_clean.append(c)
            data_clean.append(d)

    if len(data_clean) == 0:
        return pred_probs

    # Adjacency matrix representation
    A = sp.coo_matrix((data_clean, (row_clean, col_clean)), shape=(num_nodes, num_nodes)).tocsr()

    # D^-1/2 A D^-1/2 Symmetric Normalization
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees[degrees == 0] = 1e-6
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    S = D_inv_sqrt.dot(A).dot(D_inv_sqrt)

    # Power Iteration
    P = pred_probs.copy()
    P_0 = pred_probs.copy()

    for it in range(max_iter):
        P_next = alpha * S.dot(P) + (1 - alpha) * P_0
        if np.linalg.norm(P_next - P) < tol:
            P = P_next
            break
        P = P_next

    return P


# =============================================================================
# LEVEL 3: Test-Time Graph Contrastive Adaptation (TT-GCA)
# =============================================================================
def run_test_time_adaptation(model, struct_node_features, text_node_features, edge_index, device, epochs=5, lr=1e-3, batch_size=10000):
    """
    Self-supervised Test-Time adaptation using link prediction over a sampled batch of target edges.
    This avoids CUDA out-of-memory errors on large graphs.
    """
    optimizer = torch.optim.Adam([
        {'params': model.cross_attention_to_text.parameters()},
        {'params': model.cross_attention_to_struct.parameters()},
        {'params': model.struct_projector.parameters()},
        {'params': model.text_projector.parameters()},
        {'params': model.joint_projector.parameters()},
        {'params': model.gnn.conv1.parameters()}
    ], lr=lr)

    model.train()
    num_nodes = struct_node_features.shape[0]
    pos_edge_index = edge_index

    if pos_edge_index.size(1) == 0:
        return

    num_edges = pos_edge_index.size(1)
    effective_batch_size = min(batch_size, num_edges)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Decoupled projection pass
        struct_projection = model.struct_projector(struct_node_features) * model.cross_attention_to_struct(text_node_features)
        text_projection = model.text_projector(text_node_features) * model.cross_attention_to_text(struct_node_features)

        multimodal_node_features = model.joint_projector(
            torch.concat([struct_projection, text_projection], dim=-1)
        )

        # First GNN layer representation mapping
        x = model.gnn.conv1(multimodal_node_features, edge_index)
        x = model.gnn.activation_fn(x)
        x = model.gnn.dropout(x)

        # Sample positive edges to prevent CUDA OOM
        perm = torch.randperm(num_edges, device=device)[:effective_batch_size]
        sampled_pos_edges = pos_edge_index[:, perm]

        # Dot product prediction
        pos_src, pos_dst = sampled_pos_edges[0], sampled_pos_edges[1]
        pos_scores = torch.sum(x[pos_src] * x[pos_dst], dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_scores) + 1e-7).mean()

        # Sample negative edges
        neg_src = torch.randint(0, num_nodes, (effective_batch_size,), device=device)
        neg_dst = torch.randint(0, num_nodes, (effective_batch_size,), device=device)
        neg_scores = torch.sum(x[neg_src] * x[neg_dst], dim=1)
        neg_loss = -torch.log(1.0 - torch.sigmoid(neg_scores) + 1e-7).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        print(f"    [TT-GCA] Unsupervised link prediction Epoch {epoch+1}/{epochs} loss: {loss.item():.4f}")


def stratified_random_boolean_tensor(n, batch_size, device, labels):
    assert len(labels) == n, "The length of labels must match n."
    assert batch_size <= n, "Batch size cannot be larger than the number of available elements."

    bool_tensor = torch.zeros(n, dtype=torch.bool)
    indices_0 = torch.where(labels == 0)[0]
    indices_1 = torch.where(labels == 1)[0]
    batch_size_0 = batch_size // 2
    batch_size_1 = batch_size - batch_size_0

    batch_size_0 = min(batch_size_0, len(indices_0))
    batch_size_1 = min(batch_size_1, len(indices_1))

    sampled_indices_0 = indices_0[torch.randperm(len(indices_0))[:batch_size_0]]
    sampled_indices_1 = indices_1[torch.randperm(len(indices_1))[:batch_size_1]]
    bool_tensor[sampled_indices_0] = True
    bool_tensor[sampled_indices_1] = True
    return bool_tensor.to(device)


def read_all_data(device_id, dataset_name, hyper_params, train_hyperparams, model_hyperparams):
    device, base_dir, interim_data_dir, data_dir = setup_env(device_id, dataset_name, hyper_params)
    datasets = create_data_loader(data_dir, hyper_params['tsim_th'],
                                  hyper_params['train_perc'], hyper_params['undersampling'])
    datasets = move_data_to_device(datasets, device)
    _, network = handle_isolated_nodes(datasets['graph'])
    edge_index = get_edge_index(network, data_dir)
    edge_index = edge_index.to(device)

    num_mostPop = hyper_params['most_pop']
    if (data_dir / f'sbert_nodeattributes_mostPop{num_mostPop}.pt').exists():
        node_features = torch.load(data_dir / f'sbert_nodeattributes_mostPop{num_mostPop}.pt', map_location=device)
    else:
        path = str(data_dir / f'sbert_nodeattributes_mostPop{num_mostPop}.pt')
        raise Exception(f'path {path} does not exist')
    node_features = node_features.to(device)

    struct_node_features = get_gnn_embeddings(data_dir, {'type': train_hyperparams['input_embed'],
                                                          'trace_type': hyper_params['trace_type'],
                                                          'latent_dim': model_hyperparams['latent_dim'],
                                                          'seed': hyper_params['seed'],
                                                          'num_nodes': network.number_of_nodes(),
                                                          'graph': network, 'device': device,
                                                          'dataset_name': dataset_name, 'base_dir': base_dir,
                                                          'num_cores': 8,
                                                          'aggr_type': hyper_params['aggr_type']})
    struct_node_features = struct_node_features.to(device)
    return device, base_dir, interim_data_dir, data_dir, datasets, edge_index, network, node_features, struct_node_features


def create_model(model_hyperparams):
    class GNN_CrossAttention_MultiLevel(torch.nn.Module):
        def __init__(self, num_node_features, hidden_dim, num_classes, num_textual_features,
                     num_structural_features, activation_fn=torch.nn.ReLU(), dropout_p=0.2,
                     gnn_type='gcn'):
            super().__init__()
            self.gnn = GNN(num_node_features=num_node_features * 2,
                           hidden_dim=hidden_dim * 2, num_classes=num_classes,
                           dropout_p=dropout_p, gnn_type=gnn_type)
            self.cross_attention_to_text = torch.nn.Linear(num_structural_features, hidden_dim)
            self.cross_attention_to_struct = torch.nn.Linear(num_textual_features, hidden_dim)
            self.struct_projector = torch.nn.Sequential(
                torch.nn.Linear(num_structural_features, hidden_dim),
                torch.nn.ReLU()
            )
            self.text_projector = torch.nn.Sequential(
                torch.nn.Linear(num_textual_features, hidden_dim),
                torch.nn.ReLU()
            )
            self.joint_projector = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
                torch.nn.ReLU()
            )

        def forward(self, text_node_features, struct_node_features, edge_index,
                    return_text_features=False, structure_only=False):
            struct_projection = (
                self.struct_projector(struct_node_features)
                * self.cross_attention_to_struct(text_node_features)
            )
            text_projection = (
                self.text_projector(text_node_features)
                * self.cross_attention_to_text(struct_node_features)
            )

            if structure_only:
                text_projection = torch.zeros_like(text_projection)

            multimodal_node_features = self.joint_projector(
                torch.concat([struct_projection, text_projection], dim=-1)
            )
            preds = self.gnn(multimodal_node_features, edge_index)

            if return_text_features:
                return preds, text_projection
            return preds

    return GNN_CrossAttention_MultiLevel(
        num_node_features=model_hyperparams['latent_dim'],
        hidden_dim=model_hyperparams['latent_dim'],
        num_classes=2,
        dropout_p=model_hyperparams['dropout'],
        gnn_type=model_hyperparams['gnn_type'],
        num_textual_features=model_hyperparams['num_textual_features'],
        num_structural_features=model_hyperparams['num_structural_features']
    )


def coral_loss(source, target):
    d = source.size(1)
    ns = source.size(0)
    nt = target.size(0)
    source_mean = torch.mean(source, dim=0, keepdim=True)
    source_center = source - source_mean
    xm = torch.matmul(source_center.t(), source_center) / (ns - 1)
    target_mean = torch.mean(target, dim=0, keepdim=True)
    target_center = target - target_mean
    yt = torch.matmul(target_center.t(), target_center) / (nt - 1)
    loss = torch.sum((xm - yt) ** 2) / (4 * d * d)
    return loss


def main(dataset_name, train_hyperparams, model_hyperparams, hyper_params, device_id):
    if model_hyperparams is None:
        model_hyperparams = DEFAULT_MODEL_HYPERPARAMETERS
    if train_hyperparams is None:
        train_hyperparams = DEFAULT_TRAIN_HYPERPARAMETERS
    if hyper_params is None:
        hyper_params = DEFAULT_HYPERPARAMETERS

    set_seed(hyper_params['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    device, base_dir, interim_data_dir, data_dir, datasets, edge_index, network, \
        node_features, struct_node_features = read_all_data(
            device_id, dataset_name, hyper_params, train_hyperparams, model_hyperparams)

    model_hyperparams['num_textual_features'] = node_features.shape[1]
    model_hyperparams['num_structural_features'] = struct_node_features.shape[1]

    # Preload target context for MMD distance checks
    other_countries_temp = [c for c in ALL_COUNTRIES if c != dataset_name]
    countries_data = {}
    for country in other_countries_temp:
        _, _, _, _, cd, cei, cn, cnf, csnf = read_all_data(
            device_id, country, hyper_params, train_hyperparams, model_hyperparams)
        countries_data[country] = {'node_features': cnf, 'struct_node_features': csnf,
                                    'datasets': cd, 'edge_index': cei, 'network': cn}

    # LEVEL 1: Multilingual Semantic Centroid Alignment (MSCA)
    if train_hyperparams.get('msca', False):
        print(f"\n[LEVEL 1 OPT] Applying Multilingual Semantic Centroid Alignment (MSCA) on {dataset_name} text embeddings...")
        node_features = apply_msca(node_features, countries_data)
        print("  MSCA alignment complete.")

    print(f"\n[MODULE C] Checking linguistic isolation for target: {dataset_name}")
    use_structure_only, min_mmd = detect_structure_only_mode(
        node_features, countries_data,
        mmd_threshold=train_hyperparams.get('sotm_threshold', 0.3)
    )
    print(f"  -> Structure-Only Transfer Mode (SOTM): {use_structure_only}\n")

    countries_numExamples = {c: countries_data[c]['struct_node_features'].shape[0] for c in countries_data}

    # Setup loggers
    train_logger = TrainLogMetrics(hyper_params['num_splits'], ['supervised'])
    val_logger = TestLogMetrics(hyper_params['num_splits'], ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    test_logger = TestLogMetrics(hyper_params['num_splits'], ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    test_logger_coRT = TestLogMetrics(hyper_params['num_splits'],
                                      ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    test_logger_coURL = TestLogMetrics(hyper_params['num_splits'],
                                       ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    test_logger_hashSeq = TestLogMetrics(hyper_params['num_splits'],
                                         ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    test_logger_fastRT = TestLogMetrics(hyper_params['num_splits'],
                                        ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    test_logger_tweetSim = TestLogMetrics(hyper_params['num_splits'],
                                          ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])

    coRT_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    coRT_mask[list(datasets['coRT'].nodes())] = True
    coURL_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    coURL_mask[list(datasets['coURL'].nodes())] = True
    hashSeq_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    hashSeq_mask[list(datasets['hashSeq'].nodes())] = True
    fastRT_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    fastRT_mask[list(datasets['fastRT'].nodes())] = True
    tweetSim_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    tweetSim_mask[list(datasets['tweetSim'].nodes())] = True

    numpy_labels = datasets['labels'].long().detach().cpu().numpy()
    num_epochs = train_hyperparams['num_epochs']
    metric_to_optimize = train_hyperparams['metric_to_optimize']

    # Focal Loss (Inherited)
    focal_gamma = train_hyperparams.get('focal_gamma', 2.0)
    focal_alpha = train_hyperparams.get('focal_alpha', 0.75)
    loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    print(f"[MODULE B] Focal Loss configuration: gamma={focal_gamma}, alpha={focal_alpha}")

    coral_weight = train_hyperparams.get('coral_weight', 500.0)

    for run_id in tqdm(range(hyper_params['num_splits']), 'Splits training'):
        BEST_VAL_METRIC = -np.inf
        best_model_path = interim_data_dir / f'model{run_id}.pth'
        model = create_model(model_hyperparams)
        model.to(device)

        optimizer = torch.optim.Adam(list(model.parameters()), lr=train_hyperparams['learning_rate'])
        early_stopping_cnt = 0

        for epoch in range(num_epochs):
            if early_stopping_cnt > train_hyperparams["early_stopping_limit"]:
                break
            model.train()
            optimizer.zero_grad()

            loss = 0
            task_loss_sum = 0
            coral_loss_sum = 0
            text_features_dict = {}

            for country in countries_data:
                train_mask = stratified_random_boolean_tensor(
                    countries_numExamples[country],
                    batch_size=128, device=device,
                    labels=countries_data[country]['datasets']['labels']
                )
                pred, text_feats = model(
                    countries_data[country]['node_features'],
                    countries_data[country]['struct_node_features'],
                    countries_data[country]['edge_index'],
                    return_text_features=True
                )
                text_features_dict[country] = text_feats[train_mask]

                task_loss = loss_fn(
                    pred.flatten()[train_mask],
                    countries_data[country]['datasets']['labels'][train_mask]
                )
                loss += task_loss
                task_loss_sum += task_loss.item()

            coral_val = 0
            country_list = list(countries_data.keys())
            num_c = len(country_list)
            pair_count = 0
            for i in range(num_c):
                for j in range(i + 1, num_c):
                    c1, c2 = country_list[i], country_list[j]
                    coral_val += coral_loss(text_features_dict[c1], text_features_dict[c2])
                    pair_count += 1
            if pair_count > 0:
                coral_val = coral_val / pair_count
            loss += coral_val * coral_weight
            coral_loss_sum = coral_val.item()

            loss.backward()
            optimizer.step()
            train_logger.train_update(run_id, 'supervised', loss.item())

            if epoch % train_hyperparams["check_loss_freq"] == 0:
                model.eval()
                with torch.no_grad():
                    pred = model(node_features, struct_node_features, edge_index,
                                 structure_only=use_structure_only).detach().cpu().numpy().flatten()

                    val_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['val'])
                    train_logger.val_update(run_id, val_metrics[train_hyperparams["metric_to_optimize"]])

                    if val_metrics[train_hyperparams["metric_to_optimize"]] > BEST_VAL_METRIC:
                        BEST_VAL_METRIC = val_metrics[train_hyperparams["metric_to_optimize"]]
                        torch.save(model.state_dict(), best_model_path)
                        early_stopping_cnt = 0
                    else:
                        early_stopping_cnt += 1
            else:
                train_logger.val_update(run_id, 0.0)

        # Load best model for inference
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # LEVEL 3: Test-Time Graph Contrastive Adaptation (TT-GCA)
        if train_hyperparams.get('tt_gca', False):
            print(f"\n[LEVEL 3 OPT] Running Test-Time Graph Contrastive Adaptation (TT-GCA) on {dataset_name} target graph...")
            run_test_time_adaptation(
                model, struct_node_features, node_features, edge_index, device,
                epochs=train_hyperparams.get('tt_gca_epochs', 5),
                lr=train_hyperparams.get('tt_gca_lr', 1e-3)
            )

        model.eval()
        with torch.no_grad():
            pred = model(node_features, struct_node_features, edge_index,
                         structure_only=use_structure_only).detach().cpu().numpy().flatten()

        # LEVEL 2: Graph-Regularized Posterior Smoothing (GRPS)
        if train_hyperparams.get('grps', False):
            print(f"\n[LEVEL 2 OPT] Smoothing predictions via Graph-Regularized Posterior Smoothing (GRPS)...")
            pred = graph_regularized_posterior_smoothing(
                pred, datasets,
                alpha=train_hyperparams.get('grps_alpha', 0.5),
                max_iter=train_hyperparams.get('grps_max_iter', 20)
            )

        # Evaluate on val set
        val_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['val'])
        for metric_name in val_metrics:
            val_logger.update(metric_name, run_id, val_metrics[metric_name])

        # Evaluate on overall test set
        test_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['test'])
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])

        # Subnet evaluations
        test_metrics_coRT = eval_pred(numpy_labels, pred > 0.5,
                                      np.logical_and(datasets['splits'][run_id]['test'], coRT_mask),
                                      prob_pred=pred)
        for metric_name in test_metrics_coRT:
            test_logger_coRT.update(metric_name, run_id, test_metrics_coRT[metric_name])

        test_metrics_coURL = eval_pred(numpy_labels, pred > 0.5,
                                       np.logical_and(datasets['splits'][run_id]['test'], coURL_mask),
                                       prob_pred=pred)
        for metric_name in test_metrics_coURL:
            test_logger_coURL.update(metric_name, run_id, test_metrics_coURL[metric_name])

        test_metrics_hashSeq = eval_pred(numpy_labels, pred > 0.5,
                                         np.logical_and(datasets['splits'][run_id]['test'], hashSeq_mask),
                                         prob_pred=pred)
        for metric_name in test_metrics_hashSeq:
            test_logger_hashSeq.update(metric_name, run_id, test_metrics_hashSeq[metric_name])

        test_metrics_fastRT = eval_pred(numpy_labels, pred > 0.5,
                                        np.logical_and(datasets['splits'][run_id]['test'], fastRT_mask),
                                        prob_pred=pred)
        for metric_name in test_metrics_fastRT:
            test_logger_fastRT.update(metric_name, run_id, test_metrics_fastRT[metric_name])

        test_metrics_tweetSim = eval_pred(numpy_labels, pred > 0.5,
                                          np.logical_and(datasets['splits'][run_id]['test'], tweetSim_mask),
                                          prob_pred=pred)
        for metric_name in test_metrics_tweetSim:
            test_logger_tweetSim.update(metric_name, run_id, test_metrics_tweetSim[metric_name])

    for split_num in tqdm(range(hyper_params['num_splits']), 'Splits post-training'):
        mlflow.log_artifact(interim_data_dir / f'model{split_num}.pth')

    save_metrics(val_logger, interim_data_dir, 'VAL')
    save_metrics(test_logger, interim_data_dir, 'TEST')
    save_metrics(test_logger_coRT, interim_data_dir, 'TEST_coRT')
    save_metrics(test_logger_coURL, interim_data_dir, 'TEST_coURL')
    save_metrics(test_logger_hashSeq, interim_data_dir, 'TEST_hashSeq')
    save_metrics(test_logger_fastRT, interim_data_dir, 'TEST_fastRT')
    save_metrics(test_logger_tweetSim, interim_data_dir, 'TEST_tweetSim')

    update_best_model_snapshot(data_dir, metric_to_optimize, test_logger,
                               hyper_params['num_splits'], interim_data_dir)

    print(f"\n[Multi-Level Optimization Summary for {dataset_name}]")
    print(f"  MSCA (Level 1)                = {train_hyperparams.get('msca', False)}")
    print(f"  GRPS (Level 2)                = {train_hyperparams.get('grps', False)} (alpha={train_hyperparams.get('grps_alpha', 0.5)})")
    print(f"  TT-GCA (Level 3)              = {train_hyperparams.get('tt_gca', False)} (epochs={train_hyperparams.get('tt_gca_epochs', 5)})")
    print(f"  SOTM (Structure-Only Mode)    = {use_structure_only} (min_MMD={min_mmd:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Multi-Level Optimized Zero-Shot GNN (MSCA + GRPS + TT-GCA)")
    parser.add_argument('-dataset_name', '--dataset', type=str, default='cuba')
    parser.add_argument('-seed', '--seed', type=int, default=12121995)
    parser.add_argument('-train_perc', '--train', type=float, default=.6)
    parser.add_argument('-val_perc', '--val', type=float, default=.2)
    parser.add_argument('-test_perc', '--test', type=float, default=.2)
    parser.add_argument('-num_splits', '--splits', type=int, default=5)
    parser.add_argument('-tweet_sim_threshold', '--tsim_th', type=float, default=.7)
    parser.add_argument('-device_id', '--device', type=str, default='0')
    parser.add_argument('-gnn_aggr_fn', '--aggr_fn', type=str, default='mean')
    parser.add_argument('-num_epochs', '--epochs', type=int, default=1000)
    parser.add_argument('-learning_rate', '--lr', type=float, default=1e-2)
    parser.add_argument('-early_stopping_limit', '--early', type=int, default=20)
    parser.add_argument('-check_loss_freq', '--check', type=int, default=1)
    parser.add_argument('-metric_to_optimize', '--val_metric', type=str, default='f1_macro')
    parser.add_argument('-gnn_type', '--gnn', type=str, default='sage')
    parser.add_argument('-gnn_embed_type', '--embed_type', type=str, default='positional_degree')
    parser.add_argument('-latent_dim', '--latent', type=int, default=128)
    parser.add_argument('-dropout', '--dropout', type=float, default=.2)
    parser.add_argument('-min_tweets', '--min_tweets', type=int, default=10)
    parser.add_argument('-most_popular', '--most_pop', type=int, default=5)
    parser.add_argument('-under_sampling', '--under', default=None)
    # Hyperparameters
    parser.add_argument('-focal_gamma', '--focal_gamma', type=float, default=2.0)
    parser.add_argument('-focal_alpha', '--focal_alpha', type=float, default=0.75)
    parser.add_argument('-coral_weight', '--coral_weight', type=float, default=500.0)
    parser.add_argument('-sotm_threshold', '--sotm_threshold', type=float, default=0.3)
    # Multi-level opt flags
    parser.add_argument('--msca', action='store_true', help='Activate Level 1: Multilingual Semantic Centroid Alignment')
    parser.add_argument('--grps', action='store_true', help='Activate Level 2: Graph-Regularized Posterior Smoothing')
    parser.add_argument('--grps_alpha', type=float, default=0.6, help='Label propagation factor (alpha)')
    parser.add_argument('--grps_max_iter', type=int, default=30, help='Maximum label propagation iterations')
    parser.add_argument('--tt_gca', action='store_true', help='Activate Level 3: Test-Time Graph Contrastive Adaptation')
    parser.add_argument('--tt_gca_epochs', type=int, default=5, help='Number of TT-GCA epochs')
    parser.add_argument('--tt_gca_lr', type=float, default=1e-3, help='Learning rate for TT-GCA')

    args = parser.parse_args()

    hyper_parameters = {
        'train_perc': args.train, 'val_perc': args.val, 'test_perc': args.test,
        'aggr_type': args.aggr_fn, 'num_splits': args.splits, 'seed': args.seed,
        'tsim_th': args.tsim_th, 'min_tweets': args.min_tweets, 'most_pop': args.most_pop,
        'input_embed': args.embed_type, 'trace_type': 'all',
        'undersampling': float(args.under) if args.under is not None else None
    }
    train_hyperparameters = {
        'num_epochs': args.epochs, 'learning_rate': args.lr,
        'early_stopping_limit': args.early, 'check_loss_freq': args.check,
        'metric_to_optimize': args.val_metric,
        'input_embed': args.embed_type, 'trace_type': 'all',
        'focal_gamma': args.focal_gamma, 'focal_alpha': args.focal_alpha,
        'coral_weight': args.coral_weight, 'sotm_threshold': args.sotm_threshold,
        'msca': args.msca,
        'grps': args.grps, 'grps_alpha': args.grps_alpha, 'grps_max_iter': args.grps_max_iter,
        'tt_gca': args.tt_gca, 'tt_gca_epochs': args.tt_gca_epochs, 'tt_gca_lr': args.tt_gca_lr
    }
    model_hyperparameters = {'gnn_type': args.gnn, 'latent_dim': args.latent, 'dropout': args.dropout}
    main(args.dataset, train_hyperparameters, model_hyperparameters, hyper_parameters, args.device)
