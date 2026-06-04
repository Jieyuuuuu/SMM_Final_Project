import argparse
import os
import copy
import pickle
import random
import mlflow
import torch
from torch_geometric.nn import GCNConv, SAGEConv
import numpy as np
import pandas as pd
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
DEFAULT_MODEL_HYPERPARAMETERS = {'gnn_type': 'sage', 'latent_dim': 128, 'dropout': 0.2}
ALL_COUNTRIES = ['china', 'iran', 'UAE', 'cuba', 'russia', 'venezuela']


# =============================================================================
# Focal Loss
# =============================================================================
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Weight for positive class
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
# ChannelGNN (GNN processing for a single behavior graph type)
# =============================================================================
class ChannelGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, out_dim, gnn_type='sage', dropout_p=0.2):
        super().__init__()
        if gnn_type == 'gcn':
            conv_block = GCNConv
        elif gnn_type == 'sage':
            conv_block = SAGEConv
        else:
            raise Exception(f"GNN type '{gnn_type}' not supported in ChannelGNN")
        self.conv1 = conv_block(num_node_features, hidden_dim)
        self.conv2 = conv_block(hidden_dim, out_dim)
        self.activation_fn = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


# =============================================================================
# NodeLevelAttentionFusion (Dynamically weights channels per-node)
# =============================================================================
class NodeLevelAttentionFusion(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, z_list):
        z_stack = torch.stack(z_list, dim=1)  # [N, 5, hidden_dim]
        scores = self.attn_proj(z_stack)      # [N, 5, 1]
        weights = torch.softmax(scores, dim=1) # [N, 5, 1]
        z_fused = torch.sum(weights * z_stack, dim=1) # [N, hidden_dim]
        return z_fused, weights.squeeze(-1)


# =============================================================================
# GNN_CrossAttention_AMC Model Definition
# =============================================================================
class GNN_CrossAttention_AMC(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_textual_features,
                 num_structural_features, activation_fn=torch.nn.ReLU(), dropout_p=0.2,
                 gnn_type='sage'):
        super().__init__()
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

        self.gnn_coRT = ChannelGNN(hidden_dim * 2, hidden_dim, hidden_dim, gnn_type, dropout_p)
        self.gnn_coURL = ChannelGNN(hidden_dim * 2, hidden_dim, hidden_dim, gnn_type, dropout_p)
        self.gnn_hashSeq = ChannelGNN(hidden_dim * 2, hidden_dim, hidden_dim, gnn_type, dropout_p)
        self.gnn_fastRT = ChannelGNN(hidden_dim * 2, hidden_dim, hidden_dim, gnn_type, dropout_p)
        self.gnn_tweetSim = ChannelGNN(hidden_dim * 2, hidden_dim, hidden_dim, gnn_type, dropout_p)

        self.fusion = NodeLevelAttentionFusion(hidden_dim)

        self.classifier = torch.nn.Linear(hidden_dim, 1)
        self.output_fn = torch.nn.LogSigmoid()

    def get_text_projection(self, text_node_features, struct_node_features):
        text_projection = (
            self.text_projector(text_node_features)
            * self.cross_attention_to_text(struct_node_features)
        )
        return text_projection

    def forward(self, text_node_features, struct_node_features, edge_indices,
                 return_text_features=False, return_attention=False):
        struct_projection = (
            self.struct_projector(struct_node_features)
            * self.cross_attention_to_struct(text_node_features)
        )
        text_projection = (
            self.text_projector(text_node_features)
            * self.cross_attention_to_text(struct_node_features)
        )

        multimodal_node_features = self.joint_projector(
            torch.concat([struct_projection, text_projection], dim=-1)
        )

        z_coRT = self.gnn_coRT(multimodal_node_features, edge_indices['coRT'])
        z_coURL = self.gnn_coURL(multimodal_node_features, edge_indices['coURL'])
        z_hashSeq = self.gnn_hashSeq(multimodal_node_features, edge_indices['hashSeq'])
        z_fastRT = self.gnn_fastRT(multimodal_node_features, edge_indices['fastRT'])
        z_tweetSim = self.gnn_tweetSim(multimodal_node_features, edge_indices['tweetSim'])

        fused, weights = self.fusion([z_coRT, z_coURL, z_hashSeq, z_fastRT, z_tweetSim])

        out = torch.exp(self.output_fn(self.classifier(fused)))

        res = [out]
        if return_text_features:
            res.append(text_projection)
        if return_attention:
            res.append(weights)

        if len(res) == 1:
            return out
        return tuple(res)


# =============================================================================
# Multi-channel data loader preserving stable global ID mapping
# =============================================================================
def read_all_data_amc(device_id, dataset_name, hyper_params, train_hyperparams, model_hyperparams):
    device, base_dir, interim_data_dir, data_dir = setup_env(device_id, dataset_name, hyper_params)
    print(data_dir)
    datasets = create_data_loader(data_dir, hyper_params['tsim_th'],
                                  hyper_params['train_perc'], hyper_params['undersampling'])
    datasets = move_data_to_device(datasets, device)

    global_graph = datasets['graph'].copy()
    _, global_network = handle_isolated_nodes(global_graph)
    global_nodes = list(global_network.nodes())

    subnets = ['coRT', 'coURL', 'hashSeq', 'fastRT', 'tweetSim']
    edge_indices = {}
    for subnet_name in subnets:
        fname = f'edge_index_{subnet_name}.th'
        if (data_dir / fname).exists():
            edge_idx = torch.load(data_dir / fname, map_location=device)
        else:
            subnet_graph = datasets[subnet_name].copy()
            subnet_graph.add_nodes_from(global_nodes)
            _, subnet_network = handle_isolated_nodes(subnet_graph)
            edge_idx = get_edge_index(subnet_network, data_dir, type=f'_{subnet_name}')
        edge_indices[subnet_name] = edge_idx.to(device)

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
                                                          'num_nodes': global_network.number_of_nodes(),
                                                          'graph': global_network, 'device': device,
                                                          'dataset_name': dataset_name, 'base_dir': base_dir,
                                                          'num_cores': 8,
                                                          'aggr_type': hyper_params['aggr_type']})
    struct_node_features = struct_node_features.to(device)

    return device, base_dir, interim_data_dir, data_dir, datasets, edge_indices, global_network, node_features, struct_node_features


def create_model(model_hyperparams):
    return GNN_CrossAttention_AMC(
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


def main(dataset_name, train_hyperparams, model_hyperparams, hyper_params, device_id):
    if model_hyperparams is None:
        model_hyperparams = DEFAULT_MODEL_HYPERPARAMETERS
    if train_hyperparams is None:
        train_hyperparams = DEFAULT_TRAIN_HYPERPARAMETERS
    if hyper_params is None:
        hyper_params = DEFAULT_HYPERPARAMETERS

    set_seed(hyper_params['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    # 1. Load target domain data
    device, base_dir, interim_data_dir, data_dir, datasets, edge_indices, network, \
        node_features, struct_node_features = read_all_data_amc(
            device_id, dataset_name, hyper_params, train_hyperparams, model_hyperparams)

    model_hyperparams['num_textual_features'] = node_features.shape[1]
    model_hyperparams['num_structural_features'] = struct_node_features.shape[1]

    # Calculate target subnetwork edge distribution
    subnets = ['coRT', 'coURL', 'hashSeq', 'fastRT', 'tweetSim']
    target_edges = []
    for subnet in subnets:
        target_edges.append(datasets[subnet].number_of_edges())
    target_edges = np.array(target_edges, dtype=np.float32)
    target_dist = target_edges / (target_edges.sum() + 1e-9)

    print(f"\n[Target Graph Structure Distribution ({dataset_name})]:")
    for s, e, p in zip(subnets, target_edges, target_dist):
        print(f"  - {s}: {int(e)} edges ({p*100:.2f}%)")

    # 2. Preload source country datasets and extract similarity
    other_countries = [c for c in ALL_COUNTRIES if c != dataset_name]
    countries_data = {}
    countries_numExamples = {}
    for country in other_countries:
        c_device, _, _, _, c_datasets, c_edge_indices, _, c_node_features, c_struct_node_features = read_all_data_amc(
            device_id, country, hyper_params, train_hyperparams, model_hyperparams)
        
        # Calculate source subnetwork edges distribution
        source_edges = []
        for subnet in subnets:
            source_edges.append(c_datasets[subnet].number_of_edges())
        source_edges = np.array(source_edges, dtype=np.float32)
        source_dist = source_edges / (source_edges.sum() + 1e-9)
        
        # Cosine similarity on 5-subnet distribution
        dot_val = np.dot(target_dist, source_dist)
        norm_t = np.linalg.norm(target_dist)
        norm_s = np.linalg.norm(source_dist)
        similarity = dot_val / (norm_t * norm_s + 1e-9)
        
        countries_data[country] = {
            'datasets': c_datasets, 'edge_indices': c_edge_indices,
            'node_features': c_node_features, 'struct_node_features': c_struct_node_features,
            'similarity': float(similarity)
        }
        countries_numExamples[country] = c_struct_node_features.shape[0]

    # Normalize similarities using Softmax with temperature=0.1
    sims = [countries_data[c]['similarity'] for c in countries_data]
    sim_weights = np.exp(np.array(sims) / 0.1)
    sim_weights = sim_weights / sim_weights.sum() * len(countries_data)
    
    print("\n[Adaptive Source-Selection Weights (AMC-v2)]:")
    for idx, country in enumerate(other_countries):
        countries_data[country]['weight'] = float(sim_weights[idx])
        print(f"  - Source '{country}': similarity={countries_data[country]['similarity']:.4f}, normalized loss weight={sim_weights[idx]:.4f}")

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

    # Dynamic Loss Configuration
    loss_type = train_hyperparams.get('loss_type', 'bce')
    if loss_type == 'focal':
        focal_gamma = train_hyperparams.get('focal_gamma', 2.0)
        focal_alpha = train_hyperparams.get('focal_alpha', 0.75)
        loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        print(f"[Loss Config] Using Focal Loss: gamma={focal_gamma}, alpha={focal_alpha}")
    else:
        loss_fn = torch.nn.BCELoss()
        print("[Loss Config] Using standard BCELoss")

    coral_weight = train_hyperparams.get('coral_weight', 50.0)
    print(f"[DFA Config] CORAL covariance loss weight = {coral_weight}")

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
            text_features_dict = {}

            # 1. Forward pass on source countries (Weighted by structural similarity)
            for country in countries_data:
                train_mask = stratified_random_boolean_tensor(
                    countries_numExamples[country],
                    batch_size=128, device=device,
                    labels=countries_data[country]['datasets']['labels']
                )
                pred, text_feats = model(
                    countries_data[country]['node_features'],
                    countries_data[country]['struct_node_features'],
                    countries_data[country]['edge_indices'],
                    return_text_features=True
                )
                text_features_dict[country] = text_feats[train_mask]

                task_loss = loss_fn(
                    pred.flatten()[train_mask],
                    countries_data[country]['datasets']['labels'][train_mask]
                )
                
                # Apply similarity weight
                weighted_task_loss = task_loss * countries_data[country]['weight']
                loss += weighted_task_loss
                task_loss_sum += weighted_task_loss.item()

            # 2. Project target country text features
            target_size = node_features.shape[0]
            target_idx = torch.randperm(target_size)[:128].to(device)
            target_text_feats = model.get_text_projection(
                node_features[target_idx],
                struct_node_features[target_idx]
            )

            # 3. Compute target-aligned CORAL loss (Weighted by structural similarity)
            coral_val = 0
            for country in countries_data:
                weighted_coral = coral_loss(text_features_dict[country], target_text_feats) * countries_data[country]['weight']
                coral_val += weighted_coral
            coral_val = coral_val / len(countries_data)
            loss += coral_val * coral_weight

            loss.backward()
            optimizer.step()
            train_logger.train_update(run_id, 'supervised', loss.item())

            if epoch % train_hyperparams["check_loss_freq"] == 0:
                model.eval()
                with torch.no_grad():
                    pred = model(node_features, struct_node_features, edge_indices).detach().cpu().numpy().flatten()
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

        # Load best snapshot for evaluation
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            pred, final_weights = model(node_features, struct_node_features, edge_indices, return_attention=True)
            pred = pred.detach().cpu().numpy().flatten()
            mean_weights = final_weights.mean(dim=0).cpu().numpy()
            print(f"\n[Split {run_id}] Mean Channel Attention: "
                  f"coRT={mean_weights[0]:.4f}, coURL={mean_weights[1]:.4f}, "
                  f"hashSeq={mean_weights[2]:.4f}, fastRT={mean_weights[3]:.4f}, "
                  f"tweetSim={mean_weights[4]:.4f}")

        val_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['val'])
        for metric_name in val_metrics:
            val_logger.update(metric_name, run_id, val_metrics[metric_name])

        test_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['test'])
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Adaptive Multi-Channel GFM (AMC-GFM) v2 model")
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
    # AMC parameters
    parser.add_argument('-loss_type', '--loss_type', type=str, default='bce', choices=['bce', 'focal'])
    parser.add_argument('-focal_gamma', '--focal_gamma', type=float, default=2.0)
    parser.add_argument('-focal_alpha', '--focal_alpha', type=float, default=0.75)
    parser.add_argument('-coral_weight', '--coral_weight', type=float, default=50.0)

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
        'loss_type': args.loss_type,
        'focal_gamma': args.focal_gamma, 'focal_alpha': args.focal_alpha,
        'coral_weight': args.coral_weight
    }
    model_hyperparameters = {'gnn_type': args.gnn, 'latent_dim': args.latent, 'dropout': args.dropout}
    main(args.dataset, train_hyperparameters, model_hyperparameters, hyper_parameters, args.device)
