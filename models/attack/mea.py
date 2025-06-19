import os
import random
import time
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from tqdm import tqdm

from models.attack.base import BaseAttack
from models.nn import GCN, ShadowNet, AttackNet
from utils.metrics import GraphNeuralNetworkMetric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

class ModelExtractionAttack(BaseAttack):
    def __init__(self, dataset, attack_node_fraction, model_path=None, alpha=0.8):

        # Move tensors to device before calling super().__init__
        self.alpha = alpha
        self.graph = dataset.graph.to(device)
        self.features = dataset.features.to(device)
        self.labels = dataset.labels.to(device)
        self.train_mask = dataset.train_mask.to(device)
        self.test_mask = dataset.test_mask.to(device)
        
        # Store original references
        dataset.graph = self.graph
        dataset.features = self.features
        dataset.labels = self.labels
        dataset.train_mask = self.train_mask
        dataset.test_mask = self.test_mask

        super().__init__(dataset, attack_node_fraction, model_path)

    def _train_target_model(self):
        """
        Train the target model (GCN) on the original graph.
        """
        # Initialize GNN model
        self.net1 = GCN(self.feature_number, self.label_number).to(device)
        optimizer = torch.optim.Adam(self.net1.parameters(), lr=0.01, weight_decay=5e-4)
        
        # Training loop
        for epoch in range(200):
            self.net1.train()
            
            # Forward pass
            logits = self.net1(self.graph, self.features)
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[self.train_mask], self.labels[self.train_mask])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation (optional)
            if epoch % 20 == 0:
                self.net1.eval()
                with torch.no_grad():
                    logits_val = self.net1(self.graph, self.features)
                    logp_val = F.log_softmax(logits_val, dim=1)
                    pred = logp_val.argmax(dim=1)
                    acc_val = (pred[self.test_mask] == self.labels[self.test_mask]).float().mean()
                    # You could print validation accuracy here
                
        return self.net1

    def _load_model(self, model_path):
        """
        Load a pre-trained model from a file.
        """
        self.net1 = GCN(self.feature_number, self.label_number).to(device)
        self.net1.load_state_dict(torch.load(model_path))
        self.net1.eval()
        return self.net1

    def attack(self):
        raise NotImplementedError


class ModelExtractionAttack0(ModelExtractionAttack):
    def __init__(self, dataset, attack_node_fraction, model_path=None, alpha=0.8):
        super().__init__(dataset, attack_node_fraction, model_path)
        self.alpha = alpha

    def get_nonzero_indices(self, matrix_row):
        return np.where(matrix_row != 0)[0]

    def attack(self):
        """
        Main attack procedure.

        1. Samples a subset of nodes (`sub_graph_node_index`) for querying.
        2. Synthesizes features for neighboring nodes and their neighbors.
        3. Builds a sub-graph, trains a new GCN on it, and evaluates
           fidelity & accuracy w.r.t. the target model.
        """
        try:
            torch.cuda.empty_cache()
            g = self.graph.clone().to(device)
            g_matrix = g.adjacency_matrix().to_dense().cpu().numpy()
            del g

            sub_graph_node_index = np.random.choice(
                self.node_number, self.attack_node_number, replace=False).tolist()

            batch_size = 32
            features_query = self.features.clone()

            syn_nodes = []
            for node_index in sub_graph_node_index:
                one_step_node_index = self.get_nonzero_indices(g_matrix[node_index]).tolist()
                syn_nodes.extend(one_step_node_index)

                for first_order_node_index in one_step_node_index:
                    two_step_node_index = self.get_nonzero_indices(g_matrix[first_order_node_index]).tolist()
                    syn_nodes.extend(two_step_node_index)

            sub_graph_syn_node_index = list(set(syn_nodes) - set(sub_graph_node_index))
            total_sub_nodes = list(set(sub_graph_syn_node_index + sub_graph_node_index))

            # Process synthetic nodes in batches
            for i in range(0, len(sub_graph_syn_node_index), batch_size):
                batch_indices = sub_graph_syn_node_index[i:i + batch_size]

                for node_index in batch_indices:
                    features_query[node_index] = 0
                    one_step_node_index = self.get_nonzero_indices(g_matrix[node_index]).tolist()
                    one_step_node_index = list(set(one_step_node_index).intersection(set(sub_graph_node_index)))

                    num_one_step = len(one_step_node_index)
                    if num_one_step > 0:
                        for first_order_node_index in one_step_node_index:
                            this_node_degree = len(self.get_nonzero_indices(g_matrix[first_order_node_index]))
                            features_query[node_index] += (
                                    self.features[first_order_node_index] * self.alpha /
                                    torch.sqrt(torch.tensor(num_one_step * this_node_degree, device=device))
                            )

                    two_step_nodes = []
                    for first_order_node_index in one_step_node_index:
                        two_step_nodes.extend(self.get_nonzero_indices(g_matrix[first_order_node_index]).tolist())

                    total_two_step_node_index = list(set(two_step_nodes) - set(one_step_node_index))
                    total_two_step_node_index = list(
                        set(total_two_step_node_index).intersection(set(sub_graph_node_index)))

                    num_two_step = len(total_two_step_node_index)
                    if num_two_step > 0:
                        for second_order_node_index in total_two_step_node_index:
                            this_node_first_step_nodes = self.get_nonzero_indices(
                                g_matrix[second_order_node_index]).tolist()
                            this_node_second_step_nodes = set()

                            for nodes_in_this_node in this_node_first_step_nodes:
                                this_node_second_step_nodes.update(
                                    self.get_nonzero_indices(g_matrix[nodes_in_this_node]).tolist())

                            this_node_second_step_nodes = this_node_second_step_nodes - set(this_node_first_step_nodes)
                            this_node_second_degree = len(this_node_second_step_nodes)

                            if this_node_second_degree > 0:
                                features_query[node_index] += (
                                        self.features[second_order_node_index] * (1 - self.alpha) /
                                        torch.sqrt(torch.tensor(num_two_step * this_node_second_degree, device=device))
                                )

                torch.cuda.empty_cache()

            # Update masks
            for i in range(self.node_number):
                if i in sub_graph_node_index:
                    self.test_mask[i] = 0
                    self.train_mask[i] = 1
                elif i in sub_graph_syn_node_index:
                    self.test_mask[i] = 1
                    self.train_mask[i] = 0
                else:
                    self.test_mask[i] = 1
                    self.train_mask[i] = 0

            # Create subgraph adjacency matrix
            sub_g = np.zeros((len(total_sub_nodes), len(total_sub_nodes)))
            for sub_index in range(len(total_sub_nodes)):
                sub_g[sub_index] = g_matrix[total_sub_nodes[sub_index], total_sub_nodes]

            del g_matrix

            sub_train_mask = self.train_mask[total_sub_nodes]
            sub_features = features_query[total_sub_nodes]
            sub_labels = self.labels[total_sub_nodes]

            # Get query labels
            self.net1.eval()
            with torch.no_grad():
                g = self.graph.to(device)
                logits_query = self.net1(g, features_query)
                _, labels_query = torch.max(logits_query, dim=1)
                sub_labels_query = labels_query[total_sub_nodes]
                del logits_query

            # Create DGL graph
            sub_g = nx.from_numpy_array(sub_g)
            sub_g.remove_edges_from(nx.selfloop_edges(sub_g))
            sub_g.add_edges_from(zip(sub_g.nodes(), sub_g.nodes()))
            sub_g = DGLGraph(sub_g)
            sub_g = sub_g.to(device)

            degs = sub_g.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm[torch.isinf(norm)] = 0
            norm = norm.to(device)
            sub_g.ndata['norm'] = norm.unsqueeze(1)

            # Train extraction model
            net = GCN(self.feature_number, self.label_number).to(device)
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=5e-4)
            best_performance_metrics = GraphNeuralNetworkMetric()

            print("=========Model Extracting==========================")
            for epoch in tqdm(range(200)):
                net.train()
                logits = net(sub_g, sub_features)
                logp = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(logp[sub_train_mask], sub_labels_query[sub_train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    focus_gnn_metrics = GraphNeuralNetworkMetric(
                        0, 0, net, g, self.features, self.test_mask, self.labels, labels_query
                    )
                    focus_gnn_metrics.evaluate()

                    best_performance_metrics.fidelity = max(
                        best_performance_metrics.fidelity, focus_gnn_metrics.fidelity)
                    best_performance_metrics.accuracy = max(
                        best_performance_metrics.accuracy, focus_gnn_metrics.accuracy)

                if epoch % 10 == 0:
                    torch.cuda.empty_cache()

            print("========================Final results:=========================================")
            print(best_performance_metrics)

            self.net2 = net

        except RuntimeError as e:
            print(f"Runtime error: {e}")
            torch.cuda.empty_cache()
            raise

