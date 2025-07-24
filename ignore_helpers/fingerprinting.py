import torch
import torch.nn.functional as F

class TransductiveFingerprintGenerator:
    def __init__(self, model, dataset, candidate_fraction=1.0, random_seed=None, device='cpu'):
        """
        Args:
            model: Trained GNN model (PyTorch, implements forward(graph, features))
            dataset: PyGIP Dataset object with .graph, .features, .labels
            candidate_fraction: float, what fraction of nodes to consider as candidates (default 1.0 = all)
            random_seed: int, seed for reproducibility (optional)
            device: device string (cpu/cuda)
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.candidate_fraction = candidate_fraction
        self.random_seed = random_seed
        self.device = device

    def get_candidate_nodes(self):
        all_nodes = torch.arange(self.dataset.graph.num_nodes())
        if self.candidate_fraction < 1.0:
            num_candidates = int(len(all_nodes) * self.candidate_fraction)
            generator = torch.Generator(device=self.device)
            if self.random_seed is not None:
                generator.manual_seed(self.random_seed)
            idx = torch.randperm(len(all_nodes), generator=generator)[:num_candidates]
            return all_nodes[idx]
        return all_nodes

    def compute_fingerprint_scores_full(self, candidate_nodes):
        """
        Full model knowledge (Transductive-F): uses gradient norms.
        """
        self.model.eval()
        scores = []
        logits = self.model(self.dataset.graph.to(self.device), self.dataset.features.to(self.device))
        for node in candidate_nodes:
            logit = logits[node]
            label = logit.argmax().item()
            loss = F.nll_loss(F.log_softmax(logit.unsqueeze(0), dim=1), torch.tensor([label], device=self.device))
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            # Sum of gradient norms for all parameters
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    grad_norm += (p.grad ** 2).sum().item()
            scores.append(grad_norm)
        return torch.tensor(scores, device=self.device)

    def compute_fingerprint_scores_limited(self, candidate_nodes):
        """
        Limited model knowledge (Transductive-L): uses confidence.
        """
        self.model.eval()
        logits = self.model(self.dataset.graph.to(self.device), self.dataset.features.to(self.device))
        probs = F.softmax(logits, dim=1)
        labels = probs.argmax(dim=1)
        # Score is 1 - confidence of the predicted class (Eq. 6)
        scores = 1.0 - probs[candidate_nodes, labels[candidate_nodes]]
        return scores

    def select_top_fingerprints(self, scores, candidate_nodes, k):
        topk = torch.topk(scores, k)
        return candidate_nodes[topk.indices], topk.values

    def generate_fingerprints(self, k=5, method='full'):
        """
        Args:
            k: Number of fingerprints to generate
            method: 'full' for Transductive-F, 'limited' for Transductive-L
        Returns:
            List of (node_id, label) tuples
        """
        candidate_nodes = self.get_candidate_nodes().to(self.device)
        if method == 'full':
            scores = self.compute_fingerprint_scores_full(candidate_nodes)
        elif method == 'limited':
            scores = self.compute_fingerprint_scores_limited(candidate_nodes)
        else:
            raise ValueError("method must be 'full' or 'limited'")
        fingerprint_nodes, _ = self.select_top_fingerprints(scores, candidate_nodes, k)
        # Use model to get labels for fingerprint nodes
        logits = self.model(self.dataset.graph.to(self.device), self.dataset.features.to(self.device))
        labels = logits.argmax(dim=1)
        fingerprints = [(int(n), int(labels[n])) for n in fingerprint_nodes]
        return fingerprints
