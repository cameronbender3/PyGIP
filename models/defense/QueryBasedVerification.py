from .base import BaseDefense
import torch
import torch.nn.functional as F
from torch.optim import Adam
from models.nn import GCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QueryBasedVerificationDefense(BaseDefense):
    def __init__(self, dataset, attack_node_fraction, model_path=None):
        super().__init__(dataset, attack_node_fraction)
        self.model_path = model_path

    
    def defend(self, *args, **kwargs):
        """
        Main defense workflow for query-based verification.
        For now, this is just a stub for testing and must be filled in later.
        """
        print("defend() method called. Not implemented yet.")
        model = self._load_model(self.model_path) if self.model_path else self._train_target_model()
        # ... fingerprinting/verification logic ...

    def _train_target_model(self):
        """
        Trains target GCN model according to protocol in
        Wu et al. (2023), Section 6.1 for graph node classification.

        Returns
        -------
        model : torch.nn.Module
            The trained GCN model.
        """
        model = GCN(
        feature_number=self.dataset.feature_number,
        label_number=self.dataset.label_number
        ).to(device)
        print(f"Training target model on device: {device} ...")

        optimizer = Adam(model.parameters(), lr=0.02)
        loss_fn = torch.nn.NLLLoss()

        features = self.dataset.features.to(device)
        labels = self.dataset.labels.to(device)
        train_mask = self.dataset.train_mask.to(device)
        # Use test_mask for validation monitoring if val_mask is not available
        val_mask = getattr(self.dataset, "val_mask", None)
        if val_mask is None:
            val_mask = self.dataset.test_mask
        val_mask = val_mask.to(device)

        for epoch in range(200):
            model.train()
            logits = model(self.dataset.graph.to(device), features)
            log_probs = F.log_softmax(logits, dim=1)
            loss = loss_fn(log_probs[train_mask], labels[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(self.dataset.graph.to(device), features)
                    val_log_probs = F.log_softmax(val_logits, dim=1)
                    val_pred = val_log_probs[val_mask].max(1)[1]
                    val_acc = (val_pred == labels[val_mask]).float().mean().item()
                    print(f"Epoch {epoch+1}: Loss={loss.item():.4f} | Val Acc={val_acc:.4f}")

        return model

    def _load_model(self, model_path):
        # Load model weights if path is given
        model = GCN(
            in_feats=self.dataset.feature_number, 
            hidden_feats=16, 
            out_feats=self.dataset.label_number
        )
        model.load_state_dict(torch.load(model_path))
        return model

    # ... _train_defense_model(), _train_surrogate_model() as needed ...
