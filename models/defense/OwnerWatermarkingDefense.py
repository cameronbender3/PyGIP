from .base import BaseDefense
from models.nn.backbones import GCN_PyG
from integrations.watermarking.trigger import TriggerGenerator, generate_trigger_graph
from integrations.watermarking.train import bi_level_optimization
from integrations.watermarking.metrics import calculate_metrics


class OwnerWatermarkingDefense(BaseDefense):
 
    def __init__(self, dataset, attack_node_fraction=0.3, model_path=None):
        super().__init__(dataset, attack_node_fraction)
        self.model_path = model_path
        self.owner_id = dataset.graph.ndata['feat'].new_tensor([0.1, 0.3, 0.5, 0.7, 0.9])

        in_feats = dataset.graph.ndata['feat'].shape[1]
        num_classes = int(dataset.graph.ndata['label'].max().item()) + 1

        self.generator = TriggerGenerator(in_feats, 64, self.owner_id)
        self.model = GCN_PyG(in_feats, 128, num_classes)


    def defend(self, pyg_graph):
        bi_level_optimization(self.model, self.generator, pyg_graph)
        trigger_data = generate_trigger_graph(pyg_graph, self.generator, self.model)
        metrics = calculate_metrics(self.model, trigger_data)
        return metrics


    def _load_model(self):
        if self.model_path:
            self.model.load_state_dict(torch.load(self.model_path))

    def _train_target_model(self):
        # optional if you split training from watermarking
        pass

    def _train_defense_model(self):
        return self.model

    def _train_surrogate_model(self):
        pass
