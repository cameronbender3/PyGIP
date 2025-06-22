import torch
from torch_geometric.datasets import Planetoid
from sklearn.metrics import precision_score, recall_score, f1_score
from torch_geometric.transforms import NormalizeFeatures



def load_dataset(name):
    return Planetoid(root=f'/tmp/{name}', name=name, transform=NormalizeFeatures())

def calculate_metrics(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        true = data.y

        # Handle both original and watermarked data cases
        if hasattr(data, 'original_test_mask'):
            test_mask = data.original_test_mask
            if test_mask.size(0) < pred.size(0):
                pad_len = pred.size(0) - test_mask.size(0)
                test_mask = torch.cat([test_mask, torch.zeros(pad_len, dtype=torch.bool, device=test_mask.device)])
        else:
            test_mask = data.test_mask

        metrics = {
            'accuracy': (pred[test_mask] == true[test_mask]).float().mean().item(),
            'precision': precision_score(true[test_mask].cpu(), pred[test_mask].cpu(), average='macro'),
            'recall': recall_score(true[test_mask].cpu(), pred[test_mask].cpu(), average='macro'),
            'f1': f1_score(true[test_mask].cpu(), pred[test_mask].cpu(), average='macro'),
            'wm_accuracy': None
        }

        if hasattr(data, 'trigger_nodes'):
            wm_mask = torch.zeros(data.x.size(0), dtype=torch.bool, device=data.x.device)  # ✅ fixed here
            wm_mask[data.trigger_nodes] = True
            wm_pred = pred[wm_mask]
            wm_true = true[wm_mask]
            metrics['wm_accuracy'] = (wm_pred == wm_true).float().mean().item() * 100

        return metrics
