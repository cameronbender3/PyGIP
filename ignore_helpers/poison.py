import copy
import random
import torch

def random_edge_addition_poisoning(dataset, perturb_frac, random_seed=None):
    """
    Returns a new DGLGraph with random edges added.

    Args:
        dataset: Dataset object (with .graph as DGLGraph)
        perturb_frac: Fraction of edges to add (e.g., 0.01 = 1%)
        random_seed: Optional integer for reproducibility

    Returns:
        poisoned_graph: DGLGraph (deepcopy of original with new edges)
    """
    import dgl

    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    orig_graph = dataset.graph
    poisoned_graph = copy.deepcopy(orig_graph)
    num_nodes = poisoned_graph.num_nodes()
    num_edges_to_add = int(perturb_frac * orig_graph.num_edges())

    # Build set of all existing edges (as (u,v) pairs)
    existing_edges = set(zip(
        orig_graph.edges()[0].tolist(),
        orig_graph.edges()[1].tolist()
    ))

    # Generate candidate node pairs (exclude self-loops and duplicates)
    candidate_pairs = [
        (i, j)
        for i in range(num_nodes)
        for j in range(num_nodes)
        if i != j and (i, j) not in existing_edges
    ]

    if len(candidate_pairs) < num_edges_to_add:
        raise ValueError("Perturbation budget too large: not enough candidate edges.")

    new_edges = random.sample(candidate_pairs, num_edges_to_add)
    src, dst = zip(*new_edges)
    poisoned_graph.add_edges(src, dst)

    return poisoned_graph

def retrain_poisoned_model(dataset, poisoned_graph, defense_class, device='cpu'):
    """
    Retrain target GCN using the poisoned graph structure.

    Args:
        dataset: Original Dataset object (provides features, labels, masks)
        poisoned_graph: DGLGraph (with new random edges added)
        defense_class: The defense class to use for model training (e.g., QueryBasedVerificationDefense)
        device: 'cpu' or 'cuda'

    Returns:
        model: Trained GCN model
    """
    # Create a shallow copy and swap in the poisoned graph
    dataset_poisoned = copy.copy(dataset)
    dataset_poisoned.graph = poisoned_graph

    # If Dataset is more complex, you may want to rebuild it from scratch
    defense = defense_class(dataset=dataset_poisoned, attack_node_fraction=0.1)
    model = defense._train_target_model()
    return model

def evaluate_accuracy(model, dataset, device='cpu'):
    """
    Evaluates test accuracy of the given model on the dataset.

    Args:
        model: Trained GCN model
        dataset: Dataset object (provides features, labels, test_mask, graph)
        device: 'cpu' or 'cuda'

    Returns:
        accuracy: float (test accuracy, 0-1)
    """
    model.eval()
    features = dataset.features.to(device)
    labels = dataset.labels.to(device)
    test_mask = dataset.test_mask

    with torch.no_grad():
        logits = model(dataset.graph.to(device), features)
        pred = logits.argmax(dim=1)
        correct = (pred[test_mask] == labels[test_mask]).float()
        accuracy = correct.sum().item() / test_mask.sum().item()
    return accuracy

# (Optional) If you plan to support more attack types, you could add:
# def mettack_poisoning(...): ...
