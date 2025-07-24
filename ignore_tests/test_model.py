import torch
from datasets import Cora
from models.defense import QueryBasedVerificationDefense

def test_train_target_model():
    # Load dataset
    dataset = Cora()  # substitute with your actual Dataset class if different
    print("Dataset loaded.")
    print(f"Features: {dataset.features.shape}, Labels: {dataset.labels.shape}")

    # Initialize defense object
    defense = QueryBasedVerificationDefense(dataset=dataset, attack_node_fraction=0.1)
    
    # Train model
    model = defense._train_target_model()

    # Test model outputs shape
    model.eval()
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logits = model(dataset.graph.to(device), dataset.features.to(device))
        print("Logits shape:", logits.shape)
        # Optionally: check output for a few nodes
        print("First 5 node predictions:", logits[:5].argmax(dim=1).cpu().numpy())

if __name__ == "__main__":
    test_train_target_model()
