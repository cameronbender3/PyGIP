import torch
from datasets import Cora
from models.defense import QueryBasedVerificationDefense
from ignore_helpers import fingerprinting, attack_sim, poison
import torch.nn.functional as F
import copy  # Python's deepcopy


def evaluate_fingerprints(model, dataset, fingerprints, device='cpu'):
    model.eval()
    logits = model(dataset.graph.to(device), dataset.features.to(device))
    pred_labels = logits.argmax(dim=1).cpu()
    changed = []
    for node_id, clean_label in fingerprints:
        if pred_labels[node_id] != clean_label:
            changed.append((node_id, clean_label, int(pred_labels[node_id])))
    return changed



def main_poisoning(num_trials=50, poison_frac=0.01):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Cora()

    print("Training clean target model...")
    defense = QueryBasedVerificationDefense(dataset=dataset, attack_node_fraction=0.1)
    base_model = defense._train_target_model()

    # Accuracy before any poisoning
    clean_acc = poison.evaluate_accuracy(base_model, dataset, device=device)
    print(f"Clean model test accuracy: {clean_acc:.4f}")

    # # If/when you want to test fingerprints:
    # generator = fingerprinting.TransductiveFingerprintGenerator(base_model, dataset, candidate_fraction=1.0, random_seed=42, device=device)
    # fingerprints_full = generator.generate_fingerprints(k=k, method='full')
    # fingerprints_limited = generator.generate_fingerprints(k=k, method='limited')

    poisoned_accuracies = []

    for trial in range(num_trials):
        poisoned_graph = poison.random_edge_addition_poisoning(
            dataset=dataset,
            perturb_frac=poison_frac,
            random_seed=trial
        )

        # Make a dataset copy with the poisoned graph
        dataset_poisoned = copy.copy(dataset)
        dataset_poisoned.graph = poisoned_graph

        poisoned_model = poison.retrain_poisoned_model(
            dataset=dataset_poisoned,  # Use the poisoned dataset
            poisoned_graph=poisoned_graph,
            defense_class=QueryBasedVerificationDefense,
            device=device
        )

        # Evaluate on the poisoned dataset
        poisoned_acc = poison.evaluate_accuracy(poisoned_model, dataset_poisoned, device=device)
        poisoned_accuracies.append(poisoned_acc)

        if trial == 0:
            print(f"Example poisoned test accuracy: {poisoned_acc:.4f}")

        if (trial + 1) % 10 == 0:
            print(f"Poison Trial {trial+1}/{num_trials}")


        # # Evaluate fingerprints (disabled for now)
        # changed_full = evaluate_fingerprints(poisoned_model, dataset, fingerprints_full, device=device)
        # changed_limited = evaluate_fingerprints(poisoned_model, dataset, fingerprints_limited, device=device)
        # if changed_full:
        #     detected_full += 1
        # if changed_limited:
        #     detected_limited += 1

    # Final stats
    avg_poisoned_acc = sum(poisoned_accuracies) / len(poisoned_accuracies)
    print("\n==== Poisoning Results ====")
    print(f"Average clean model test accuracy: {clean_acc:.4f}")
    print(f"Average poisoned model test accuracy: {avg_poisoned_acc:.4f}")
    print(f"Average accuracy drop: {clean_acc - avg_poisoned_acc:.4f}")
    # print("\n==== Poisoning Detection Rate Results ====")
    # print(f"Transductive-F (full knowledge) DR: {detected_full/num_trials:.3f}")
    # print(f"Transductive-L (limited knowledge) DR: {detected_limited/num_trials:.3f}")



def main(num_trials=100, k=5, attack_type='random', bit=0):
    """
    :param num_trials: Number of attack rounds
    :param k: Number of fingerprints
    :param attack_type: 'random', 'BFA-F', 'BFA-L'
    :param bit: Which bit to flip (0 = LSB, 23 = mantissa, 30 = exponent, etc.)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Cora()
    print("Training target model (baseline)...")
    defense = QueryBasedVerificationDefense(dataset=dataset, attack_node_fraction=0.1)
    base_model = defense._train_target_model()  # Train ONCE

    generator = fingerprinting.TransductiveFingerprintGenerator(base_model, dataset, candidate_fraction=1.0, random_seed=42, device=device)
    fingerprints_full = generator.generate_fingerprints(k=k, method='full')
    fingerprints_limited = generator.generate_fingerprints(k=k, method='limited')

    detected_full = 0
    detected_limited = 0

    for trial in range(num_trials):
        attacked_model = copy.deepcopy(base_model)
        attack = attack_sim.BitFlipAttack(attacked_model, attack_type=attack_type, bit=bit)
        attack_result = attack.apply()
        if trial < 5:
            def float_to_bits(val):
                import struct
                [d] = struct.unpack(">L", struct.pack(">f", val))
                return f"{d:032b}"
            old_val = attack_result['old_val']
            new_val = attack_result['new_val']
            bit_idx = attack_result['bit']
            print(f"Trial {trial+1} bit-flip details:")
            print(f"  Flipped bit: {bit_idx}")
            print(f"  Old value: {old_val} ({float_to_bits(old_val)})")
            print(f"  New value: {new_val} ({float_to_bits(new_val)})")
        
        changed_full = evaluate_fingerprints(attacked_model, dataset, fingerprints_full, device=device)
        changed_limited = evaluate_fingerprints(attacked_model, dataset, fingerprints_limited, device=device)
        if changed_full:
            detected_full += 1
        if changed_limited:
            detected_limited += 1
        if (trial + 1) % 10 == 0:
            print(f"Trial {trial+1}/{num_trials}: F={detected_full} L={detected_limited}")



    print("\n==== Detection Rate Results ====")
    print(f"Transductive-F (full knowledge) DR: {detected_full/num_trials:.3f}")
    print(f"Transductive-L (limited knowledge) DR: {detected_limited/num_trials:.3f}")

if __name__ == '__main__':
    main_poisoning(num_trials=50, poison_frac=0.01)

