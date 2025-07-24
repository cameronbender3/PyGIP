import torch
import numpy as np
import random

def true_bit_flip(tensor, index=None, bit=0):
    """
    Flips a single bit (bit index) of a float32 tensor element at a specified index.
    bit=0: least significant bit (LSB)
    """
    # Copy as numpy array for bit manipulation
    a = tensor.detach().cpu().numpy().copy()
    flat = a.ravel()
    if index is None:
        index = np.random.randint(0, flat.size)
    old_val = flat[index]
    # Get float as int
    int_view = np.frombuffer(flat[index].tobytes(), dtype=np.uint32)[0]
    # Flip the bit
    int_view ^= (1 << bit)
    # Back to float
    new_val = np.frombuffer(np.uint32(int_view).tobytes(), dtype=np.float32)[0]
    flat[index] = new_val
    # Restore to tensor
    a = flat.reshape(a.shape)
    tensor.data = torch.from_numpy(a).to(tensor.device)
    return old_val, new_val, index

class BitFlipAttack:
    def __init__(self, model, attack_type='random', bit=0):
        """
        attack_type: 'random' (any param), 'BFA-F' (first layer), 'BFA-L' (last layer)
        bit: which bit to flip (0 = LSB, 23 = start of mantissa, 30 = exponent, etc.)
        """
        self.model = model
        self.attack_type = attack_type
        self.bit = bit

    def _get_target_params(self):
        params = [p for p in self.model.parameters() if p.requires_grad and p.numel() > 0]
        if self.attack_type == 'random':
            return params
        elif self.attack_type == 'BFA-F':  # First layer only
            return [params[0]]  # Assumes first param is first layer (usually weights)
        elif self.attack_type == 'BFA-L':  # Last layer only
            return [params[-1]]  # Assumes last param is last layer (usually bias or weights)
        else:
            raise ValueError(f"Unknown attack_type {self.attack_type}")

    def apply(self):
        """
        Apply the bit-flip attack in-place.
        Returns: (layer_idx, param_idx, old_val, new_val)
        """
        params = self._get_target_params()
        with torch.no_grad():
            layer_idx = random.randrange(len(params))
            param = params[layer_idx]
            idx = random.randrange(param.numel())
            old_val, new_val, actual_idx = true_bit_flip(param, index=idx, bit=self.bit)
        return {
            'layer': layer_idx,
            'param_idx': actual_idx,
            'old_val': old_val,
            'new_val': new_val,
            'bit': self.bit,
            'attack_type': self.attack_type
        }

