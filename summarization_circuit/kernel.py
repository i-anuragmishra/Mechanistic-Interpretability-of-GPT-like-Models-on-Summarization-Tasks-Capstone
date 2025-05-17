import torch
from typing import List, Dict

class SummarizationCircuitKernel:
    """Attach hooks to specified GPT-2 layers to capture activations."""

    def __init__(self, model: torch.nn.Module, layers: List[int]):
        self.model = model
        self.layers = layers
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.activations: Dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_id: int):
        def hook(module, input, output):
            self.activations[layer_id] = output.detach().cpu()
        return hook

    def attach(self) -> None:
        """Attach forward hooks to the specified layers."""
        self.detach()
        for layer_id in self.layers:
            layer = self.model.transformer.h[layer_id]
            handle = layer.register_forward_hook(self._make_hook(layer_id))
            self.handles.append(handle)

    def detach(self) -> None:
        """Remove all attached hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Return the collected activations."""
        return self.activations
