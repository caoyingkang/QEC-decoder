"""Base class for PyTorch iterative decoder models."""
import torch
import torch.nn as nn


class DecoderModel(nn.Module):
    """
    Base class for PyTorch iterative decoder models.

    This is the nn.Module that implements the neural network architecture. Every decoder model
    should inherit from this class, and is assumed to have an iterative (recurrent) nature: the 
    forward pass is made of multiple iterations, with each iteration producing an LLR value for 
    each variable node.
    """

    def __init__(self, num_chks: int, num_vars: int, num_iters: int):
        """Initialize the PyTorch iterative decoder model."""
        super().__init__()
        self.num_chks = num_chks
        self.num_vars = num_vars
        if num_iters < 1:
            raise ValueError(f"num_iters must be at least 1, but got {num_iters}")
        self.num_iters = num_iters

    def forward(self, syndromes: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
            syndromes : torch.Tensor
                Syndrome bits, shape=(batch_size, num_chks), int ∈ {0,1}

        Returns
        -------
            llrs : torch.Tensor
                LLR outputs at all iterations, shape=(num_iters, batch_size, num_vars), float
        """
        raise NotImplementedError("Subclasses must implement this method.")
