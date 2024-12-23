import torch
from torch import nn


class HighwayLayer(nn.Module):  # type: ignore[misc]
    """
    A highway layer allowing for a gate driven skip connection (residual).

    Args:
        dim (int): The dimensionality of the input and output.

    Attributes:
        transform (nn.Linear): Linear layer to compute the transform gate.
    """

    def __init__(self, dim: int) -> None:
        super(HighwayLayer, self).__init__()
        self.transform = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, processed_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the highway layer.

        Args:
            x (torch.Tensor): The original input tensor.
            processed_x (torch.Tensor): The processed input tensor
                                        (e.g., output of another layer).

        Returns:
            torch.Tensor: The output tensor after applying the highway layer.
        """
        t_x = torch.sigmoid(self.transform(x))

        # Combine processed input and original input
        # based on the transform gate
        return processed_x * t_x + x * (1.0 - t_x)
