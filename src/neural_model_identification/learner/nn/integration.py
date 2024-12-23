import torch
from torch import nn
from neural_model_identification.parameters.train_params import TrainParams

class IntegrationLayer(nn.Module):  # type: ignore[misc]
    """
    A integration layer allowing for a gate driven skip connection (residual).

    Args:
        dim (int): The dimensionality of the input and output.

    Attributes:
        transform (nn.Linear): Linear layer to compute the transform gate.
    """

    def __init__(self, dim: int) -> None:
        super(IntegrationLayer, self).__init__()
        self.is_dt_initialized = False

    def forward(self, x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the integration layer.

        Args:
            x (torch.Tensor): The original input tensor.
            dx (torch.Tensor): The derivative tensor.

        Returns:
            torch.Tensor: The output tensor after applying the integration layer.
        """
        if not self.is_dt_initialized:
            self.dt = torch.zeros_like(x) + TrainParams.dt
            self.is_dt_initialized = True
        return x + dx * self.dt
