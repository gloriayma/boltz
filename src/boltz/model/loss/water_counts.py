import torch
from torch import nn


def water_counts_loss(
    pred_water_counts,
    feats,
    multiplicity=1,
    mask_loss=None,
    loss_type="mse",
    delta=1.0,
):
    """Compute loss for water fractional credit prediction.
    
    This is a standalone loss function for training ONLY the water counts head,
    without any confidence-related losses.
    
    Parameters
    ----------
    pred_water_counts : torch.Tensor
        Predicted water counts, shape [B, N] (token-level)
    feats : dict
        Features dictionary containing ground truth water_counts and masks
    multiplicity : int
        Number of samples per input
    mask_loss : torch.Tensor, optional
        Optional mask for loss computation
    loss_type : str, optional
        Type of loss to use: "mse" or "huber", by default "mse"
    delta : float, optional
        Delta parameter for Huber loss (threshold for switching between MSE and L1),
        by default 1.0. Only used when loss_type="huber"
        
    Returns
    -------
    dict
        Dictionary with 'loss' and 'loss_breakdown' keys
    """
    with torch.autocast("cuda", enabled=False):
        # Get ground truth water counts
        true_water_counts = feats["water_counts"].float()
        true_water_counts = true_water_counts.repeat_interleave(multiplicity, 0)
        
        # Token-level only: use token_pad_mask
        pad_mask = feats["token_pad_mask"].repeat_interleave(multiplicity, 0).float()
        
        # Ensure shapes match
        if pred_water_counts.shape != true_water_counts.shape:
            # Handle case where pred might have extra dimension
            if len(pred_water_counts.shape) > len(true_water_counts.shape):
                pred_water_counts = pred_water_counts.squeeze(-1)
        
        # Compute errors
        errors = pred_water_counts - true_water_counts
        
        # Compute loss based on loss_type
        if loss_type == "mse":
            squared_errors = errors ** 2
            loss = torch.sum(squared_errors * pad_mask, dim=-1) / (
                1e-7 + torch.sum(pad_mask, dim=-1)
            )
        elif loss_type == "huber":
            # Huber loss: 0.5 * error^2 if |error| <= delta, else delta * (|error| - 0.5 * delta)
            abs_errors = torch.abs(errors)
            squared_errors = errors ** 2
            # For |error| <= delta: use 0.5 * error^2
            # For |error| > delta: use delta * (|error| - 0.5 * delta)
            huber_loss = torch.where(
                abs_errors <= delta,
                0.5 * squared_errors,
                delta * (abs_errors - 0.5 * delta)
            )
            loss = torch.sum(huber_loss * pad_mask, dim=-1) / (
                1e-7 + torch.sum(pad_mask, dim=-1)
            )
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Must be 'mse' or 'huber'")
        
        # Average over the batch dimension
        if mask_loss is not None:
            mask_loss = (
                mask_loss.repeat_interleave(multiplicity, 0)
                .reshape(-1, multiplicity)
                .float()
            )
            loss = torch.sum(loss.reshape(-1, multiplicity) * mask_loss) / (
                torch.sum(mask_loss) + 1e-7
            )
        else:
            loss = torch.mean(loss)
    
    return {
        "loss": loss,
        "loss_breakdown": {
            "water_counts_loss": loss,
        },
    }
