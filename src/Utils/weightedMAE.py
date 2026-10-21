import torch


class WeightedMAELoss(torch.nn.Module):
    def __init__(self, weight=None) -> None:
        super(WeightedMAELoss, self).__init__()
        self.weight = weight

    def forward(self, preds, targets):
        """
        Compute the class-weighted Mean Absolute Error (MAE) loss.

        Args:
            preds (torch.Tensor): The predicted values, shape (num_samples, num_classes).
            targets (torch.Tensor): The true values, shape (num_samples, num_classes).

        Returns:
            torch.Tensor: The computed weighted MAE loss.
        """

        # Class weights must match the number of classes, not the number of samples.
        if self.weight is None:
            weight = torch.ones(preds.size(1), device=preds.device, dtype=preds.dtype)
        else:
            weight = self.weight.to(device=preds.device, dtype=preds.dtype)

        if weight.dim() != 1 or weight.numel() != preds.size(1):
            raise ValueError(
                f"weight must be a 1D tensor with one value per class; "
                f"got shape {tuple(weight.shape)} for {preds.size(1)} classes."
            )

        # Normalize class weights so the loss scale stays comparable across runs.
        weight = weight / weight.sum().clamp_min(torch.finfo(weight.dtype).eps)

        # Ensure labels are one-hot encoded if they are not already.
        if targets.dim() == 1 or (targets.dim() == 2 and targets.size(1) == 1):
            targets = torch.nn.functional.one_hot(targets.long(), num_classes=preds.size(1)).float()

        targets = targets.to(device=preds.device, dtype=preds.dtype)

        # Apply class weights to each class error, then average across samples.
        l1_diff = (preds - targets).abs() # shape (num_samples, num_classes)
        weighted_l1 = l1_diff * weight.unsqueeze(0) # shape (num_samples, num_classes)
        loss = weighted_l1.sum(dim=1).mean() # Sum to have weighted l1 per sample, then average across samples

        return loss


if __name__ == "__main__":
    # Example usage
    preds = torch.tensor([[0.5, 0.2, 0.6, 0.9], [0.5, 0.8, 0.4, 0.1]]).T # shape (4, 2) for 4 samples and 2 classes
    targets_soft = torch.tensor([[1., 0., 0., 1.], [0., 1., 1., 0.]]).T     # shape (4, 2) for 4 samples and 2 classes
    targets_hard = torch.tensor([0., 1., 1., 0.])     # shape (4,) for 4 samples
    weight = torch.tensor([0.8, 0.2])  # Example class weights for 2 classes

    loss_fn = WeightedMAELoss(weight=weight)
    loss = loss_fn(preds, targets_soft)
    print(f"Weighted MAE Loss (soft targets): {loss.item()}")
    
    loss_fn_no_weight = WeightedMAELoss()
    loss_no_weight = loss_fn_no_weight(preds, targets_soft)
    print(f"MAE Loss without weights: {loss_no_weight.item()}")

    loss_fn = WeightedMAELoss(weight=weight)
    loss = loss_fn(preds, targets_hard)
    print(f"Weighted MAE Loss (hard targets): {loss.item()}")
    
    loss_fn_no_weight = WeightedMAELoss()
    loss_no_weight = loss_fn_no_weight(preds, targets_hard)
    print(f"MAE Loss without weights: {loss_no_weight.item()}")