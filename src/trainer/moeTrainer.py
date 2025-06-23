from .base import Base_Trainer
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os
from utils import r_squared

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoETrainer(Base_Trainer):

    def __init__(self, model, cfg, tc_rng):
        super().__init__(model, cfg, tc_rng)

    def _model_forward(self, data):
        """
        Perform a forward pass through the model.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            tuple: predicted target and mode logits.
        """
        input, _ = self.get_input_target(data)
        pred, mode_logits = self.model(input)
        return pred, mode_logits
    
    def get_input_target(self, data):
        """
        Extract the input and target data from the data object.

        Args:
            data: PyTorch Data object containing input and target data.

        Returns:
            tuple: (input data, (target data, mode data))
        """
        inputs = data[0]
        targets, modes = data[1]
        return inputs, (targets, modes)
    
    def accumulate(self, data):
        """
        Accumulate statistics for the model's normalizers.

        Args:
            data: PyTorch Geometric Data object containing input and target data.
        """
        data = self.move_to_device(data)
        input, (target, _) = self.get_input_target(data)
        self.model.accumulate(input, target)
    
    def get_pred(self, data):
        """
        Get the prediction.

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            tuple: (predictions, mode)
        """
        data = self.move_to_device(data)
        predict, mode_logits = self._model_forward(data)
        return predict, mode_logits

    def _loss_fn(self, data):
        """
        Calcuate the combined loss of the following components
        1. Task Loss: MSE between predictions and target
        2. Mode Loss: Cross-entropy between mode logits and true mode (1 hot matrix)

        Args:
            data: PyTorch Geometric Data object containing input data.

        Returns:
            torch.Tensor: Total Loss of model
        """

        pred, mode_logits = self.get_pred(data)
        _, (target, mode) = self.get_input_target(data)

        # Task loss: Mean Square Error between predictions and targets
        task_loss = torch.mean((pred - target) ** 2)
        
        # Mode loss: Cross-entropy between mode logits and true modes
        # Convert logits to probabilities
        mode_probs = torch.softmax(mode_logits, dim=1)
        
        # Calculate cross-entropy loss using one-hot encoded mode
        # We add a small epsilon to avoid log(0)
        mode_loss = -torch.mean(torch.sum(mode * torch.log(mode_probs + 1e-10), dim=1))
        
        # Get mode loss coefficient from config (default to 1.0 if not specified)
        mode_loss_coef = getattr(self.cfg.opt, 'mode_loss_coef', 1.0)
        
        # Combine losses
        total_loss = task_loss + mode_loss_coef * mode_loss
        
        # Store individual loss components for logging
        self.task_loss = task_loss.item()
        self.mode_loss = mode_loss.item()
        
        return total_loss

    def move_to_device(self, data):
        """
        Move data to the appropriate device, handling nested tuples.
        
        Args:
            data: Data to move (can be tensor, tuple, or nested tuple)
            
        Returns:
            Data moved to the device
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def _move_recursive(item):
            if isinstance(item, torch.Tensor):
                return item.to(device)
            elif isinstance(item, tuple):
                return tuple(_move_recursive(x) for x in item)
            elif isinstance(item, list):
                return [_move_recursive(x) for x in item]
            else:
                return item
        
        return _move_recursive(data)

    def get_metrics(self, data):
        """
        Calculate metrics for evaluating model performance.
        
        Args:
            data: PyTorch Data object containing input data.
            
        Returns:
            tuple: (r2, mode_accuracy)
                R-squared values and mode accuracy.
        """
        # Get predictions and mode logits
        pred, mode_logits = self.get_pred(data)
        _, (target, mode) = self.get_input_target(data)
        
        # Calculate R-squared for regression evaluation
        r2 = r_squared(pred, target)
        
        # Calculate mode prediction accuracy
        mode_probs = torch.softmax(mode_logits, dim=1)
        pred_mode = torch.argmax(mode_probs, dim=1)
        true_mode = torch.argmax(mode, dim=1)
        mode_accuracy = (pred_mode == true_mode).float().mean()
        
        return r2, mode_accuracy
    
    def print_metrics(self, data, prefix):

        # getting metrics
        r2, mode_accuracy = self.get_metrics(data)
        loss = self.get_loss(data)

        # Print overall mean R² first
        print(f"{prefix}_r2_mean: {r2.mean().item():.6f}")
        
        # Print R² for each dimension
        # for i in range(r2.shape[0]):
        #    print(f"{prefix}_r2_dim{i}: {r2[i].item():.6f}")
        # print(f"{prefix}_mode_accuracy: {mode_accuracy.item():.6f}")

        # printing metrics
        # print(f"Step {self.train_step}, {prefix}_total_loss: {loss.item():.6f}")
        # moved below beacuse baord_loss also prints total loss
        print(f"{prefix}_task_loss: {self.task_loss:.6f}")
        print(f"{prefix}_mode_loss: {self.mode_loss:.6f}")
    
    def eval_plot(self, data, prefix, board=True):
        """
        Create evaluation plots for model analysis.
        
        Args:
            data: PyTorch Data object.
            prefix: Prefix for the plot output ('train' or 'test').
            board: Whether to log plots to wandb (default: True).
        """
        # Get predictions and metrics
        pred, mode_logits = self.get_pred(data)
        _, (target, mode) = self.get_input_target(data)
        
        # Convert to numpy for plotting
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # 1. Create scatter plots for predictions vs targets
        n_outputs = pred_np.shape[1]
        fig, axes = plt.subplots(1, n_outputs, figsize=(n_outputs*5, 5))
        
        # Handle case of single output
        if n_outputs == 1:
            axes = [axes]
        
        for i in range(n_outputs):
            ax = axes[i]
            ax.scatter(target_np[:, i], pred_np[:, i], alpha=0.5)
            
            # Add diagonal line for perfect predictions
            min_val = min(target_np[:, i].min(), pred_np[:, i].min())
            max_val = max(target_np[:, i].max(), pred_np[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            # Calculate and display R-squared
            r2_val = r_squared(torch.tensor(pred_np[:, i]), torch.tensor(target_np[:, i])).item()
            ax.set_title(f'Output {i}: R² = {r2_val:.4f}')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
        
        plt.tight_layout()
        
        # Log to wandb if enabled
        if board:
            wandb.log({f"{prefix}_predictions": wandb.Image(fig)})
        
        # Optionally save the figure
        if hasattr(self.cfg, 'plot_dir'):
            os.makedirs(self.cfg.plot_dir, exist_ok=True)
            plt.savefig(f"{self.cfg.plot_dir}/{prefix}_preds_step_{self.train_step}.png")
        
        plt.close(fig)
        
        # 2. Create confusion matrix for mode predictions
        mode_probs = torch.softmax(mode_logits, dim=1)
        pred_mode = torch.argmax(mode_probs, dim=1).cpu().numpy()
        true_mode = torch.argmax(mode, dim=1).cpu().numpy()
        
        n_modes = mode.shape[1]
        confusion = np.zeros((n_modes, n_modes))
        
        # Fill confusion matrix
        for i in range(len(true_mode)):
            confusion[true_mode[i], pred_mode[i]] += 1
        
        # Normalize by row (true mode)
        row_sums = confusion.sum(axis=1, keepdims=True)
        norm_confusion = np.divide(confusion, row_sums, out=np.zeros_like(confusion), where=row_sums!=0)
        
        # Create confusion matrix plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(norm_confusion, cmap='Blues')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        
        # Set labels and title
        ax.set_xticks(np.arange(n_modes))
        ax.set_yticks(np.arange(n_modes))
        ax.set_xlabel('Predicted Mode')
        ax.set_ylabel('True Mode')
        ax.set_title('Mode Confusion Matrix')
        
        # Add text annotations in cells
        for i in range(n_modes):
            for j in range(n_modes):
                text = ax.text(j, i, f'{norm_confusion[i, j]:.2f}', 
                            ha="center", va="center", 
                            color="black" if norm_confusion[i, j] < 0.5 else "white")
        
        # Log to wandb if enabled
        if board:
            wandb.log({f"{prefix}_mode_confusion": wandb.Image(fig)})
        
        # Optionally save the figure
        if hasattr(self.cfg, 'plot_dir'):
            plt.savefig(f"{self.cfg.plot_dir}/{prefix}_confusion_step_{self.train_step}.png")
        
        plt.close(fig)