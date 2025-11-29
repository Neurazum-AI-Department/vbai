"""
Visualization Utilities for Vbai
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union, List
from pathlib import Path

import torch
from PIL import Image

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class VisualizationManager:
    """
    Manager for creating visualizations of model predictions.
    
    Creates attention heatmaps overlaid on input images,
    with dementia attention in blue and tumor attention in red.
    
    Args:
        output_dir: Directory to save visualizations
        dementia_color: Color for dementia attention (default: blue)
        tumor_color: Color for tumor attention (default: red)
    
    Example:
        >>> vis = VisualizationManager(output_dir='./visualizations')
        >>> vis.visualize(image, result, save=True)
    """
    
    def __init__(
        self,
        output_dir: str = './visualizations',
        dementia_color: str = 'blue',
        tumor_color: str = 'red',
        alpha: float = 0.4
    ):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dementia_cmap = self._get_colormap(dementia_color)
        self.tumor_cmap = self._get_colormap(tumor_color)
        self.alpha = alpha
    
    def _get_colormap(self, color: str):
        """Get matplotlib colormap for given color."""
        colormaps = {
            'blue': cm.Blues,
            'red': cm.Reds,
            'green': cm.Greens,
            'purple': cm.Purples,
            'orange': cm.Oranges,
        }
        return colormaps.get(color, cm.viridis)
    
    def visualize(
        self,
        image: Union[torch.Tensor, np.ndarray, Image.Image],
        prediction_result,
        save: bool = True,
        filename: Optional[str] = None,
        show: bool = False
    ) -> Optional[np.ndarray]:
        """
        Create visualization of model prediction.
        
        Args:
            image: Input image
            prediction_result: Result from model.predict()
            save: Whether to save the visualization
            filename: Filename for saved image
            show: Whether to display the plot
        
        Returns:
            Visualization as numpy array if not saving
        """
        # Convert image to numpy
        if isinstance(image, torch.Tensor):
            img_np = self._tensor_to_numpy(image)
        elif isinstance(image, Image.Image):
            img_np = np.array(image)
        else:
            img_np = image
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Dementia attention
        if prediction_result.dementia_attention is not None:
            dem_heatmap = self._create_heatmap(
                prediction_result.dementia_attention,
                img_np.shape[:2],
                self.dementia_cmap
            )
            axes[1].imshow(img_np)
            axes[1].imshow(dem_heatmap, alpha=self.alpha)
            axes[1].set_title(
                f'Dementia: {prediction_result.dementia_class}\n'
                f'({prediction_result.dementia_confidence:.1%})'
            )
        else:
            axes[1].imshow(img_np)
            axes[1].set_title(
                f'Dementia: {prediction_result.dementia_class}\n'
                f'({prediction_result.dementia_confidence:.1%})'
            )
        axes[1].axis('off')
        
        # Tumor attention
        if prediction_result.tumor_attention is not None:
            tum_heatmap = self._create_heatmap(
                prediction_result.tumor_attention,
                img_np.shape[:2],
                self.tumor_cmap
            )
            axes[2].imshow(img_np)
            axes[2].imshow(tum_heatmap, alpha=self.alpha)
            axes[2].set_title(
                f'Tumor: {prediction_result.tumor_class}\n'
                f'({prediction_result.tumor_confidence:.1%})'
            )
        else:
            axes[2].imshow(img_np)
            axes[2].set_title(
                f'Tumor: {prediction_result.tumor_class}\n'
                f'({prediction_result.tumor_confidence:.1%})'
            )
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f'vis_{prediction_result.dementia_class}_{prediction_result.tumor_class}.png'
            save_path = self.output_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return None
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array for visualization."""
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        tensor = tensor.cpu() * std + mean
        tensor = tensor.clamp(0, 1)
        
        # To numpy (H, W, C)
        return (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    def _create_heatmap(
        self,
        attention: torch.Tensor,
        target_size: Tuple[int, int],
        cmap
    ) -> np.ndarray:
        """Create colored heatmap from attention tensor."""
        # Get attention as numpy
        if attention.dim() == 4:
            attn = attention[0, 0].cpu().numpy()
        elif attention.dim() == 3:
            attn = attention[0].cpu().numpy()
        else:
            attn = attention.cpu().numpy()
        
        # Normalize to 0-1
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        
        # Resize to target size
        from PIL import Image as PILImage
        attn_pil = PILImage.fromarray((attn * 255).astype(np.uint8))
        attn_resized = attn_pil.resize((target_size[1], target_size[0]), PILImage.BILINEAR)
        attn_np = np.array(attn_resized) / 255.0
        
        # Apply colormap
        colored = cmap(attn_np)
        
        return colored


def visualize_prediction(
    model,
    image_path: str,
    output_path: Optional[str] = None,
    show: bool = True
) -> Dict:
    """
    Quick visualization of a single prediction.
    
    Args:
        model: Trained MultiTaskBrainModel
        image_path: Path to input image
        output_path: Path to save visualization
        show: Whether to display the plot
    
    Returns:
        Prediction result dictionary
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    # Load and predict
    image = Image.open(image_path).convert('RGB')
    result = model.predict(image, return_attention=True)
    
    # Visualize
    vis = VisualizationManager()
    vis.visualize(image, result, save=(output_path is not None), 
                  filename=output_path, show=show)
    
    return {
        'dementia_class': result.dementia_class,
        'dementia_confidence': result.dementia_confidence,
        'tumor_class': result.tumor_class,
        'tumor_confidence': result.tumor_confidence,
    }


def create_attention_heatmap(
    attention: torch.Tensor,
    original_image: Union[torch.Tensor, np.ndarray, Image.Image],
    colormap: str = 'jet',
    alpha: float = 0.4
) -> np.ndarray:
    """
    Create an attention heatmap overlay.
    
    Args:
        attention: Attention tensor from model
        original_image: Original input image
        colormap: Matplotlib colormap name
        alpha: Transparency of overlay
    
    Returns:
        Combined image as numpy array
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    # Convert image
    if isinstance(original_image, torch.Tensor):
        if original_image.dim() == 4:
            original_image = original_image[0]
        img = original_image.permute(1, 2, 0).cpu().numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
    elif isinstance(original_image, Image.Image):
        img = np.array(original_image) / 255.0
    else:
        img = original_image / 255.0 if original_image.max() > 1 else original_image
    
    # Process attention
    if attention.dim() == 4:
        attn = attention[0, 0].cpu().numpy()
    elif attention.dim() == 3:
        attn = attention[0].cpu().numpy()
    else:
        attn = attention.cpu().numpy()
    
    # Normalize
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    
    # Resize attention to image size
    attn_pil = Image.fromarray((attn * 255).astype(np.uint8))
    attn_resized = np.array(attn_pil.resize((img.shape[1], img.shape[0]))) / 255.0
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(attn_resized)[:, :, :3]
    
    # Blend
    combined = (1 - alpha) * img + alpha * heatmap
    combined = np.clip(combined, 0, 1)
    
    return (combined * 255).astype(np.uint8)


def plot_training_history(
    history,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training history curves.
    
    Args:
        history: TrainingHistory object from Trainer
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history.train_loss) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history.train_loss, 'b-', label='Train')
    if history.val_loss:
        axes[0, 0].plot(epochs, history.val_loss, 'r-', label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dementia Accuracy
    axes[0, 1].plot(epochs, history.dementia_acc, 'b-', label='Train')
    if history.val_dementia_acc:
        axes[0, 1].plot(epochs, history.val_dementia_acc, 'r-', label='Val')
    axes[0, 1].set_title('Dementia Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Tumor Accuracy
    axes[1, 0].plot(epochs, history.tumor_acc, 'b-', label='Train')
    if history.val_tumor_acc:
        axes[1, 0].plot(epochs, history.val_tumor_acc, 'r-', label='Val')
    axes[1, 0].set_title('Tumor Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    if history.lr:
        axes[1, 1].plot(epochs, history.lr, 'g-')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"History plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
