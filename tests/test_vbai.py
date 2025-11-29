"""
Tests for Vbai Library
"""

import pytest
import torch
import numpy as np
from PIL import Image

import vbai
from vbai.models import MultiTaskBrainModel, SharedBackbone, AttentionModule
from vbai.configs import ModelConfig, TrainingConfig


class TestMultiTaskBrainModel:
    """Tests for MultiTaskBrainModel."""
    
    def test_model_creation_variant_q(self):
        """Test creating quality variant model."""
        model = MultiTaskBrainModel(variant='q')
        assert model.variant == 'q'
        assert model.num_dementia_classes == 6
        assert model.num_tumor_classes == 4
    
    def test_model_creation_variant_f(self):
        """Test creating fast variant model."""
        model = MultiTaskBrainModel(variant='f')
        assert model.variant == 'f'
    
    def test_forward_pass(self):
        """Test model forward pass."""
        model = MultiTaskBrainModel(variant='f')
        x = torch.randn(2, 3, 224, 224)
        
        dementia_out, tumor_out = model(x)
        
        assert dementia_out.shape == (2, 6)
        assert tumor_out.shape == (2, 4)
    
    def test_forward_with_attention(self):
        """Test forward pass returning attention maps."""
        model = MultiTaskBrainModel(variant='f')
        x = torch.randn(2, 3, 224, 224)
        
        dementia_out, tumor_out, dem_attn, tum_attn = model(x, return_attention=True)
        
        assert dem_attn is not None
        assert tum_attn is not None
    
    def test_predict_with_tensor(self):
        """Test prediction with tensor input."""
        model = MultiTaskBrainModel(variant='f')
        x = torch.randn(1, 3, 224, 224)
        
        result = model.predict(x)
        
        assert result.dementia_class in model.DEMENTIA_CLASSES
        assert result.tumor_class in model.TUMOR_CLASSES
        assert 0 <= result.dementia_confidence <= 1
        assert 0 <= result.tumor_confidence <= 1
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = MultiTaskBrainModel(variant='f')
        params = model.count_parameters()
        
        assert 'total' in params
        assert 'trainable' in params
        assert params['total'] > 0


class TestSharedBackbone:
    """Tests for SharedBackbone."""
    
    def test_backbone_variant_q(self):
        """Test quality variant backbone."""
        backbone = SharedBackbone(variant='q')
        x = torch.randn(2, 3, 224, 224)
        
        out = backbone(x)
        
        assert out.shape[1] == 512  # Final channels for 'q'
    
    def test_backbone_variant_f(self):
        """Test fast variant backbone."""
        backbone = SharedBackbone(variant='f')
        x = torch.randn(2, 3, 224, 224)
        
        out = backbone(x)
        
        assert out.shape[1] == 128  # Final channels for 'f'
    
    def test_invalid_variant(self):
        """Test that invalid variant raises error."""
        with pytest.raises(ValueError):
            SharedBackbone(variant='invalid')


class TestAttentionModule:
    """Tests for AttentionModule."""
    
    def test_attention_forward(self):
        """Test attention module forward pass."""
        attention = AttentionModule(in_channels=128)
        x = torch.randn(2, 128, 14, 14)
        
        attended, attn_map = attention(x)
        
        assert attended.shape == x.shape
        assert attn_map.shape == (2, 1, 14, 14)


class TestConfigs:
    """Tests for configuration classes."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        
        assert config.variant == 'q'
        assert config.num_dementia_classes == 6
        assert config.image_size == 224
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        
        assert config.epochs == 10
        assert config.batch_size == 32
        assert config.lr == 0.0005
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = ModelConfig(variant='f', dropout=0.3)
        d = config.to_dict()
        
        assert d['variant'] == 'f'
        assert d['dropout'] == 0.3
    
    def test_config_from_dict(self):
        """Test config deserialization."""
        d = {'variant': 'f', 'dropout': 0.3}
        config = ModelConfig.from_dict(d)
        
        assert config.variant == 'f'
        assert config.dropout == 0.3
    
    def test_get_default_config(self):
        """Test preset configurations."""
        config = vbai.get_default_config('fast')
        
        assert config.model.variant == 'f'
        assert config.training.epochs == 5


class TestDataTransforms:
    """Tests for data transforms."""
    
    def test_train_transforms(self):
        """Test training transforms."""
        from vbai.data import get_train_transforms
        
        transform = get_train_transforms()
        img = Image.new('RGB', (256, 256))
        
        tensor = transform(img)
        
        assert tensor.shape == (3, 224, 224)
    
    def test_val_transforms(self):
        """Test validation transforms."""
        from vbai.data import get_val_transforms
        
        transform = get_val_transforms()
        img = Image.new('RGB', (256, 256))
        
        tensor = transform(img)
        
        assert tensor.shape == (3, 224, 224)


class TestLosses:
    """Tests for loss functions."""
    
    def test_multitask_loss(self):
        """Test MultiTaskLoss computation."""
        from vbai.training import MultiTaskLoss
        
        loss_fn = MultiTaskLoss()
        
        dementia_logits = torch.randn(4, 6)
        tumor_logits = torch.randn(4, 4)
        dementia_labels = torch.tensor([0, 1, 2, -1])  # -1 is ignored
        tumor_labels = torch.tensor([-1, -1, 1, 2])
        
        loss = loss_fn(dementia_logits, tumor_logits, dementia_labels, tumor_labels)
        
        assert loss.item() >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
