"""
Tests for Vbai Hub - Model Registry and HuggingFace Integration
"""

import pytest
import torch

import vbai
from vbai.hub.registry import ModelInfo, MODEL_REGISTRY, list_models, get_model_info, register_model
from vbai.hub.hub_utils import _build_checkpoint, _infer_model_type
from vbai.hub.model_card import generate_model_card


class TestModelRegistry:
    """Test model registry functionality."""

    def test_registry_has_models(self):
        assert len(MODEL_REGISTRY) >= 4
        assert 'vbai-2d-q' in MODEL_REGISTRY
        assert 'vbai-2d-f' in MODEL_REGISTRY
        assert 'vbai-3d-q' in MODEL_REGISTRY
        assert 'vbai-3d-f' in MODEL_REGISTRY

    def test_model_info_fields(self):
        info = MODEL_REGISTRY['vbai-3d-q']
        assert info.name == 'vbai-3d-q'
        assert info.model_type == '3d'
        assert info.variant == 'q'
        assert info.hub_id == 'Neurazum/vbai-3d-q'
        assert isinstance(info.tasks, dict)
        assert isinstance(info.tags, list)

    def test_list_all_models(self):
        models = list_models()
        assert len(models) >= 4

    def test_list_2d_models(self):
        models = list_models('2d')
        assert all(m.model_type == '2d' for m in models)
        assert len(models) >= 2

    def test_list_3d_models(self):
        models = list_models('3d')
        assert all(m.model_type == '3d' for m in models)
        assert len(models) >= 2

    def test_get_model_info(self):
        info = get_model_info('vbai-2d-q')
        assert info.name == 'vbai-2d-q'
        assert info.model_type == '2d'

    def test_get_model_info_not_found(self):
        with pytest.raises(KeyError, match="not found"):
            get_model_info('nonexistent-model')

    def test_register_custom_model(self):
        custom = ModelInfo(
            name='test-custom',
            description='Test custom model',
            model_type='3d',
            variant='q',
            tasks={'test_task': 2},
            default_input_shape=(1, 64, 64, 64),
        )
        register_model('test-custom', custom)
        assert 'test-custom' in MODEL_REGISTRY
        assert get_model_info('test-custom').name == 'test-custom'
        # Cleanup
        del MODEL_REGISTRY['test-custom']

    def test_vbai_top_level_access(self):
        """Test that hub functions are accessible from vbai namespace."""
        assert callable(vbai.list_models)
        assert callable(vbai.get_model_info)
        assert callable(vbai.register_model)
        assert callable(vbai.from_hub)
        assert callable(vbai.push_to_hub)
        assert callable(vbai.download_from_hub)
        assert vbai.ModelInfo is ModelInfo
        assert vbai.MODEL_REGISTRY is MODEL_REGISTRY


class TestBuildCheckpoint:
    """Test checkpoint building from models."""

    def test_build_2d_checkpoint(self):
        model = vbai.MultiTaskBrainModel(variant='f', tasks=['dementia'])
        checkpoint = _build_checkpoint(model)
        assert checkpoint['model_type'] == '2d'
        assert 'model_state_dict' in checkpoint
        assert checkpoint['config']['variant'] == 'f'

    def test_build_3d_checkpoint(self):
        model = vbai.MultiTask3DBrainModel(
            variant='f', tasks={'alzheimer': 3}, input_shape=(64, 64, 64)
        )
        checkpoint = _build_checkpoint(model)
        assert checkpoint['model_type'] == '3d'
        assert checkpoint['config']['variant'] == 'f'
        assert checkpoint['config']['tasks'] == {'alzheimer': 3}
        assert checkpoint['config']['input_shape'] == (64, 64, 64)


class TestInferModelType:
    """Test model type inference from config."""

    def test_infer_3d_from_dict_tasks(self):
        assert _infer_model_type({'tasks': {'alzheimer': 3}}) == '3d'

    def test_infer_3d_from_input_shape(self):
        assert _infer_model_type({'input_shape': (96, 96, 96)}) == '3d'

    def test_infer_2d_from_list_tasks(self):
        assert _infer_model_type({'tasks': ['dementia', 'tumor']}) == '2d'

    def test_infer_2d_default(self):
        assert _infer_model_type({}) == '2d'


class TestModelCard:
    """Test model card generation."""

    def test_generate_2d_card(self):
        model = vbai.MultiTaskBrainModel(variant='f')
        card = generate_model_card(model)
        assert '---' in card  # YAML frontmatter
        assert 'vbai' in card
        assert '2D' in card
        assert 'Neurazum' in card

    def test_generate_3d_card(self):
        model = vbai.MultiTask3DBrainModel(variant='q', tasks={'alzheimer': 3})
        card = generate_model_card(model)
        assert '3D' in card
        assert 'alzheimer' in card
        assert 'NIfTI' in card

    def test_generate_card_with_metrics(self):
        model = vbai.MultiTask3DBrainModel(variant='f', tasks={'alzheimer': 3})
        card = generate_model_card(model, metrics={'accuracy': 0.92, 'f1_score': 0.89})
        assert 'accuracy' in card
        assert '0.9200' in card

    def test_generate_card_with_dataset_info(self):
        model = vbai.MultiTaskBrainModel(variant='f')
        card = generate_model_card(model, dataset_info='ADNI dataset with 1000 subjects')
        assert 'ADNI' in card


class TestConvenienceMethods:
    """Test push_to_hub and export_onnx convenience methods exist on models."""

    def test_2d_model_has_push_to_hub(self):
        model = vbai.MultiTaskBrainModel(variant='f')
        assert hasattr(model, 'push_to_hub')
        assert callable(model.push_to_hub)

    def test_2d_model_has_export_onnx(self):
        model = vbai.MultiTaskBrainModel(variant='f')
        assert hasattr(model, 'export_onnx')
        assert callable(model.export_onnx)

    def test_3d_model_has_push_to_hub(self):
        model = vbai.MultiTask3DBrainModel(variant='f', tasks={'test': 2})
        assert hasattr(model, 'push_to_hub')
        assert callable(model.push_to_hub)

    def test_3d_model_has_export_onnx(self):
        model = vbai.MultiTask3DBrainModel(variant='f', tasks={'test': 2})
        assert hasattr(model, 'export_onnx')
        assert callable(model.export_onnx)
