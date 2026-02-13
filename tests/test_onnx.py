"""
Tests for Vbai ONNX Export and Inference
"""

import os
import tempfile

import pytest
import torch
import numpy as np

import vbai
from vbai.export.onnx_export import export_onnx, _MultiTask2DWrapper, _Dict2TupleWrapper


class TestONNXWrappers:
    """Test ONNX wrapper classes."""

    def test_2d_wrapper_both_tasks(self):
        model = vbai.MultiTaskBrainModel(variant='f', tasks=['dementia', 'tumor'])
        wrapper = _MultiTask2DWrapper(model)
        x = torch.randn(1, 3, 224, 224)
        wrapper.eval()
        with torch.no_grad():
            outputs = wrapper(x)
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2

    def test_2d_wrapper_single_task(self):
        model = vbai.MultiTaskBrainModel(variant='f', tasks=['dementia'])
        wrapper = _MultiTask2DWrapper(model)
        x = torch.randn(1, 3, 224, 224)
        wrapper.eval()
        with torch.no_grad():
            output = wrapper(x)
        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == 6  # dementia classes

    def test_3d_wrapper_single_task(self):
        model = vbai.MultiTask3DBrainModel(variant='f', tasks={'alzheimer': 3})
        wrapper = _Dict2TupleWrapper(model, ['alzheimer'])
        x = torch.randn(1, 1, 64, 64, 64)
        wrapper.eval()
        with torch.no_grad():
            output = wrapper(x)
        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == 3

    def test_3d_wrapper_multi_task(self):
        model = vbai.MultiTask3DBrainModel(
            variant='f', tasks={'alzheimer': 3, 'tumor': 4}
        )
        wrapper = _Dict2TupleWrapper(model, ['alzheimer', 'tumor'])
        x = torch.randn(1, 1, 64, 64, 64)
        wrapper.eval()
        with torch.no_grad():
            outputs = wrapper(x)
        assert isinstance(outputs, tuple)
        assert len(outputs) == 2
        assert outputs[0].shape[1] == 3
        assert outputs[1].shape[1] == 4


class TestONNXExport2D:
    """Test ONNX export for 2D models."""

    def test_export_2d_both_tasks(self):
        model = vbai.MultiTaskBrainModel(variant='f', tasks=['dementia', 'tumor'])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model_2d.onnx')
            result = export_onnx(model, path, verify=False)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0

    def test_export_2d_single_task(self):
        model = vbai.MultiTaskBrainModel(variant='f', tasks=['tumor'])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model_tumor.onnx')
            result = export_onnx(model, path, verify=False)
            assert os.path.exists(result)

    def test_export_2d_custom_shape(self):
        model = vbai.MultiTaskBrainModel(variant='f', tasks=['dementia'])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model_custom.onnx')
            result = export_onnx(
                model, path, input_shape=(3, 224, 224), verify=False
            )
            assert os.path.exists(result)

    def test_export_2d_via_convenience_method(self):
        model = vbai.MultiTaskBrainModel(variant='f', tasks=['dementia'])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model_conv.onnx')
            result = model.export_onnx(path, verify=False)
            assert os.path.exists(result)


class TestONNXExport3D:
    """Test ONNX export for 3D models."""

    def test_export_3d_single_task(self):
        model = vbai.MultiTask3DBrainModel(
            variant='f', tasks={'alzheimer': 3}, input_shape=(64, 64, 64)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model_3d.onnx')
            result = export_onnx(model, path, verify=False)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0

    def test_export_3d_multi_task(self):
        model = vbai.MultiTask3DBrainModel(
            variant='f', tasks={'alzheimer': 3, 'tumor': 4}, input_shape=(64, 64, 64)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model_3d_multi.onnx')
            result = export_onnx(model, path, verify=False)
            assert os.path.exists(result)

    def test_export_3d_via_convenience_method(self):
        model = vbai.MultiTask3DBrainModel(
            variant='f', tasks={'test': 2}, input_shape=(64, 64, 64)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model_conv_3d.onnx')
            result = model.export_onnx(path, verify=False)
            assert os.path.exists(result)


class TestONNXExportTopLevel:
    """Test vbai.export_onnx top-level function."""

    def test_top_level_export(self):
        model = vbai.MultiTaskBrainModel(variant='f', tasks=['dementia'])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.onnx')
            result = vbai.export_onnx(model, path, verify=False)
            assert os.path.exists(result)


class TestONNXModelClass:
    """Test ONNXModel inference class (requires onnxruntime)."""

    @pytest.fixture
    def exported_2d_model(self):
        model = vbai.MultiTaskBrainModel(variant='f', tasks=['dementia'])
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, 'test_2d.onnx')
        export_onnx(model, path, verify=False)
        yield path
        os.unlink(path)
        os.rmdir(tmpdir)

    @pytest.fixture
    def exported_3d_model(self):
        model = vbai.MultiTask3DBrainModel(
            variant='f', tasks={'alzheimer': 3}, input_shape=(64, 64, 64)
        )
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, 'test_3d.onnx')
        export_onnx(model, path, verify=False)
        yield path
        os.unlink(path)
        os.rmdir(tmpdir)

    def test_onnx_model_2d_inference(self, exported_2d_model):
        try:
            import onnxruntime
        except ImportError:
            pytest.skip("onnxruntime not installed")

        onnx_model = vbai.ONNXModel(exported_2d_model)
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        output = onnx_model.predict(input_data)
        assert isinstance(output, np.ndarray)
        assert output.shape[0] == 1
        assert output.shape[1] == 6  # dementia classes

    def test_onnx_model_3d_inference(self, exported_3d_model):
        try:
            import onnxruntime
        except ImportError:
            pytest.skip("onnxruntime not installed")

        onnx_model = vbai.ONNXModel(exported_3d_model)
        input_data = np.random.randn(1, 1, 64, 64, 64).astype(np.float32)
        output = onnx_model.predict(input_data)
        assert isinstance(output, np.ndarray)
        assert output.shape[0] == 1
        assert output.shape[1] == 3  # alzheimer classes

    def test_onnx_model_softmax(self, exported_2d_model):
        try:
            import onnxruntime
        except ImportError:
            pytest.skip("onnxruntime not installed")

        onnx_model = vbai.ONNXModel(exported_2d_model)
        logits = np.array([[1.0, 2.0, 3.0, 0.5, 0.1, 0.2]])
        probs = onnx_model.softmax(logits)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert probs.argmax() == 2

    def test_onnx_model_repr(self, exported_2d_model):
        try:
            import onnxruntime
        except ImportError:
            pytest.skip("onnxruntime not installed")

        onnx_model = vbai.ONNXModel(exported_2d_model)
        repr_str = repr(onnx_model)
        assert 'ONNXModel' in repr_str
        assert 'input_shape' in repr_str
