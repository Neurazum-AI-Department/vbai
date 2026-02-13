"""
ONNX Inference for Vbai Models

PyTorch-free inference using ONNX Runtime.
"""

from typing import Optional, List, Tuple, Dict, Union

import numpy as np


class ONNXModel:
    """
    ONNX Runtime inference wrapper for vbai models.

    Allows running inference without PyTorch installed, using only
    numpy and onnxruntime.

    Args:
        onnx_path: Path to the .onnx model file
        providers: ONNX Runtime execution providers
            Default: ['CUDAExecutionProvider', 'CPUExecutionProvider']

    Example:
        >>> model = vbai.ONNXModel('model.onnx')
        >>> output = model.predict(input_array)

        >>> # For 3D NIfTI
        >>> model = vbai.ONNXModel('model_3d.onnx')
        >>> output = model.predict_nifti('brain_scan.nii.gz')
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[List[str]] = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX inference. "
                "Install with: pip install vbai[onnx] "
                "or: pip install onnxruntime>=1.12.0"
            )

        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(
        self,
        input_array: np.ndarray,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Run inference on a numpy array.

        Args:
            input_array: Input array matching the model's expected shape.
                For 2D: (B, 3, 224, 224) or (3, 224, 224)
                For 3D: (B, 1, D, H, W) or (1, D, H, W)

        Returns:
            Model output(s) as numpy array(s).
            Single output → np.ndarray, multiple → list of np.ndarray.
        """
        # Add batch dim if needed
        if input_array.ndim == len(self.input_shape) - 1:
            input_array = np.expand_dims(input_array, axis=0)

        input_array = input_array.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: input_array})

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def predict_nifti(
        self,
        nifti_path: str,
        target_shape: Tuple[int, int, int] = (96, 96, 96),
    ) -> np.ndarray:
        """
        Run inference on a NIfTI (.nii/.nii.gz) file.

        Handles loading, normalization, and resizing.

        Args:
            nifti_path: Path to the NIfTI file
            target_shape: Target volume shape (D, H, W) for resizing

        Returns:
            Model output as numpy array (logits or probabilities)
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required for NIfTI loading. "
                "Install with: pip install nibabel"
            )

        # Load volume
        img = nib.load(nifti_path).get_fdata()
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # Brain mask normalization
        brain_mask = img > img.mean()
        if brain_mask.sum() > 0:
            brain_pixels = img[brain_mask]
            p1, p99 = np.percentile(brain_pixels, [1, 99])
            img = np.clip(img, p1, p99)
            mean, std = brain_pixels.mean(), brain_pixels.std()
            if std > 1e-6:
                img = (img - mean) / (std + 1e-8)
            else:
                img = img - mean
        else:
            mean, std = img.mean(), img.std()
            if std > 1e-6:
                img = (img - mean) / (std + 1e-8)

        # Min-max to [0, 1]
        v_min, v_max = img.min(), img.max()
        if abs(v_max - v_min) > 1e-6:
            img = (img - v_min) / (v_max - v_min + 1e-8)
        else:
            img = np.zeros_like(img)
        img = np.clip(img, 0, 1)

        # Resize using scipy
        try:
            from scipy.ndimage import zoom
        except ImportError:
            raise ImportError(
                "scipy is required for volume resizing. "
                "Install with: pip install scipy"
            )

        current_shape = img.shape
        zoom_factors = tuple(t / c for t, c in zip(target_shape, current_shape))
        img = zoom(img, zoom_factors, order=1)

        # Prepare input: (1, 1, D, H, W)
        input_array = img[np.newaxis, np.newaxis, ...].astype(np.float32)

        return self.predict(input_array)

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits."""
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    @property
    def num_outputs(self) -> int:
        """Number of model outputs."""
        return len(self.output_names)

    def __repr__(self) -> str:
        return (
            f"ONNXModel(\n"
            f"  input_shape={self.input_shape},\n"
            f"  outputs={self.output_names}\n"
            f")"
        )
