"""
Tests for Vbai Advanced Augmentation
"""

import pytest
import numpy as np
import torch

import vbai
from vbai.data.augmentation import (
    simulate_bias_field,
    simulate_ghosting,
    simulate_spike_noise,
    simulate_rician_noise,
    simulate_mri_artifacts,
    elastic_deformation_2d,
    elastic_deformation_3d,
    mixup,
    cutmix,
    MRIAutoAugment,
)


# ── MRI Artifact Simulation ──


class TestBiasField:

    def test_2d(self):
        img = np.random.rand(64, 64).astype(np.float32)
        result = simulate_bias_field(img, intensity=0.3)
        assert result.shape == img.shape
        assert result.min() >= 0 and result.max() <= 1

    def test_3d(self):
        img = np.random.rand(32, 32, 32).astype(np.float32)
        result = simulate_bias_field(img, intensity=0.3)
        assert result.shape == img.shape
        assert result.min() >= 0 and result.max() <= 1

    def test_changes_image(self):
        img = np.random.rand(64, 64).astype(np.float32)
        result = simulate_bias_field(img, intensity=0.5)
        assert not np.allclose(img, result)


class TestGhosting:

    def test_2d(self):
        img = np.random.rand(64, 64).astype(np.float32)
        result = simulate_ghosting(img, num_ghosts=3, intensity=0.15)
        assert result.shape == img.shape
        assert result.min() >= 0 and result.max() <= 1

    def test_3d(self):
        img = np.random.rand(32, 32, 32).astype(np.float32)
        result = simulate_ghosting(img, num_ghosts=2, intensity=0.1)
        assert result.shape == img.shape

    def test_axis_selection(self):
        img = np.random.rand(32, 32, 32).astype(np.float32)
        result = simulate_ghosting(img, axis=1)
        assert result.shape == img.shape


class TestSpikeNoise:

    def test_2d(self):
        img = np.random.rand(64, 64).astype(np.float32)
        result = simulate_spike_noise(img, num_spikes=1, intensity=0.5)
        assert result.shape == img.shape
        assert result.min() >= 0 and result.max() <= 1

    def test_3d(self):
        img = np.random.rand(32, 32, 32).astype(np.float32)
        result = simulate_spike_noise(img, num_spikes=2, intensity=0.5)
        assert result.shape == img.shape


class TestRicianNoise:

    def test_2d(self):
        img = np.random.rand(64, 64).astype(np.float32)
        result = simulate_rician_noise(img, std=0.03)
        assert result.shape == img.shape
        assert result.min() >= 0 and result.max() <= 1

    def test_3d(self):
        img = np.random.rand(32, 32, 32).astype(np.float32)
        result = simulate_rician_noise(img, std=0.05)
        assert result.shape == img.shape


class TestMRIArtifactsCombined:

    def test_combined_2d(self):
        img = np.random.rand(64, 64).astype(np.float32)
        result = simulate_mri_artifacts(img, p=1.0)
        assert result.shape == img.shape
        assert result.min() >= 0 and result.max() <= 1

    def test_combined_3d(self):
        img = np.random.rand(32, 32, 32).astype(np.float32)
        result = simulate_mri_artifacts(img, p=1.0)
        assert result.shape == img.shape

    def test_selective_artifacts(self):
        img = np.random.rand(64, 64).astype(np.float32)
        result = simulate_mri_artifacts(img, p=1.0, artifacts=['bias_field', 'rician_noise'])
        assert result.shape == img.shape

    def test_top_level_access(self):
        assert callable(vbai.simulate_bias_field)
        assert callable(vbai.simulate_ghosting)
        assert callable(vbai.simulate_mri_artifacts)


# ── Elastic Deformation ──


class TestElasticDeformation:

    def test_2d(self):
        img = np.random.rand(64, 64).astype(np.float32)
        result = elastic_deformation_2d(img, alpha=50, sigma=5)
        assert result.shape == img.shape
        assert result.min() >= 0 and result.max() <= 1

    def test_2d_multichannel(self):
        img = np.random.rand(64, 64, 3).astype(np.float32)
        result = elastic_deformation_2d(img, alpha=50, sigma=5)
        assert result.shape == img.shape

    def test_3d(self):
        img = np.random.rand(32, 32, 32).astype(np.float32)
        result = elastic_deformation_3d(img, alpha=30, sigma=4)
        assert result.shape == img.shape
        assert result.min() >= 0 and result.max() <= 1

    def test_changes_image(self):
        img = np.random.rand(32, 32, 32).astype(np.float32)
        result = elastic_deformation_3d(img, alpha=30, sigma=4)
        assert not np.allclose(img, result)

    def test_top_level_access(self):
        assert callable(vbai.elastic_deformation_2d)
        assert callable(vbai.elastic_deformation_3d)


# ── MixUp / CutMix ──


class TestMixUp:

    def test_2d(self):
        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 4, (8,))
        mixed, la, lb, lam = mixup(images, labels, alpha=0.2)
        assert mixed.shape == images.shape
        assert la.shape == labels.shape
        assert lb.shape == labels.shape
        assert 0 <= lam <= 1

    def test_3d(self):
        images = torch.randn(4, 1, 16, 16, 16)
        labels = torch.randint(0, 3, (4,))
        mixed, la, lb, lam = mixup(images, labels, alpha=0.4)
        assert mixed.shape == images.shape

    def test_alpha_zero(self):
        images = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 4, (4,))
        mixed, la, lb, lam = mixup(images, labels, alpha=0)
        assert lam == 1.0
        assert torch.allclose(mixed, images)


class TestCutMix:

    def test_2d(self):
        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 4, (8,))
        mixed, la, lb, lam = cutmix(images, labels, alpha=1.0)
        assert mixed.shape == images.shape
        assert 0 <= lam <= 1

    def test_3d(self):
        images = torch.randn(4, 1, 16, 16, 16)
        labels = torch.randint(0, 3, (4,))
        mixed, la, lb, lam = cutmix(images, labels, alpha=1.0)
        assert mixed.shape == images.shape
        assert 0 <= lam <= 1

    def test_top_level_access(self):
        assert callable(vbai.mixup)
        assert callable(vbai.cutmix)


# ── AutoAugment ──


class TestAutoAugment:

    def test_2d(self):
        aug = MRIAutoAugment(mode='2d', num_policies=5)
        img = np.random.rand(64, 64).astype(np.float32)
        result = aug(img)
        assert result.shape == img.shape
        assert result.min() >= 0 and result.max() <= 1

    def test_3d(self):
        aug = MRIAutoAugment(mode='3d', num_policies=5)
        img = np.random.rand(32, 32, 32).astype(np.float32)
        result = aug(img)
        assert result.shape == img.shape
        assert result.min() >= 0 and result.max() <= 1

    def test_repr(self):
        aug = MRIAutoAugment(mode='2d', num_policies=8, num_ops=3)
        r = repr(aug)
        assert 'MRIAutoAugment' in r
        assert '2d' in r

    def test_top_level_access(self):
        assert vbai.MRIAutoAugment is MRIAutoAugment
