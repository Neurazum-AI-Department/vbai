"""
Tests for VbaiSegNet3D, VbaiProgressionNet, and all supporting modules.
"""

import pytest
import torch
import numpy as np

import vbai
from vbai.models.segmentation3d import (
    VbaiSegNet3D, SEBlock3D, CBAM3D, ResBlock3D, ASPP3D, AttentionGate3D,
    EncoderBlock, DecoderBlock,
)
from vbai.models.progression3d import (
    VbaiProgressionNet, MRIEncoder3D, TabularEncoder, CrossModalFusion,
    ClassificationHead, ProgressionHead,
)
from vbai.training.segmentation_losses import (
    DiceLoss, MulticlassDiceLoss, FocalLoss, TumorSegmentationLoss,
    TissueSegmentationLoss, DeepSupervisionLoss,
)
from vbai.training.progression_losses import (
    FocalLoss3Class, ProgressionLoss, InfoNCELoss, VbaiProgressionLoss,
)
from vbai.data.progression_dataset import TabularNormalizer, BIOMARKER_FEATURES, N_FEATURES
from vbai.configs.segmentation_config import get_segmentation_config
from vbai.configs.progression_config import get_progression_config


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation model tests
# ──────────────────────────────────────────────────────────────────────────────

class TestVbaiSegNet3D:

    def test_binary_forward(self):
        model = VbaiSegNet3D(in_channels=1, out_channels=1)
        x = torch.randn(1, 1, 96, 96, 96)
        out = model(x)
        assert out.shape == (1, 1, 96, 96, 96)

    def test_multiclass_forward(self):
        model = VbaiSegNet3D(in_channels=1, out_channels=3)
        x = torch.randn(1, 1, 96, 96, 96)
        out = model(x)
        assert out.shape == (1, 3, 96, 96, 96)

    def test_two_channel_input(self):
        model = VbaiSegNet3D(in_channels=2, out_channels=1)
        x = torch.randn(1, 2, 96, 96, 96)
        out = model(x)
        assert out.shape == (1, 1, 96, 96, 96)

    def test_deep_supervision(self):
        model = VbaiSegNet3D(in_channels=1, out_channels=1, use_deep_supervision=True)
        x = torch.randn(1, 1, 96, 96, 96)
        main, aux = model(x, return_aux=True)
        assert main.shape == (1, 1, 96, 96, 96)
        assert len(aux) == 3  # ds3, ds2, ds1

    def test_no_deep_supervision(self):
        model = VbaiSegNet3D(in_channels=1, out_channels=1, use_deep_supervision=False)
        x = torch.randn(1, 1, 96, 96, 96)
        out = model(x)
        assert out.shape == (1, 1, 96, 96, 96)

    def test_batch_size_2(self):
        model = VbaiSegNet3D(in_channels=1, out_channels=1)
        x = torch.randn(2, 1, 96, 96, 96)
        out = model(x)
        assert out.shape == (2, 1, 96, 96, 96)

    def test_parameter_count(self):
        model = VbaiSegNet3D(in_channels=1, out_channels=1)
        p = model.count_parameters()
        assert p['total'] > 0
        assert p['trainable'] == p['total']

    def test_predict_volume(self):
        model = VbaiSegNet3D(in_channels=1, out_channels=1, use_deep_supervision=False)
        vol = torch.randn(1, 1, 64, 64, 64)
        probs = model.predict_volume(vol, patch_size=(64, 64, 64))
        assert probs.shape == (1, 1, 64, 64, 64)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_invalid_channel_mult(self):
        with pytest.raises(ValueError):
            VbaiSegNet3D(channel_mult=(1, 2, 4))  # too few elements

    def test_save_load(self, tmp_path):
        model = VbaiSegNet3D(in_channels=1, out_channels=1)
        model.eval()
        path = str(tmp_path / 'seg.pth')
        model.save(path)
        model2 = VbaiSegNet3D.load(path)
        x = torch.randn(1, 1, 96, 96, 96)
        with torch.no_grad():
            torch.testing.assert_close(model(x), model2(x))


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation atomic blocks
# ──────────────────────────────────────────────────────────────────────────────

class TestSegmentationBlocks:

    def test_se_block(self):
        m = SEBlock3D(32)
        x = torch.randn(2, 32, 8, 8, 8)
        assert m(x).shape == x.shape

    def test_cbam(self):
        m = CBAM3D(32)
        x = torch.randn(2, 32, 8, 8, 8)
        assert m(x).shape == x.shape

    def test_res_block(self):
        m = ResBlock3D(16, 32, stride=2)
        x = torch.randn(2, 16, 16, 16, 16)
        assert m(x).shape == (2, 32, 8, 8, 8)

    def test_aspp(self):
        m = ASPP3D(64, 128)
        x = torch.randn(2, 64, 6, 6, 6)
        assert m(x).shape == (2, 128, 6, 6, 6)

    def test_attention_gate(self):
        m = AttentionGate3D(skip_ch=64, gate_ch=128)
        skip = torch.randn(2, 64, 12, 12, 12)
        gate = torch.randn(2, 128, 6, 6, 6)
        out = m(skip, gate)
        assert out.shape == skip.shape


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation loss tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSegmentationLosses:

    def test_dice_loss(self):
        fn = DiceLoss()
        logits = torch.randn(2, 1, 32, 32, 32, requires_grad=True)
        mask = (torch.rand(2, 1, 32, 32, 32) > 0.7).float()
        loss = fn(logits, mask)
        assert loss.item() >= 0.0
        assert loss.requires_grad  # gradient flows through sigmoid(logits)

    def test_multiclass_dice(self):
        fn = MulticlassDiceLoss()
        logits = torch.randn(2, 3, 32, 32, 32)
        mask = torch.rand(2, 3, 32, 32, 32)
        loss = fn(logits, mask)
        assert loss.item() >= 0.0

    def test_tumor_loss(self):
        fn = TumorSegmentationLoss()
        logits = torch.randn(2, 1, 32, 32, 32)
        mask = (torch.rand(2, 1, 32, 32, 32) > 0.8).float()
        loss = fn(logits, mask)
        assert 0.0 <= loss.item() <= 2.0

    def test_tissue_loss(self):
        fn = TissueSegmentationLoss()
        logits = torch.randn(2, 3, 32, 32, 32)
        mask = torch.rand(2, 3, 32, 32, 32)
        loss = fn(logits, mask)
        assert loss.item() >= 0.0

    def test_deep_supervision_loss(self):
        base = TumorSegmentationLoss()
        fn = DeepSupervisionLoss(base)
        logits = torch.randn(2, 1, 32, 32, 32)
        mask = (torch.rand(2, 1, 32, 32, 32) > 0.8).float()
        aux = [torch.randn(2, 1, 4, 4, 4), torch.randn(2, 1, 8, 8, 8)]
        loss = fn(logits, mask, aux)
        loss_no_aux = fn(logits, mask, None)
        assert loss.item() > 0.0
        assert loss.item() != loss_no_aux.item()


# ──────────────────────────────────────────────────────────────────────────────
# Progression model tests
# ──────────────────────────────────────────────────────────────────────────────

class TestVbaiProgressionNet:

    def test_full_multimodal(self):
        model = VbaiProgressionNet()
        mri = torch.randn(2, 1, 96, 96, 96)
        tab = torch.randn(2, 26)
        out = model(mri=mri, tab=tab)
        assert 'fused_logits' in out
        assert 'mri_logits' in out
        assert 'tab_logits' in out
        assert 'progression' in out
        assert out['fused_logits'].shape == (2, 3)
        assert out['progression']['will_progress_logits'].shape == (2, 1)
        assert out['progression']['time_to_conversion'].shape == (2, 1)
        assert out['progression']['time_distribution'].shape == (2, 24)

    def test_mri_only(self):
        model = VbaiProgressionNet()
        mri = torch.randn(2, 1, 96, 96, 96)
        out = model(mri=mri)
        assert 'fused_logits' in out
        assert out['fused_logits'].shape == (2, 3)
        assert 'progression' not in out

    def test_tab_only(self):
        model = VbaiProgressionNet()
        tab = torch.randn(2, 26)
        out = model(tab=tab)
        assert out['fused_logits'].shape == (2, 3)

    def test_time_to_conversion_bounded(self):
        model = VbaiProgressionNet(max_progression_months=120)
        mri = torch.randn(2, 1, 96, 96, 96)
        tab = torch.randn(2, 26)
        out = model(mri=mri, tab=tab)
        ttc = out['progression']['time_to_conversion']
        assert ttc.min().item() >= 0.0
        assert ttc.max().item() <= 120.0

    def test_time_distribution_sums_to_one(self):
        model = VbaiProgressionNet()
        out = model(mri=torch.randn(1, 1, 96, 96, 96), tab=torch.randn(1, 26))
        dist = out['progression']['time_distribution']
        assert torch.allclose(dist.sum(dim=-1), torch.ones(1), atol=1e-5)

    def test_predict_api(self):
        model = VbaiProgressionNet()
        result = model.predict(
            mri=torch.randn(1, 1, 96, 96, 96),
            tab=torch.randn(26),
        )
        assert 'class_name' in result
        assert result['class_name'] in ['CN', 'MCI', 'AD']
        assert 0.0 <= result['confidence'] <= 1.0
        assert 'progression' in result
        assert result['progression']['risk_category'] in ['Low', 'Moderate', 'High']

    def test_predict_mri_only_no_progression(self):
        model = VbaiProgressionNet()
        result = model.predict(mri=torch.randn(1, 1, 96, 96, 96))
        assert 'class_name' in result
        assert 'progression' not in result

    def test_parameter_count(self):
        model = VbaiProgressionNet()
        p = model.count_parameters()
        assert p['total'] > 1_000_000  # should be ~16M

    def test_save_load(self, tmp_path):
        model = VbaiProgressionNet()
        model.eval()
        path = str(tmp_path / 'prog.pth')
        model.save(path)
        model2 = VbaiProgressionNet.load(path)
        mri = torch.randn(1, 1, 96, 96, 96)
        tab = torch.randn(1, 26)
        with torch.no_grad():
            out1 = model(mri=mri, tab=tab)['fused_logits']
            out2 = model2(mri=mri, tab=tab)['fused_logits']
        torch.testing.assert_close(out1, out2)


# ──────────────────────────────────────────────────────────────────────────────
# Progression loss tests
# ──────────────────────────────────────────────────────────────────────────────

class TestProgressionLosses:

    def test_focal_3class(self):
        fn = FocalLoss3Class()
        logits = torch.randn(4, 3, requires_grad=True)
        labels = torch.randint(0, 3, (4,))
        loss = fn(logits, labels)
        assert loss.item() > 0.0
        assert loss.requires_grad

    def test_info_nce(self):
        fn = InfoNCELoss(temperature=0.1)
        zm = torch.randn(4, 128)
        zm = zm / zm.norm(dim=-1, keepdim=True)
        zt = torch.randn(4, 128)
        zt = zt / zt.norm(dim=-1, keepdim=True)
        loss = fn(zm, zt)
        assert loss.item() > 0.0

    def test_progression_loss_no_mci(self):
        fn = ProgressionLoss()
        prog_out = {
            'will_progress_logits': torch.randn(4, 1),
            'time_to_conversion': torch.rand(4, 1) * 120,
        }
        loss = fn(
            prog_out,
            will_progress=torch.zeros(4),
            progression_months=torch.zeros(4),
            has_progression=torch.zeros(4, dtype=torch.bool),
        )
        assert loss.item() == 0.0  # no valid MCI samples

    def test_vbai_progression_loss_composite(self):
        fn = VbaiProgressionLoss()
        model = VbaiProgressionNet()
        out = model(mri=torch.randn(2, 1, 96, 96, 96), tab=torch.randn(2, 26))
        targets = {
            'labels': torch.tensor([0, 1]),
            'has_progression': torch.tensor([False, True]),
            'will_progress': torch.tensor([0.0, 1.0]),
            'progression_months': torch.tensor([0.0, 24.0]),
        }
        losses = fn(out, targets)
        assert 'total' in losses
        assert losses['total'].item() > 0.0
        for k in ['cls_fused', 'cls_mri', 'cls_tab', 'progression', 'contrastive']:
            assert k in losses


# ──────────────────────────────────────────────────────────────────────────────
# TabularNormalizer tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTabularNormalizer:

    @pytest.fixture
    def records(self):
        base = {f: None for f in BIOMARKER_FEATURES}
        rows = []
        for i in range(20):
            r = {**base,
                 'Age': 70.0 + i,
                 'MMSE': 28.0 - i * 0.3,
                 'CDRSB': i * 0.1,
                 'APOE4_count': i % 3}
            rows.append(r)
        return rows

    def test_fit_transform_shape(self, records):
        norm = TabularNormalizer()
        norm.fit(records)
        vec = norm.transform(records[0])
        assert vec.shape == (N_FEATURES * 2,)

    def test_masks_correct(self, records):
        norm = TabularNormalizer()
        norm.fit(records)
        vec = norm.transform(records[0])
        masks = vec[N_FEATURES:]
        # Age, MMSE, CDRSB, APOE4_count present; rest NaN → mask=0
        assert masks[0] == 1.0  # Age
        assert masks[1] == 0.0  # Sex (None)
        assert masks[2] == 1.0  # MMSE

    def test_no_warning_all_nan(self, records):
        import warnings
        norm = TabularNormalizer()
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            norm.fit(records)  # Some features are all NaN → should not warn

    def test_save_load(self, records, tmp_path):
        norm = TabularNormalizer()
        norm.fit(records)
        path = str(tmp_path / 'norm.npz')
        norm.save(path)
        norm2 = TabularNormalizer.load(path)
        vec1 = norm.transform(records[0])
        vec2 = norm2.transform(records[0])
        np.testing.assert_array_almost_equal(vec1, vec2)


# ──────────────────────────────────────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────────────────────────────────────

class TestConfigs3D:

    def test_segmentation_config_tumor(self):
        cfg = get_segmentation_config('tumor')
        assert cfg.model.in_channels == 2
        assert cfg.model.out_channels == 1

    def test_segmentation_config_tissue(self):
        cfg = get_segmentation_config('tissue')
        assert cfg.model.out_channels == 3

    def test_segmentation_config_debug(self):
        cfg = get_segmentation_config('debug')
        assert cfg.training.epochs == 2
        assert not cfg.training.use_amp

    def test_segmentation_config_invalid(self):
        with pytest.raises(ValueError):
            get_segmentation_config('nonexistent')

    def test_progression_config_default(self):
        cfg = get_progression_config('default')
        assert cfg.model.num_classes == 3
        assert cfg.training.phase1_epochs == 40

    def test_progression_config_debug(self):
        cfg = get_progression_config('debug')
        assert cfg.training.phase1_epochs == 2

    def test_segmentation_config_save_load(self, tmp_path):
        cfg = get_segmentation_config('tumor')
        path = str(tmp_path / 'cfg.json')
        cfg.save(path)
        cfg2 = vbai.FullSegmentationConfig.load(path)
        assert cfg2.model.in_channels == cfg.model.in_channels

    def test_progression_config_save_load(self, tmp_path):
        cfg = get_progression_config('default')
        path = str(tmp_path / 'pcfg.json')
        cfg.save(path)
        cfg2 = vbai.FullProgressionConfig.load(path)
        assert cfg2.model.num_classes == 3


# ──────────────────────────────────────────────────────────────────────────────
# Visualization tests (no rendering, just ensure no crash)
# ──────────────────────────────────────────────────────────────────────────────

class TestVisualization:

    def test_compute_segmentation_metrics_binary(self):
        pred = np.random.rand(1, 32, 32, 32)
        gt = (np.random.rand(1, 32, 32, 32) > 0.5).astype(np.float32)
        m = vbai.compute_segmentation_metrics(pred, gt)
        assert 'dice' in m
        assert 0.0 <= m['dice'] <= 1.0

    def test_compute_segmentation_metrics_multiclass(self):
        pred = np.random.rand(3, 32, 32, 32)
        gt = np.random.rand(3, 32, 32, 32)
        m = vbai.compute_segmentation_metrics(pred, gt)
        assert 'dice_0' in m
        assert 'dice_1' in m
        assert 'dice_2' in m

    def test_create_report_figure(self):
        model = VbaiProgressionNet()
        result = model.predict(
            mri=torch.randn(1, 1, 96, 96, 96),
            tab=torch.randn(26),
        )
        fig = vbai.create_report_figure(result, subject_id='T001', scan_date='2024-01')
        assert fig is not None

    def test_report_mri_only(self):
        model = VbaiProgressionNet()
        result = model.predict(mri=torch.randn(1, 1, 96, 96, 96))
        fig = vbai.create_report_figure(result)
        assert fig is not None
