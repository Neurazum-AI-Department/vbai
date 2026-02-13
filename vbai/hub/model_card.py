"""
Model Card Generator for Vbai

Generates HuggingFace-compatible model cards (README.md) with YAML frontmatter.
"""

from typing import Optional, Dict, Any

import torch


def generate_model_card(
    model: torch.nn.Module,
    metrics: Optional[Dict[str, float]] = None,
    dataset_info: Optional[str] = None,
) -> str:
    """
    Generate a HuggingFace model card (README.md) for a vbai model.

    Args:
        model: The vbai model instance
        metrics: Optional dict of evaluation metrics (e.g., {'accuracy': 0.95})
        dataset_info: Optional description of the training dataset

    Returns:
        Model card as a markdown string with YAML frontmatter
    """
    from ..models.multitask3d import MultiTask3DBrainModel
    from ..models.multitask import MultiTaskBrainModel

    # Determine model properties
    if isinstance(model, MultiTask3DBrainModel):
        model_type = "3D"
        variant = getattr(model, 'variant', 'q')
        tasks = getattr(model, 'tasks', {})
        input_shape = getattr(model, 'input_shape', (96, 96, 96))
        tasks_str = ', '.join(f'{k} ({v} classes)' for k, v in tasks.items())
        input_desc = f"NIfTI volume {input_shape}"
    elif isinstance(model, MultiTaskBrainModel):
        model_type = "2D"
        variant = getattr(model, 'variant', 'q')
        tasks = getattr(model, 'tasks', ['dementia', 'tumor'])
        tasks_str = ', '.join(tasks)
        input_desc = "RGB image 224x224"
    else:
        model_type = "Unknown"
        variant = "?"
        tasks_str = "N/A"
        input_desc = "N/A"

    params = model.count_parameters() if hasattr(model, 'count_parameters') else {}
    total_params = params.get('total', sum(p.numel() for p in model.parameters()))

    # YAML frontmatter
    tags = ['brain-mri', 'medical-imaging', 'pytorch', 'vbai']
    if model_type == '3D':
        tags.extend(['nifti', '3d-cnn', 'volumetric'])
    else:
        tags.extend(['image-classification', '2d-cnn'])

    tags_yaml = '\n'.join(f'- {t}' for t in tags)

    card = f"""---
library_name: vbai
tags:
{tags_yaml}
license: mit
---

# Vbai {model_type} Brain MRI Model (variant='{variant}')

A {model_type} multi-task brain MRI analysis model built with the [Vbai](https://github.com/Neurazum-AI-Department/vbai) library.

## Model Details

| Property | Value |
|----------|-------|
| Model Type | {model_type} |
| Variant | {variant} |
| Tasks | {tasks_str} |
| Input | {input_desc} |
| Parameters | {total_params:,} |

## Usage

```python
import vbai

# Load from Hub
model = vbai.from_hub('{f"Neurazum/vbai-{model_type.lower()}-{variant}"}', device='cuda')
"""

    if model_type == "3D":
        card += """
# Predict on a NIfTI scan
result = model.predict('brain_scan.nii.gz', task='alzheimer',
                       class_names=['CN', 'MCI', 'AD'])
print(result.predicted_class, result.confidence)
```
"""
    else:
        card += """
# Predict on a brain MRI image
result = model.predict('brain_scan.jpg')
print(result.dementia_class, result.tumor_class)
```
"""

    if metrics:
        card += "\n## Evaluation Metrics\n\n"
        card += "| Metric | Value |\n|--------|-------|\n"
        for k, v in metrics.items():
            card += f"| {k} | {v:.4f} |\n"

    if dataset_info:
        card += f"\n## Training Data\n\n{dataset_info}\n"

    card += """
## Framework

Built with [Vbai](https://github.com/Neurazum-AI-Department/vbai) - Visual Brain AI Library by Neurazum.

```
pip install vbai
```
"""

    return card
