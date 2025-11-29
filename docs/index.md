# Vbai Documentation

Welcome to the Vbai documentation!

## Contents

1. [Getting Started](getting_started.md)
2. [API Reference](api_reference.md)
3. [Examples](examples.md)
4. [Configuration](configuration.md)

## Quick Links

- [GitHub Repository](https://github.com/Neurazum-AI-Department/vbai)
- [PyPI Package](https://pypi.org/project/vbai/)
- [Issue Tracker](https://github.com/healfuture/vbai/issues)

## Installation

```bash
pip install vbai
```

## Minimal Example

```python
import vbai

# Create and train a model
model = vbai.MultiTaskBrainModel(variant='q')
dataset = vbai.UnifiedMRIDataset(
    dementia_path='./data/dementia',
    tumor_path='./data/tumor'
)

trainer = vbai.Trainer(model=model, epochs=10)
trainer.fit(dataset)
trainer.save('model.pt')

# Make predictions
model = vbai.load('model.pt')
result = model.predict('brain_scan.jpg')
print(result.dementia_class, result.tumor_class)
```
