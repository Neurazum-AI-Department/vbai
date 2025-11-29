"""
Vbai Command Line Interface
===========================

Provides command-line tools for training and inference.

Usage:
    vbai-train --dementia_path ./data/dementia --tumor_path ./data/tumor
    vbai-predict --model model.pt --image brain_scan.jpg
"""

import argparse
import sys


def train_cli():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(
        description='Train a Vbai multi-task brain MRI model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vbai-train --dementia_path ./data/dementia --tumor_path ./data/tumor
  vbai-train --variant f --epochs 5 --batch_size 64
  vbai-train --config config.yaml
        """
    )
    
    # Data arguments
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--dementia_path', type=str, default=None,
                           help='Path to dementia dataset')
    data_group.add_argument('--tumor_path', type=str, default=None,
                           help='Path to tumor dataset')
    data_group.add_argument('--val_split', type=float, default=0.2,
                           help='Validation split ratio (default: 0.2)')
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--variant', type=str, default='q', choices=['f', 'q'],
                            help='Model variant: f=fast, q=quality (default: q)')
    model_group.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate (default: 0.5)')
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--epochs', type=int, default=10,
                            help='Number of epochs (default: 10)')
    train_group.add_argument('--batch_size', type=int, default=32,
                            help='Batch size (default: 32)')
    train_group.add_argument('--lr', type=float, default=0.0005,
                            help='Learning rate (default: 0.0005)')
    train_group.add_argument('--device', type=str, default='auto',
                            help='Device: auto, cuda, cpu (default: auto)')
    
    # Output arguments
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output', type=str, default='vbai_model.pt',
                             help='Output model path (default: vbai_model.pt)')
    output_group.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                             help='Checkpoint directory')
    
    # Other
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML/JSON config file')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level: 0, 1, 2 (default: 1)')
    
    args = parser.parse_args()
    
    # Import here to avoid slow startup
    import torch
    import vbai
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 60)
    print("Vbai Training CLI")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Variant: {args.variant}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    # Check data paths
    if args.dementia_path is None and args.tumor_path is None:
        print("Error: At least one of --dementia_path or --tumor_path required")
        sys.exit(1)
    
    # Create model
    print("\nCreating model...")
    model = vbai.MultiTaskBrainModel(
        variant=args.variant,
        dropout=args.dropout
    )
    
    # Create dataset
    print("Loading dataset...")
    train_dataset = vbai.UnifiedMRIDataset(
        dementia_path=args.dementia_path,
        tumor_path=args.tumor_path,
        is_training=True
    )
    
    val_dataset = vbai.UnifiedMRIDataset(
        dementia_path=args.dementia_path,
        tumor_path=args.tumor_path,
        is_training=False
    )
    
    stats = train_dataset.get_statistics()
    print(f"  Total samples: {stats['total_samples']}")
    
    # Setup callbacks
    callbacks = [
        vbai.EarlyStopping(monitor='val_loss', patience=5),
        vbai.ModelCheckpoint(
            filepath=f"{args.checkpoint_dir}/best.pt",
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Create trainer
    trainer = vbai.Trainer(
        model=model,
        lr=args.lr,
        device=device,
        callbacks=callbacks
    )
    
    # Train
    print(f"\nTraining...")
    history = trainer.fit(
        train_data=train_dataset,
        val_data=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose
    )
    
    # Save
    trainer.save(args.output)
    print(f"\nModel saved to {args.output}")


def predict_cli():
    """Command-line interface for prediction."""
    parser = argparse.ArgumentParser(
        description='Make predictions with a trained Vbai model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  vbai-predict --model model.pt --image brain_scan.jpg
  vbai-predict --model model.pt --image scan.jpg --visualize --output result.png
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, cpu')
    parser.add_argument('--visualize', action='store_true',
                       help='Create attention visualization')
    parser.add_argument('--output', type=str, default='prediction.png',
                       help='Output visualization path')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    
    args = parser.parse_args()
    
    import torch
    import json
    import vbai
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Load model
    model = vbai.load(args.model, device=device)
    
    # Predict
    result = model.predict(args.image, return_attention=args.visualize)
    
    # Output results
    if args.json:
        output = {
            'dementia': {
                'class': result.dementia_class,
                'confidence': result.dementia_confidence,
                'probabilities': {
                    cls: result.dementia_probs[i].item()
                    for i, cls in enumerate(model.DEMENTIA_CLASSES)
                }
            },
            'tumor': {
                'class': result.tumor_class,
                'confidence': result.tumor_confidence,
                'probabilities': {
                    cls: result.tumor_probs[i].item()
                    for i, cls in enumerate(model.TUMOR_CLASSES)
                }
            }
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nDementia: {result.dementia_class} ({result.dementia_confidence:.1%})")
        print(f"Tumor: {result.tumor_class} ({result.tumor_confidence:.1%})")
    
    # Visualize
    if args.visualize:
        from PIL import Image
        image = Image.open(args.image)
        vis = vbai.VisualizationManager()
        vis.visualize(image, result, save=True, filename=args.output)
        print(f"\nVisualization saved to {args.output}")


if __name__ == '__main__':
    # For testing
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        train_cli()
    elif len(sys.argv) > 1 and sys.argv[1] == 'predict':
        sys.argv.pop(1)
        predict_cli()
    else:
        print("Usage: python -m vbai.cli [train|predict] [options]")
