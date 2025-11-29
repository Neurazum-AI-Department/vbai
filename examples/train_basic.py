"""
Vbai Basic Training Example
===========================

This example shows how to train a multi-task brain MRI model
using the Vbai library.

Usage:
    python train_basic.py --dementia_path ./data/dementia --tumor_path ./data/tumor
"""

import argparse
import vbai


def main():
    parser = argparse.ArgumentParser(description='Train Vbai model')
    parser.add_argument('--dementia_path', type=str, default=None,
                        help='Path to dementia dataset')
    parser.add_argument('--tumor_path', type=str, default=None,
                        help='Path to tumor dataset')
    parser.add_argument('--variant', type=str, default='q', choices=['f', 'q'],
                        help='Model variant (f=fast, q=quality)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output', type=str, default='vbai_model.pt',
                        help='Output model path')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Vbai Training")
    print("=" * 60)
    
    # Create model
    print(f"\nCreating model (variant='{args.variant}')...")
    model = vbai.MultiTaskBrainModel(variant=args.variant)
    print(model)
    
    # Create dataset
    print("\nLoading dataset...")
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
    print(f"  Dementia samples: {stats['dementia_samples']}")
    print(f"  Tumor samples: {stats['tumor_samples']}")
    
    # Create trainer with callbacks
    print("\nSetting up trainer...")
    callbacks = [
        vbai.EarlyStopping(monitor='val_loss', patience=5, verbose=True),
        vbai.ModelCheckpoint(
            filepath='checkpoints/best_model.pt',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    trainer = vbai.Trainer(
        model=model,
        lr=args.lr,
        device=args.device,
        callbacks=callbacks
    )
    
    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    history = trainer.fit(
        train_data=train_dataset,
        val_data=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    
    # Save final model
    print(f"\nSaving model to {args.output}...")
    trainer.save(args.output)
    
    # Plot training history
    print("\nSaving training history plot...")
    vbai.plot_training_history(history, save_path='training_history.png', show=False)
    
    print("\nTraining complete!")
    print(f"  Final train loss: {history.train_loss[-1]:.4f}")
    if history.val_loss:
        print(f"  Final val loss: {history.val_loss[-1]:.4f}")
    print(f"  Model saved to: {args.output}")


if __name__ == '__main__':
    main()
