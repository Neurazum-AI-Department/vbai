"""
Vbai Advanced Training Example
==============================

This example demonstrates advanced training features:
- Custom configurations
- Learning rate scheduling
- Multiple callbacks
- Mixed precision training
- Custom loss functions

Usage:
    python train_advanced.py
"""

import torch
import vbai
from vbai.training import MultiTaskLoss, UncertaintyWeightedLoss


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    
    # Use preset or create custom config
    # config = vbai.get_default_config('quality')
    
    model_config = vbai.ModelConfig(
        variant='q',
        dropout=0.3,
        use_edge_branch=True,
    )
    
    training_config = vbai.TrainingConfig(
        epochs=30,
        batch_size=16,
        lr=0.0001,
        weight_decay=0.0001,
        scheduler='cosine',
        early_stopping_patience=10,
        augmentation_strength='strong',
        label_smoothing=0.1,
    )
    
    # Save config for reproducibility
    model_config.save('configs/model_config.yaml')
    training_config.save('configs/training_config.yaml')
    
    # =========================================================================
    # Model Setup
    # =========================================================================
    
    print("Creating model...")
    model = vbai.MultiTaskBrainModel(
        variant=model_config.variant,
        dropout=model_config.dropout,
        use_edge_branch=model_config.use_edge_branch,
    )
    
    # Print model info
    params = model.count_parameters()
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    
    # =========================================================================
    # Dataset Setup
    # =========================================================================
    
    print("\nLoading datasets...")
    
    # Custom transforms with strong augmentation
    from vbai.data import get_train_transforms, get_val_transforms
    
    train_transform = get_train_transforms(
        augmentation_strength=training_config.augmentation_strength
    )
    val_transform = get_val_transforms()
    
    train_dataset = vbai.UnifiedMRIDataset(
        dementia_path='./data/dementia/train',
        tumor_path='./data/tumor/train',
        transform=train_transform,
        is_training=True
    )
    
    val_dataset = vbai.UnifiedMRIDataset(
        dementia_path='./data/dementia/val',
        tumor_path='./data/tumor/val',
        transform=val_transform,
        is_training=False
    )
    
    # Create dataloaders
    train_loader = train_dataset.get_dataloader(
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    val_loader = val_dataset.get_dataloader(
        batch_size=training_config.batch_size,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory
    )
    
    stats = train_dataset.get_statistics()
    print(f"  Training samples: {stats['total_samples']}")
    print(f"  Dementia distribution: {stats['dementia_distribution']}")
    print(f"  Tumor distribution: {stats['tumor_distribution']}")
    
    # =========================================================================
    # Loss Function
    # =========================================================================
    
    # Option 1: Standard multi-task loss with label smoothing
    loss_fn = MultiTaskLoss(
        dementia_weight=training_config.dementia_loss_weight,
        tumor_weight=training_config.tumor_loss_weight,
        label_smoothing=training_config.label_smoothing,
    )
    
    # Option 2: Uncertainty-weighted loss (learns task weights automatically)
    # loss_fn = UncertaintyWeightedLoss()
    
    # Option 3: Focal loss for class imbalance
    # from vbai.training import MultiTaskFocalLoss
    # loss_fn = MultiTaskFocalLoss(gamma=2.0)
    
    # =========================================================================
    # Optimizer & Scheduler
    # =========================================================================
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.lr,
        weight_decay=training_config.weight_decay
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.epochs,
        eta_min=1e-6
    )
    
    # =========================================================================
    # Callbacks
    # =========================================================================
    
    callbacks = [
        # Early stopping
        vbai.EarlyStopping(
            monitor='val_loss',
            patience=training_config.early_stopping_patience,
            min_delta=training_config.early_stopping_min_delta,
            verbose=True
        ),
        
        # Save best model
        vbai.ModelCheckpoint(
            filepath='checkpoints/best_model.pt',
            monitor='val_loss',
            save_best_only=True,
            verbose=True
        ),
        
        # Save every 5 epochs
        vbai.ModelCheckpoint(
            filepath='checkpoints/model_epoch_{epoch:02d}.pt',
            save_best_only=False,
            verbose=False
        ),
    ]
    
    # Optional: TensorBoard logging
    # from vbai.training.callbacks import TensorBoardLogger
    # callbacks.append(TensorBoardLogger(log_dir='./runs/experiment1'))
    
    # =========================================================================
    # Training
    # =========================================================================
    
    print("\nSetting up trainer...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    trainer = vbai.Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        callbacks=callbacks,
        scheduler=scheduler,
    )
    
    print(f"\nStarting training for {training_config.epochs} epochs...")
    print("=" * 70)
    
    history = trainer.fit(
        train_data=train_loader,
        val_data=val_loader,
        epochs=training_config.epochs,
        verbose=1
    )
    
    # =========================================================================
    # Results
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    # Save final model
    trainer.save('models/final_model.pt')
    
    # Plot and save training history
    vbai.plot_training_history(
        history,
        save_path='results/training_history.png',
        show=False
    )
    
    # Print best results
    best_epoch = history.val_loss.index(min(history.val_loss))
    print(f"\nBest Results (Epoch {best_epoch + 1}):")
    print(f"  Val Loss: {history.val_loss[best_epoch]:.4f}")
    print(f"  Val Dementia Acc: {history.val_dementia_acc[best_epoch]:.2%}")
    print(f"  Val Tumor Acc: {history.val_tumor_acc[best_epoch]:.2%}")
    
    # =========================================================================
    # Evaluation
    # =========================================================================
    
    print("\nFinal Evaluation...")
    
    # Load best model
    best_model = vbai.load('checkpoints/best_model.pt', device=device)
    
    # Evaluate on validation set
    eval_trainer = vbai.Trainer(model=best_model, device=device)
    metrics = eval_trainer.evaluate(val_loader)
    
    print(f"  Final Val Loss: {metrics['loss']:.4f}")
    print(f"  Final Dementia Accuracy: {metrics['dementia_acc']:.2%}")
    print(f"  Final Tumor Accuracy: {metrics['tumor_acc']:.2%}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
