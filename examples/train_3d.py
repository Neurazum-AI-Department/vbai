"""
Vbai 3D NIfTI Training Example
================================

This example shows how to train a 3D brain MRI model using NIfTI files
with the Vbai library.

Usage:
    python train_3d.py --data_path ./data/alzheimer_3d
    python train_3d.py --data_path ./data/alzheimer_3d --variant f --epochs 10

Prerequisites:
    pip install vbai[nifti]

Dataset Structure:
    data/alzheimer_3d/
        CN/
            subject_001.nii.gz
            subject_002.nii.gz
        MCI/
            subject_003.nii.gz
        AD/
            subject_004.nii.gz
"""

import argparse
import vbai


def main():
    parser = argparse.ArgumentParser(description='Train Vbai 3D Model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to NIfTI dataset (with class subdirectories)')
    parser.add_argument('--variant', type=str, default='q', choices=['f', 'q'],
                        help='Model variant (f=fast, q=quality)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (keep small for 3D, e.g. 2-8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--input_shape', type=int, nargs=3, default=[96, 96, 96],
                        help='Volume input shape D H W (default: 96 96 96)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output', type=str, default='vbai_3d_model.pt',
                        help='Output model path')
    parser.add_argument('--classes', type=str, nargs='+', default=None,
                        help='Class names (default: auto-detect from folders)')
    args = parser.parse_args()

    input_shape = tuple(args.input_shape)

    print("=" * 60)
    print("Vbai 3D NIfTI Training")
    print("=" * 60)

    # ── Create dataset ──
    print("\nLoading NIfTI dataset...")
    dataset = vbai.NIfTIDataset(
        root=args.data_path,
        target_shape=input_shape,
        is_training=True,
    )

    class_names = args.classes or dataset.classes
    num_classes = len(class_names)
    class_counts = dataset.get_class_counts()

    print(f"  Classes: {class_names}")
    print(f"  Samples: {len(dataset)}")
    for cls, count in class_counts.items():
        print(f"    {cls}: {count}")

    # ── Create dataloaders ──
    train_loader, val_loader = vbai.create_3d_dataloaders(
        root=args.data_path,
        target_shape=input_shape,
        batch_size=args.batch_size,
        val_split=0.2,
    )

    # ── Create model ──
    print(f"\nCreating 3D model (variant='{args.variant}')...")
    task_name = 'classification'
    model = vbai.MultiTask3DBrainModel(
        variant=args.variant,
        tasks={task_name: num_classes},
        input_shape=input_shape,
    )
    print(model)

    # ── Class weights for imbalanced data ──
    class_weights = dataset.get_class_weights()
    print(f"\nClass weights: {class_weights}")

    # ── Setup callbacks ──
    callbacks = [
        vbai.EarlyStopping(monitor='val_loss', patience=10, verbose=True),
        vbai.ModelCheckpoint(
            filepath='checkpoints_3d/best_model.pt',
            monitor='val_loss',
            save_best_only=True,
        )
    ]

    # ── Create trainer ──
    trainer = vbai.Trainer3D(
        model=model,
        lr=args.lr,
        device=args.device,
        callbacks=callbacks,
        class_weights={task_name: class_weights},
        mixed_precision=True,
    )

    # ── Train ──
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    history = trainer.fit(
        train_data=train_loader,
        val_data=val_loader,
        epochs=args.epochs,
        verbose=1,
    )

    # ── Save final model ──
    print(f"\nSaving model to {args.output}...")
    trainer.save(args.output)

    # ── Print summary ──
    print("\nTraining complete!")
    print(f"  Final train loss: {history.train_loss[-1]:.4f}")
    if history.val_loss:
        print(f"  Final val loss: {history.val_loss[-1]:.4f}")
    for task, accs in history.task_acc.items():
        print(f"  Final {task} acc: {accs[-1]:.4f}")
    print(f"  Model saved to: {args.output}")

    # ── Example: Load and predict ──
    print("\n--- Inference Example ---")
    print("  model = vbai.load_3d('vbai_3d_model.pt', device='cuda')")
    print(f"  result = model.predict('scan.nii.gz', task='{task_name}',")
    print(f"                         class_names={class_names})")
    print("  print(result.predicted_class, result.confidence)")


if __name__ == '__main__':
    main()
