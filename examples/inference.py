"""
Vbai Inference Example
======================

This example shows how to load a trained model and make predictions.

Usage:
    python inference.py --model model.pt --image brain_scan.jpg
"""

import argparse
import vbai


def main():
    parser = argparse.ArgumentParser(description='Vbai Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--visualize', action='store_true',
                        help='Create attention visualization')
    parser.add_argument('--output', type=str, default='prediction.png',
                        help='Output visualization path')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Vbai Inference")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = vbai.load(args.model, device=args.device)
    print(f"  Model variant: {model.variant}")
    print(f"  Device: {args.device}")
    
    # Make prediction
    print(f"\nAnalyzing {args.image}...")
    result = model.predict(args.image, return_attention=args.visualize)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nDementia Classification:")
    print(f"  Class: {result.dementia_class}")
    print(f"  Confidence: {result.dementia_confidence:.1%}")
    print(f"  All probabilities:")
    for i, cls in enumerate(model.DEMENTIA_CLASSES):
        prob = result.dementia_probs[i].item()
        bar = "█" * int(prob * 20)
        print(f"    {cls:25s} {prob:6.1%} {bar}")
    
    print(f"\nTumor Detection:")
    print(f"  Class: {result.tumor_class}")
    print(f"  Confidence: {result.tumor_confidence:.1%}")
    print(f"  All probabilities:")
    for i, cls in enumerate(model.TUMOR_CLASSES):
        prob = result.tumor_probs[i].item()
        bar = "█" * int(prob * 20)
        print(f"    {cls:15s} {prob:6.1%} {bar}")
    
    # Visualize if requested
    if args.visualize:
        print(f"\nCreating visualization...")
        from PIL import Image
        image = Image.open(args.image)
        vis = vbai.VisualizationManager()
        vis.visualize(image, result, save=True, filename=args.output)
        print(f"  Saved to: {args.output}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
