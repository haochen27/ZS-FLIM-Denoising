import argparse
from train import train_model, train_FLIM_model
from test import test_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the DNFLIM model.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=800, help='Number of epochs')
    parser.add_argument('--root', type=str, required=True, help='Root directory for the dataset')
    parser.add_argument('--noise_levels', type=int, default=1, nargs='+', required=False, help='Noise levels in the dataset')
    parser.add_argument('--types', type=str, nargs='*', default=None, help='Specific types or categories of data to use')
    parser.add_argument('--train', action='store_true', help='Trigger training')
    parser.add_argument('--test', action='store_true', help='Trigger testing')
    parser.add_argument('--pretrained_model', type=str, default='./model2_mix/best_dnflim.pth', help='Pretrained model to test')
    parser.add_argument('--FLIM', action='store_true', help='Zero shot training for FLIM')
    parser.add_argument('--alpha', type=float, default=3e-3, help='Alpha value for loss function')
    args = parser.parse_args()

    if args.train:
        # Call DNFLIM training function (implementation assumed elsewhere)
        train_model(root=args.root,
                    noise_levels=args.noise_levels,
                    types=args.types,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr)
    
    if args.test:
        # Call DNFLIM testing function (implementation assumed elsewhere)
        test_model(batch_size=args.batch_size,
                   root=args.root,
                   noise_levels=args.noise_levels,
                   types=args.types,
                   pretrained_model=args.pretrained_model)
    
    if args.FLIM:
        # Prepare configuration for FLIM training
        config = argparse.Namespace(
            root=args.root,
            epochs=args.epochs,
            batch_size=1,  # As specified for FLIM training
            lr=args.lr,
            weight_decay=1e-5,
            types=args.types,
            num_augmentations=1,
            alpha=args.alpha,
            ssim_weight=1e-4,
            tv_weight=0.5e-6,
            base_channels=32,
            checkpoint_interval=100,
            log_interval=10,
            intensity_model_path=args.pretrained_model,  # Using the provided pretrained model path
        )
        train_FLIM_model(config)

