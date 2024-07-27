import argparse
from train import train_model
from test import test_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the DNFLIM model.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--root', type=str, required=True, help='Root directory for the dataset')
    parser.add_argument('--noise_levels', type=int, default=1,nargs='+', required=True, help='Noise levels in the dataset')
    parser.add_argument('--types', type=str, nargs='*', default=None, help='Specific types or categories of data to use')
    parser.add_argument('--train', action='store_true', help='Trigger training')
    parser.add_argument('--test', action='store_true', help='Trigger testing')
    parser.add_argument('--pretrained_model', type=str, default='./model/dnflim.pth', help='Pretrained model to test')
    args = parser.parse_args()

    if args.train:
        train_model(root=args.root, noise_levels=args.noise_levels, types=args.types, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    if args.test:
        test_model(batch_size=args.batch_size, root=args.root, noise_levels=args.noise_levels, types=args.types, pretrained_model=None)

