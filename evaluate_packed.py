import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from wrn_mcdonnell import WRN_McDonnell
from main import create_dataset

import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description='Binary Wide Residual Networks')
    # Model options
    parser.add_argument('--depth', default=20, type=int)
    parser.add_argument('--width', default=1, type=float)
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--dataroot', default='.', type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    return parser.parse_args()


def create_mat_dataset(args, train):
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: 255 * x)
    ])
    if train:
        transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])
    return getattr(datasets, args.dataset)(args.dataroot, train=train, download=True, transform=transform)


def main():
    args = parse_args()
    num_classes = 10 if args.dataset == 'CIFAR10' else 100

    have_cuda = torch.cuda.is_available()

    def cast(x):
        return x.cuda() if have_cuda else x

    model = WRN_McDonnell(args.depth, args.width, num_classes)

    # For loading matconvnet checkpoint
    if '.mat' in args.checkpoint:
        from scipy.io import loadmat

        checkpoint = loadmat(args.checkpoint)

        # get model parameter names and move conv_last to end
        named_params = list(model.state_dict().items())
        # move conv_last to end of list
        if 'conv_last' in named_params[1][0]:
            named_params.append(named_params.pop(1))

        # initialize layer counters
        conv_cnt = bn_mean_cnt = bn_var_cnt = 0
        weights_unpacked = {}
        for name, param in named_params:
            if 'conv' in name:
                scale = checkpoint['LayerWeights'][conv_cnt, 0]
                signed = checkpoint['BinaryWeights'][0, conv_cnt].astype(np.int) * 2 - 1
                # swap conv filter axes
                signed = np.transpose(signed, (3, 2, 0, 1))
                weights_unpacked[name] = torch.from_numpy(signed).float() * scale
                conv_cnt += 1
            elif 'mean' in name:
                weights_unpacked[name] = torch.from_numpy(checkpoint['Moments'][bn_mean_cnt+1, 0][:, 0])
                bn_mean_cnt += 1
            elif 'var' in name:
                weights_unpacked[name] = torch.from_numpy(checkpoint['Moments'][bn_var_cnt+1, 0][:, 1]**2)
                bn_var_cnt += 1

        # initialize bn_last weight and bias
        weights_unpacked['bn_last.weight'] = torch.ones_like(model.bn_last.weight)
        weights_unpacked['bn_last.bias'] = torch.zeros_like(model.bn_last.bias)

        # initialize input bn parameters
        bn_inp_mean = cast(torch.from_numpy(checkpoint['Moments'][0, 0][:, 0]))
        bn_inp_var = cast(torch.from_numpy(checkpoint['Moments'][0, 0][:, 1]**2))
        bn_inp_scale = cast(torch.from_numpy(checkpoint['BNG'][0, 0]))
        bn_inp_offset = cast(torch.from_numpy(checkpoint['BNB'][0, 0]))

        # Create dataloader
        data_loader = DataLoader(create_mat_dataset(args, train=False), 256)
    else:
        checkpoint = torch.load(args.checkpoint)

        weights_unpacked = {}
        for k, w in checkpoint.items():
            if w.dtype == torch.uint8:
                # weights are packed with np.packbits function
                scale = np.sqrt(2 / (w.shape[1] * w.shape[2] * w.shape[3] * 8))
                signed = np.unpackbits(w, axis=1).astype(np.int) * 2 - 1
                weights_unpacked[k[7:]] = torch.from_numpy(signed).float() * scale
            else:
                weights_unpacked[k[7:]] = w

        # Create dataloader
        data_loader = DataLoader(create_dataset(args, train=False), 256)

    model.load_state_dict(weights_unpacked)
    model = cast(model)
    model.eval()

    correct = 0
    total = 0
    for inputs, targets in data_loader:
        with torch.no_grad():

            inputs, targets = cast(inputs), cast(targets)

            # apply input bn
            if '.mat' in args.checkpoint:
                inputs = F.relu(F.batch_norm(inputs, bn_inp_mean, bn_inp_var, bn_inp_scale, bn_inp_offset))

            outputs = model.forward(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(100.*correct/total)


if __name__ == '__main__':
    main()
