import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from utils.optimizer import LeeSeung, ANGD, ZOAHessB
from utils.log_utils import prepare_csv, write_csv
from utils.data_utils import *
from utils.visual_utils import *

set_seed(3407)

def train():
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_type', type=str, default='error',
                        help='Type of experiment: error, mnist, cifar10, cifar100, svhn, imagenet')
    parser.add_argument('--n_samples', type=int, default=16, help='Number of samples for MNIST dataset')
    parser.add_argument('--m', type=int, default=128, help='Number of rows in the matrix V')
    parser.add_argument('--n', type=int, default=128, help='Number of columns in the matrix V')
    parser.add_argument('--r', type=int, default=64, help='Rank for NMF factorization')
    parser.add_argument('--num_iters', type=int, default=2000, help='Number of iterations for training')
    parser.add_argument('--eps', type=float, default=1e-3, help='Stability parameter for multiplicative updates of Lee & Seung')
    parser.add_argument('--optim_name', type=str, default='LeeSeungNMF', help='Name of the optimizer to use: Lee&Seung,ANGD,ZOAHessB')
    parser.add_argument('--log_path', type=str, default='./logs', help='Path to save logs')
    parser.add_argument('--eta', type=float, default=0.1, help='Learning rate for ANGD')
    parser.add_argument('--a', type=float, default=3e-5, help='Auxiliary update step a for ANGD')
    parser.add_argument('--beta_type', type=str, default='nesterov',
                        choices=['nesterov', 'constant'], help='Beta schedule for ANGD')
    parser.add_argument('--beta_const', type=float, default=0.9,
                        help='If beta_type==constant, use this beta value')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='Optional gradient clipping threshold for ANGD (e.g. 1.0)')
    parser.add_argument('--a_zoa', type=float, default=0.8, help='Auxiliary update step a for ZOAHessB') 
    parser.add_argument('--h_init', type=float, default=1e-3, help='Initial smoothing parameter h0 for ZOAHessB')
    parser.add_argument('--rho', type=float, default=0.9, help='Decay factor for h: h <- rho*h')
    parser.add_argument('--alpha_init', type=float, default=0.1, help='Initial/main step size alpha for ZOAHessB')
    parser.add_argument('--armijo', action='store_true', help='Whether to use Armijo line search in ZOAHessB')
    parser.add_argument('--armijo_beta', type=float, default=0.5, help='Armijo backtracking factor')
    parser.add_argument('--armijo_c', type=float, default=1e-3, help='Armijo condition constant c')
    parser.add_argument('--armijo_max_iters', type=int, default=5, help='Armijo max line-search iters')
    parser.add_argument('--finite_difference', type=bool, default=True,
                        help='Use coordinate-wise finite difference for ZOAHessB (default: False)')
    parser.add_argument('--img_size', type=int, default=224,      # 仅 ImageNet 用
                        help='Resize side length for ImageNet images (default=64)')
    args = parser.parse_args()

    # Initialize V, W, H
    if args.experiment_type == 'error':
        V = synthetic_data(args.m, args.n).to(device)
        W = nn.Parameter(torch.rand(args.m, args.r, device=device)*0.1)
        H = nn.Parameter(torch.rand(args.r, args.n, device=device)*0.1)
    elif args.experiment_type == 'mnist':
        V, img_stack = load_mnist_matrix(n_samples=args.n_samples, device=device)
        args.m, args.n = V.shape
        W = nn.Parameter(torch.rand(args.m, args.r, device=device)*0.1)
        H = nn.Parameter(torch.rand(args.r, args.n, device=device)*0.1)
    elif args.experiment_type == 'fashionmnist':
        V, img_stack = load_fashionmnist_matrix(n_samples=args.n_samples, device=device)
        args.m, args.n = V.shape
        W = nn.Parameter(torch.rand(args.m, args.r, device=device)*0.1)
        H = nn.Parameter(torch.rand(args.r, args.n, device=device)*0.1)
    elif args.experiment_type == 'cifar10':
        V, img_stack = load_cifar10_matrix(n_samples=args.n_samples, device=device)
        args.m, args.n = V.shape                      # m = 3072
        W = nn.Parameter(torch.rand(args.m, args.r, device=device)*0.1)
        H = nn.Parameter(torch.rand(args.r, args.n, device=device)*0.1)
    elif args.experiment_type == 'cifar100':
        V, img_stack = load_cifar100_matrix(n_samples=args.n_samples, device=device)
        args.m, args.n = V.shape
        W = nn.Parameter(torch.rand(args.m, args.r, device=device)*0.1)
        H = nn.Parameter(torch.rand(args.r, args.n, device=device)*0.1)
    elif args.experiment_type == 'svhn':
        V, img_stack = load_svhn_matrix(n_samples=args.n_samples, device=device)
        args.m, args.n = V.shape                   # m = 3×32×32 = 3072
        W = nn.Parameter(torch.rand(args.m, args.r, device=device)*0.1)
        H = nn.Parameter(torch.rand(args.r, args.n, device=device)*0.1)
    elif args.experiment_type == 'imagenet':
        V, img_stack = load_imagenet_matrix(
            n_samples=args.n_samples, device=device,
            root='E:\\datasets\\imagenet\\train',
            img_size=args.img_size)
        args.m, args.n = V.shape                   # m = 3×S×S
        W = nn.Parameter(torch.rand(args.m, args.r, device=device)*0.1)
        H = nn.Parameter(torch.rand(args.r, args.n, device=device)*0.1)
    else:
        raise ValueError(f"Unsupported experiment type: {args.experiment_type}")

    # Set up optimizer
    optim_name = args.optim_name.lower()
    optimizer_label = {'lee&seung':"Lee&Seung",
                        'angd':"ANGD",
                        'zoahessb':"ZOAHessB"}
    optimizer_classes = {
        'lee&seung': lambda: LeeSeung([W, H], V, eps=args.eps),
        'angd': lambda: ANGD([W, H], V, eta=args.eta, a_const=args.a, 
                                beta_type=args.beta_type, beta_const=args.beta_const,
                                grad_clip=args.grad_clip),
        'zoahessb': lambda: ZOAHessB([W, H], V,
                             a_const=args.a_zoa, beta_type=args.beta_type, beta_const=args.beta_const,
                             h_init=args.h_init, rho=args.rho,
                             alpha_init=args.alpha_init,
                             armijo=args.armijo, armijo_beta=args.armijo_beta,
                             armijo_c=args.armijo_c, armijo_max_iters=args.armijo_max_iters,
                             device=device),
        'ngd': lambda: ANGD([W, H], V, eta=0.2, a_const=3e-2,
                          beta_type='constant', beta_const=0.0,
                            grad_clip=args.grad_clip, eps=args.eps)
    }
    if optim_name not in optimizer_classes:
        raise ValueError(f"Unknown optimizer name: {args.optim_name}")

    optimizer = optimizer_classes[optim_name]()
    num_iters = args.num_iters

    # Training with tqdm progress bar
    pbar = tqdm(range(num_iters), total=num_iters, desc=f'Training with {args.optim_name}', ncols=100, position=0, leave=True)

    args.log_path = args.log_path+'/'+args.experiment_type
    # csv_,csv_writer = prepare_csv(args.log_path,args.optim_name.lower(),args)
    # write_csv(csv_,csv_writer,head=True)
    for iters in pbar:
            V_approx = W @ H
            loss = torch.norm(V - V_approx, p='fro')**2 / V.shape[0]
            optimizer.step()
            # write_csv(csv_, csv_writer, loss=loss.item(), iter=iters+1)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iter': iters+1,
                'optim': args.optim_name
            })
    if args.experiment_type not in ['error']:
        print("Saving Recovered Matrix of "+args.experiment_type)
        plot_comparison(img_stack, (W @ H).detach(), title=f'{args.optim_name} Recovered Matrix',label=args.optim_name, exp_type=args.experiment_type)
        plot_spectrum_comparision(img_stack, (W @ H).detach(), label=args.optim_name, exp_type=args.experiment_type, iter_num=args.num_iters)
def main():
    train()
if __name__ == '__main__':
    main()

