import torch
import torchvision
from torchvision import transforms
import os, random, numpy as np, torch

def set_seed(seed: int = 42, deterministic: bool = True):
    """Fix every relevant RNG to make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)         
    torch.cuda.manual_seed(seed)    
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ.setdefault(
        "CUBLAS_WORKSPACE_CONFIG", ":4096:8") 

def _imgs_to_matrix(imgs):
    """imgs : Tensor [n, C, H, W]  →  V : Tensor [C·H·W, n]"""
    return imgs.view(imgs.shape[0], -1).T.contiguous()

def load_svhn_matrix(n_samples=16, device='cuda', root='E:\\datasets'):
    from torchvision.datasets import SVHN
    transform = transforms.ToTensor()          # 保持 0-1 浮点
    ds = SVHN(root=root, split='train', download=True, transform=transform)
    imgs = torch.stack([ds[i][0] for i in range(n_samples)])   # (n, 3, 32, 32)
    V = _imgs_to_matrix(imgs).to(device)                      # (3072, n)
    return V, imgs                                            # imgs 仍在 CPU, 供显示

def load_imagenet_matrix(n_samples=16, device='cuda',
                         root='E:\\datasets\\Imagenet\\train', img_size=224):
    from torchvision.datasets import ImageFolder
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()              # [0-1]
    ])
    ds = ImageFolder(root, transform=transform)
    imgs = torch.stack([ds[i][0] for i in range(n_samples)])   # (n, 3, S, S)
    V = _imgs_to_matrix(imgs).to(device)                      # (3·S·S, n)
    return V, imgs

# Generate a random non-negative matrix V
def synthetic_data(m=1024, n=1024):
    """
    Generate a synthetic non-negative matrix V of shape (m, n).
    """
    return torch.rand(m, n) 

def load_mnist_matrix(n_samples=16, device='cuda'):
    transform = transforms.ToTensor()
    mnist = torchvision.datasets.MNIST(root='E:\datasets', train=True,
                                       download=True, transform=transform)
    imgs = torch.stack([mnist[i][0].squeeze() for i in range(n_samples)])
    V = imgs.view(n_samples, -1).T.contiguous()
    return V.to(device), imgs

def load_fashionmnist_matrix(n_samples=15, device='cuda'):
    transform = transforms.ToTensor()
    fashionmnist = torchvision.datasets.FashionMNIST(root='E:\datasets', train=True,
                                       download=True, transform=transform)
    imgs = torch.stack([fashionmnist[i][0].squeeze() for i in range(n_samples)])  # (n, 28, 28)
    V = imgs.view(n_samples, -1).T.contiguous()
    return V.to(device), imgs

def _load_cifar_matrix(dataset: str = 'cifar10',
                       n_samples: int = 16,
                       device: str | torch.device = 'cuda'):
    """
    Return
    -------
    V      : [3072, n_samples]  flattened RGB column-matrix (32×32×3)
    imgs   : [n_samples, 3, 32, 32]  tensor for later visualisation
    """
    transform = transforms.ToTensor()          # 0-1, C×H×W
    if dataset.lower() == 'cifar10':
        ds = torchvision.datasets.CIFAR10(root='E:\\datasets',
                                          train=True, download=True,
                                          transform=transform)
    elif dataset.lower() == 'cifar100':
        ds = torchvision.datasets.CIFAR100(root='E:\\datasets',
                                           train=True, download=True,
                                           transform=transform)
    else:
        raise ValueError("dataset must be 'cifar10' or 'cifar100'")
    imgs = torch.stack([ds[i][0] for i in range(n_samples)])   # [n, 3, 32, 32]
    V = imgs.view(n_samples, -1).T.contiguous()                # [3072, n]
    return V.to(device), imgs

def load_cifar10_matrix(n_samples=16, device='cuda'):
    return _load_cifar_matrix('cifar10', n_samples, device)

def load_cifar100_matrix(n_samples=16, device='cuda'):
    return _load_cifar_matrix('cifar100', n_samples, device)