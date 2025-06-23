#!/bin/bash

# ANGD experiments
# m=n>r
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 8 --n 8 --r 4 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 16 --n 16 --r 8  --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 32 --n 32 --r 16 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 64 --n 64 --r 32 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 128 --n 128 --r 64 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 256 --n 256 --r 128 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 512 --n 512 --r 256 --eta 6e-2 --a 1e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 1024 --n 1024 --r 512 --eta 8e-3 --a 1e-5
# m=n=r
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 8 --n 8 --r 8 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 16 --n 16 --r 16 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 32 --n 32 --r 32 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 64 --n 64 --r 64 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 128 --n 128 --r 128 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 256 --n 256 --r 256 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 512 --n 512 --r 512 --eta 6e-2 --a 1e-5
python ./main.py  --log_path ./logs --optim_name ANGD --num_iters 10000 --m 1024 --n 1024 --r 1024 --eta 6e-3 --a 1e-5

# NGD experiments
# m=n>r
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 8 --n 8 --r 4 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 16 --n 16 --r 8  --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 32 --n 32 --r 16 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 64 --n 64 --r 32 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 128 --n 128 --r 64 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 256 --n 256 --r 128 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 512 --n 512 --r 256 --eta 6e-2 --a 1e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 1024 --n 1024 --r 512 --eta 8e-3 --a 1e-5
# m=n=r
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 8 --n 8 --r 8 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 16 --n 16 --r 16 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 32 --n 32 --r 32 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 64 --n 64 --r 64 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 128 --n 128 --r 128 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 256 --n 256 --r 256 --eta 0.1 --a 3e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 512 --n 512 --r 512 --eta 6e-2 --a 1e-5
python ./main.py  --log_path ./logs --optim_name NGD --num_iters 10000 --m 1024 --n 1024 --r 1024 --eta 6e-3 --a 1e-5

# ZOAHessB experiments
# m=n>r
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 8 --n 8 --r 4 --a_zoa 0.8 --h_init 1e-10 --alpha_init 0.5 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 16 --n 16 --r 8  --a_zoa 0.8 --h_init 1e-10 --alpha_init 0.5 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 32 --n 32 --r 16 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.2 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 64 --n 64 --r 32 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.1 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 128 --n 128 --r 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.1 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 256 --n 256 --r 128 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.1 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 512 --n 512 --r 256 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.1 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 1024 --n 1024 --r 512 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.1 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
# m=n=r
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 8 --n 8 --r 8 --a_zoa 0.8 --h_init 1e-10 --alpha_init 0.5 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 16 --n 16 --r 16 --a_zoa 0.8 --h_init 1e-10 --alpha_init 0.5 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 32 --n 32 --r 32 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.2 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 64 --n 64 --r 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.1 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 128 --n 128 --r 128 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.1 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 256 --n 256 --r 256 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 512 --n 512 --r 512 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.1 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name ZOAHessB --num_iters 10000 --m 1024 --n 1024 --r 1024 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.1 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5

# Lee&Seung experiments
# m=n>r
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 8 --n 8 --r 4
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 16 --n 16 --r 8
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 32 --n 32 --r 16
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 64 --n 64 --r 32
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 128 --n 128 --r 64
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 256 --n 256 --r 128
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 512 --n 512 --r 256
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 1024 --n 1024 --r 512
# m=n=r
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 8 --n 8 --r 8
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 16 --n 16 --r 16
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 32 --n 32 --r 32
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 64 --n 64 --r 64
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 128 --n 128 --r 128
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 256 --n 256 --r 256
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 512 --n 512 --r 512
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 10000 --m 1024 --n 1024 --r 1024


# recover
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30000 --experiment_type mnist --r 64 --n_samples 64 --eta 6e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30000 --experiment_type fashionmnist --r 64 --n_samples 64 --eta 6e-3 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30000 --experiment_type cifar10 --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30000 --experiment_type cifar100 --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30000 --experiment_type svhn --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30000 --experiment_type mnist --r 48 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30000 --experiment_type fashionmnist --r 48 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30000 --experiment_type cifar10 --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30000 --experiment_type cifar100 --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30000 --experiment_type svhn --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30000 --experiment_type mnist --r 48 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30000 --experiment_type fashionmnist --r 48 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30000 --experiment_type cifar10 --r 128 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30000 --experiment_type cifar100 --r 128 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30000 --experiment_type svhn --r 128 --n_samples 64 --eps 1e-1

# recover spectrum
# mnist
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30 --experiment_type mnist --r 64 --n_samples 64 --eta 6e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 300 --experiment_type mnist --r 64 --n_samples 64 --eta 6e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 3000 --experiment_type mnist --r 64 --n_samples 64 --eta 6e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30000 --experiment_type mnist --r 64 --n_samples 64 --eta 6e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30 --experiment_type mnist --r 48 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 300 --experiment_type mnist --r 48 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 3000 --experiment_type mnist --r 48 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30000 --experiment_type mnist --r 48 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30 --experiment_type mnist --r 48 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 300 --experiment_type mnist --r 48 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 3000 --experiment_type mnist --r 48 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30000 --experiment_type mnist --r 48 --n_samples 64 --eps 1e-1
# fashionmnist
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30 --experiment_type fashionmnist --r 64 --n_samples 64 --eta 6e-3 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 300 --experiment_type fashionmnist --r 64 --n_samples 64 --eta 6e-3 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 3000 --experiment_type fashionmnist --r 64 --n_samples 64 --eta 6e-3 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30000 --experiment_type fashionmnist --r 64 --n_samples 64 --eta 6e-3 --a 3e-5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30 --experiment_type fashionmnist --r 48 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 300 --experiment_type fashionmnist --r 48 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 3000 --experiment_type fashionmnist --r 48 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30000 --experiment_type fashionmnist --r 48 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30 --experiment_type fashionmnist --r 48 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 300 --experiment_type fashionmnist --r 48 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 3000 --experiment_type fashionmnist --r 48 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30000 --experiment_type fashionmnist --r 48 --n_samples 64 --eps 1e-1
# cifar10
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30 --experiment_type cifar10 --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 300 --experiment_type cifar10 --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 3000 --experiment_type cifar10 --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30000 --experiment_type cifar10 --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30 --experiment_type cifar10 --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 300 --experiment_type cifar10 --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 3000 --experiment_type cifar10 --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30000 --experiment_type cifar10 --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30 --experiment_type cifar10 --r 128 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 300 --experiment_type cifar10 --r 128 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 3000 --experiment_type cifar10 --r 128 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30000 --experiment_type cifar10 --r 128 --n_samples 64 --eps 1e-1
# cifar100
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30 --experiment_type cifar100 --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 300 --experiment_type cifar100 --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 3000 --experiment_type cifar100 --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30000 --experiment_type cifar100 --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30 --experiment_type cifar100 --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 300 --experiment_type cifar100 --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 3000 --experiment_type cifar100 --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30000 --experiment_type cifar100 --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30 --experiment_type cifar100 --r 128 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 300 --experiment_type cifar100 --r 128 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 3000 --experiment_type cifar100 --r 128 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30000 --experiment_type cifar100 --r 128 --n_samples 64 --eps 1e-1
# svhn
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30 --experiment_type svhn --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 300 --experiment_type svhn --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 3000 --experiment_type svhn --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ANGD --num_iters 30000 --experiment_type svhn --r 128 --n_samples 64 --eta 1e-2 --a 3e-5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30 --experiment_type svhn --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 300 --experiment_type svhn --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 3000 --experiment_type svhn --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py --log_path ./logs --optim_name ZOAHessB --num_iters 30000 --experiment_type svhn --r 128 --n_samples 64 --a_zoa 0.8 --h_init 1e-3 --alpha_init 0.01 --rho 0.9 --armijo_beta 0.5 --armijo_c 1e-3 --armijo_max_iters 5
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30 --experiment_type svhn --r 128 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 300 --experiment_type svhn --r 128 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 3000 --experiment_type svhn --r 128 --n_samples 64 --eps 1e-1
python ./main.py  --log_path ./logs --optim_name "Lee&Seung" --num_iters 30000 --experiment_type svhn --r 128 --n_samples 64 --eps 1e-1


