#!/bin/bash

echo "Training original undefended model..."
python cifar10-train-org.py 

# echo "Training regularized model..."
# python cifar10-train-reg.py --wd 0.0005

# echo "Training regularized model..."
# python cifar10-train-reg.py --wd 0.001

# echo "Training regularized model..."
# python cifar10-train-reg.py --wd 0.005

# echo "Training regularized model with dropout..."
# python cifar10-train-regdrop.py --wd 0.0005 --drp 0.25

# echo "Training regularized model with dropout..."
# python cifar10-train-regdrop.py --wd 0.0005 --drp 0.5

# echo "Training regularized model with label smoothing..."
# python cifar10-train-ls.py --epsilon 0.01

# echo "Training DPSGD model"
# python cifar10-train-dpsgd.py  --dp_batchsize 64 --lr 0.1 --dp_norm_clip 1.0 --dp_noise_multiplier 0.5

