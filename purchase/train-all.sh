#!/bin/bash

# echo "Training original undefended model..."
# python purchase-train-org.py 

# echo "Training regularized model..."
# python purchase-train-reg.py --wd 0.0005

# echo "Training regularized model..."
# python purchase-train-reg.py --wd 0.001

echo "Training regularized model..."
python purchase-train-reg.py --wd 0.005

# echo "Training regularized model with dropout..."
# python purchase-train-regdrop.py --wd 0.0005 --drp 0.25

# echo "Training regularized model with dropout..."
# python purchase-train-regdrop.py --wd 0.0005 --drp 0.5

# echo "Training regularized model with label smoothing..."
# python purchase-train-ls.py --epsilon 0.03

# echo "Training DPSGD model"
# python purchase-train-dpsgd.py  --dp_batchsize 128 --lr 0.0005 --dp_norm_clip 1.0 --dp_noise_multiplier 1.0  --epochs 200


