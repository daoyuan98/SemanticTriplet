python train_weighter3hc.py \
        -net "resnet18" \
        -beta 1.0   \
        -weighter_gamma1 0.5 \
        -weighter_gamma2 0.5 \
        -weighter_gamma3 0.5 \
        -similarity_source "glove" \
        -exp_name "r18_glove_cifar100"   \
        -dataset "cifar100" \
        -seed 0



