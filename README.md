# SEAT (Self-Ensemble Adversarial Training)

This is the official code for ICLR'22 paper "Self-Ensemble Adversarial Training"



## Prerequisites

- Python (3.7)
- Pytorch (1.5)
- Torchvision
- CUDA
- Numpy
- [AutoAttack](https://github.com/fra31/auto-attack)



## Training and Testing

- Train ResNet-18 on CIFAR10:

```
  $ CUDA_VISIBLE_DEVICES={your GPU number} python3 seat.py 
```

- Train WRN-32-10 on CIFAR10

```
  $ CUDA_VISIBLE_DEVICES={your GPU number} python3 seat.py --arch 'WRN'
```

Then, it will automatically run all the robustness evaluation in our paper, including NAT, PGD20/100, MIM, CW, $APGD_{ce}$, $APGD_{dlr}$, $APGD_{t}$​, $FAB_{t}$, $Square$ and AutoAttack.



## Citation

If you use this code or idea in your work, please consider citing our paper:

```

```
