#!/bin/bash

# 3.3.4(1) 位姿表示方法及损失函数实验结果
# 对比Quat的损失函数
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot Quat -olt Cos --ALPHA 5 1 --BETA 1 1
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot Quat -olt CosDistance --ALPHA 5 1 --BETA 1 1
# 对比Cart的损失函数
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L2 -ot Quat -olt Cos --ALPHA 5 1 --BETA 1 1
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt SmoothL1 -ot Quat -olt Cos --ALPHA 5 1 --BETA 1 1
# 对比Euler的损失函数
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot Euler -olt L1 --ALPHA 5 1 --BETA 1 1
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot Euler -olt L2 --ALPHA 5 1 --BETA 1 1
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot Euler -olt SmoothL1 --ALPHA 5 1 --BETA 1 1
# 对比Discrete
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 1 --alpha 0.0 --neighbor 0 -olt CE --ALPHA 5 1 --BETA 1 1 0 0

# beta系数 暂时弃用
# python3 train.py --exp_type beta-ratio --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 1 --alpha 0.0 --neighbor 0 -olt CE --ALPHA 5 1 --BETA 1 3 0 0
# python3 train.py --exp_type beta-ratio --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 1 --alpha 0.0 --neighbor 0 -olt CE --ALPHA 5 1 --BETA 1 5 0 0
# python3 train.py --exp_type beta-ratio --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 1 --alpha 0.0 --neighbor 0 -olt CE --ALPHA 5 1 --BETA 1 7 0 0


# 3.3.4(2) 概率分布平滑实验结果
# alpha=0.1 n=2 4 6
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 1 --alpha 0.1 --neighbor 2 -olt CE --ALPHA 5 1 --BETA 1 1 0 0
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 1 --alpha 0.1 --neighbor 4 -olt CE --ALPHA 5 1 --BETA 1 1 0 0
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 1 --alpha 0.1 --neighbor 6 -olt CE --ALPHA 5 1 --BETA 1 1 0 0
# alpha=0.01 0.2 n=2
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 1 --alpha 0.01 --neighbor 2 -olt CE --ALPHA 5 1 --BETA 1 1 0 0
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 1 --alpha 0.2 --neighbor 2 -olt CE --ALPHA 5 1 --BETA 1 1 0 0
# python3 train.py --exp_type pose-loss --cache --compile --img_size 400 640 --epochs 200 --batch_size 40 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 1 --alpha 0.3 --neighbor 2 -olt CE --ALPHA 5 1 --BETA 1 1 0 0




# test
# python3 train.py --exp_type test --epochs 300 --batch_size 50 --num_workers 15 --backbone mobilenetv4_conv_small --neck TailNeck --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type Cos --ALPHA 0 1 --BETA 0 1
# python3 train.py --exp_type best_test --epochs 300 --batch_size 50 --num_workers 15 --backbone mobilenetv4_conv_medium --neck DensAttFPN --att_type SSIA --pos_type DiscreteSpher --pos_loss_type CE --ori_type DiscreteEuler --ori_loss_type CE --avg_size 4 2 1 --ALPHA 1 5 --BETA 1 5
# python3 train.py --cache --exp_type test --epochs 300 --batch_size 60 --num_workers 15 --backbone mobilenetv4_conv_small --neck TailNeck --avg_size 2 --pos_type DiscreteSpher --pos_loss_type KL --r_stride 1 --angle_stride 2 --ALPHA 1 0 --BETA 1 0

# cart+quat
# python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type Cos --ALPHA 1 1 --BETA 1 1
# python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type CosDistance --ALPHA 1 1 --BETA 1 1
# python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --pos_type Cart --pos_loss_type L1 --ori_type Quat --ori_loss_type ExpCos --ALPHA 1 1 --BETA 1 1

# python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --pos_type Cart --pos_loss_type L2 --ori_type Quat --ori_loss_type Cos --ALPHA 1 1 --BETA 1 1
# python3 train.py --cache --exp_type CartAndQuat --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --pos_type Cart --pos_loss_type SmoothL1 --ori_type Quat --ori_loss_type Cos --ALPHA 1 1 --BETA 1 1

# spher+Euler
# python3 train.py --cache --exp_type SpherAndEuler --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --pos_type Spher --pos_loss_type L1 --ori_type Euler --ori_loss_type L1 --ALPHA 1 1 --BETA 1 1
# python3 train.py --cache --exp_type SpherAndEuler --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --pos_type Spher --pos_loss_type L2 --ori_type Euler --ori_loss_type L2 --ALPHA 1 1 --BETA 1 1
# python3 train.py --cache --exp_type SpherAndEuler --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --pos_type Spher --pos_loss_type SmoothL1 --ori_type Euler --ori_loss_type SmoothL1 --ALPHA 1 1 --BETA 1 1

# Discrete
# python3 train.py --cache --exp_type Discrete --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --pos_type DiscreteSpher --pos_loss_type CE --ori_type DiscreteEuler --ori_loss_type CE --ALPHA 1 1 --BETA 1 1 1 1
# python3 train.py --cache --exp_type Discrete --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --pos_type DiscreteSpher --pos_loss_type KL --ori_type DiscreteEuler --ori_loss_type KL --ALPHA 1 1 --BETA 1 1 1 1

# BETA
# python3 train.py --exp_type BETA --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 1 --ori_loss_type CE --ALPHA 1 1 --BETA 1 1 1 1
# python3 train.py --exp_type BETA --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 1 --ori_loss_type CE --ALPHA 1 1 --BETA 1 3 1 3
# python3 train.py --exp_type BETA --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 1 --ori_loss_type CE --ALPHA 1 1 --BETA 1 5 1 5
# python3 train.py --exp_type BETA --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 1 --ori_loss_type CE --ALPHA 1 1 --BETA 1 7 1 7

# BETA_COS
# python3 train.py --exp_type BETA --cache --compile --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 1 --ori_loss_type CE --ALPHA 1 1 --BETA 1 5 1 5


# DiscreteHyper
## euler stride
# python3 train.py --exp_type EulerStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 2 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type EulerStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type EulerStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 10 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5

## spher angle stride
# python3 train.py --exp_type AngleStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 0.5 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type AngleStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 2 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type AngleStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 5 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type AngleStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 10 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 5 1 --BETA 1 5 1 5

## r stride
# python3 train.py --exp_type RStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 0.5 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type RStride --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 2 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5

# Perspective Augmentation
# python3 train.py --exp_type Perspective --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5


# neck
# python3 train.py --exp_type Neck --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck PAFPN --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type Neck --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck BiFPN --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type Neck --cache --compile --beta_cos --img_size 400 640 --backbone mobilenetv3_large_100 --neck DensAttFPN --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type Neck --cache --compile --beta_cos --lr0 0.0005 --lr_min 0.000005 --img_size 400 640 --backbone mobilenetv3_large_100 --neck DensAttFPN --att_type SE --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5
# python3 train.py --exp_type Neck --cache --compile --beta_cos --lr0 0.0005 --lr_min 0.000005 --img_size 400 640 --backbone mobilenetv3_large_100 --neck DensAttFPN --att_type CBAM --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 100 1 --BETA 1 5 1 5
# python3 train.py --exp_type Neck --cache --compile --beta_cos --lr0 0.0005 --lr_min 0.000005 --img_size 400 640 --backbone mobilenetv3_large_100 --neck DensAttFPN --att_type SAM --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 100 1 --BETA 1 5 1 5
# python3 train.py --exp_type Neck --cache --compile --beta_cos --lr0 0.001 --lr_min 0.000001 --img_size 400 640 --backbone mobilenetv3_large_100 --neck DensAttFPN --att_type SSIA --head AvgPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 100 1 --BETA 1 5 1 5

# head
# python3 train.py --exp_type Head --cache --compile --beta_cos --lr0 0.001 --lr_min 0.000001 --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head MixPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 100 1 --BETA 1 5 1 5
# python3 train.py --exp_type Head --cache --compile --beta_cos --lr0 0.001 --lr_min 0.000001 --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head MixPoolHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 100 1 --BETA 1 5 1 5
# python3 train.py --exp_type Head --cache --compile --beta_cos --lr0 0.001 --lr_min 0.000001 --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head MixPoolHead --pool_size 1 --weighted_learnable --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 100 1 --BETA 1 5 1 5
# python3 train.py --exp_type Head --cache --compile --beta_cos --lr0 0.001 --lr_min 0.000001 --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head SPPHead --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 100 1 --BETA 1 5 1 5
# python3 train.py --exp_type Head --cache --compile --beta_cos --lr0 0.001 --lr_min 0.000001 --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head MHAHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 100 1 --BETA 1 5 1 5
# python3 train.py --exp_type Head --cache --compile --beta_cos --lr0 0.001 --lr_min 0.000001 --img_size 400 640 --backbone mobilenetv3_large_100 --neck TailNeck --head TokenHead --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 10 1 --BETA 1 5 1 5

# 1.0几度了
# python3 train.py --exp_type Best --cache --compile --beta_cos --img_size 600 960 --epochs 400 --batch_size 30 --beta_epochs 400 --backbone mobilenetv3_large_100 --neck TailNeck --head TokenHead --num_heads 4 --num_layers 8 --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 5 1 --BETA 1 5 1 5
# 1.2几度
# python3 train.py --exp_type Best --cache --compile --beta_cos --img_size 500 800 --epochs 400 --batch_size 30 --beta_epochs 400 --backbone mobilenetv3_large_100 --neck TailNeck --head TokenHead --num_heads 4 --num_layers 8 --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Best --cache --compile --beta_cos --img_size 780 1248 --epochs 400 --batch_size 30 --beta_epochs 400 --backbone mobilenetv3_large_100 --neck TailNeck --head TokenHead --num_heads 4 --num_layers 8 --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Best --cache --compile --beta_cos --img_size 900 1440 --epochs 400 --batch_size 30 --beta_epochs 400 --backbone mobilenetv3_large_100 --neck TailNeck --head TokenHead --num_heads 4 --num_layers 8 --pool_size 1 --pos_type DiscreteSpher --r_stride 1 --angle_stride 1 --pos_loss_type CE --ori_type DiscreteEuler --stride 5 --ori_loss_type CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Best --cache --compile --beta_cos --img_size 480 768 --epochs 5 --batch_size 30 --beta_epochs 5 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 --pool_size 1 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 5 -olt CE --ALPHA 5 1 --BETA 1 5 1 5 --zr_p 0.8



# python3 train.py --exp_type Attention --cache --compile --beta_cos --img_size 480 768 --epochs 400 --batch_size 30 --beta_epochs 400 -b mobilenetv3_large_100 --WMSA -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5 --zr_p 0.8
# python3 train.py --exp_type DiscreteStride --cache --compile --beta_cos --img_size 480 768 --epochs 400 --batch_size 30 --beta_epochs 400 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 2 -olt CE --ALPHA 5 1 --BETA 1 5 1 5 --zr_p 0.8
# python3 train.py --exp_type DiscreteStride --cache --compile --beta_cos --img_size 480 768 --epochs 400 --batch_size 30 --beta_epochs 400 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 5 -olt CE --ALPHA 5 1 --BETA 1 5 1 5 --zr_p 0.8
# python3 train.py --exp_type DiscreteStride --cache --compile --beta_cos --img_size 480 768 --epochs 400 --batch_size 30 --beta_epochs 400 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 2 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5 --zr_p 0.8
# python3 train.py --exp_type DiscreteStride --cache --compile --beta_cos --img_size 480 768 --epochs 400 --batch_size 30 --beta_epochs 400 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 5 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5 --zr_p 0.8
# python3 train.py --exp_type DiscreteStride --cache --compile --beta_cos --img_size 480 768 --epochs 400 --batch_size 30 --beta_epochs 400 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5 --zr_p 0.8
# python3 train.py --exp_type DiscreteStride --cache --compile --beta_cos --img_size 480 768 --epochs 400 --batch_size 30 --beta_epochs 400 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 2 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5 --zr_p 0.8


# python3 train.py --exp_type Head --cache --compile --beta_cos --img_size 480 768 --epochs 400 --batch_size 30 --beta_epochs 400 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead --pool_size 1 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5 --zr_p 0.8
# python3 train.py --exp_type CosBeta --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 --beta_epochs 400 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 --pool_size 1 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5 --zr_p 0.8



# head 消融实验
# python3 train.py --exp_type Ab_head --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_head --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 1 --num_layers 1 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_head --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 1 --num_layers 4 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_head --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 1 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_head --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 1 --num_layers 12 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_head --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 8 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_head --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 0 0
# python3 train.py --exp_type Ab_head --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head MaxPoolHead -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 0 0
# python3 train.py --exp_type Ab_head --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n IdentityNeck --head AvgPoolHead --pool_size 1 1 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 0 0
# python3 train.py --exp_type Ab_head --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 0 0

# cos beta消融实验
# 仅仅使用概率分布进行优化
# python3 train.py --exp_type Ab_cosbeta --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 0 0
# 同时使用概率和求和，但是没有cos decay
# python3 train.py --exp_type Ab_cosbeta --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# 有额外约束项
# python3 train.py --exp_type Ab_cosbeta --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# cos
# python3 train.py --exp_type Ab_cosbeta --seed 0 --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 3 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_cosbeta --seed 42 --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 3 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_cosbeta --seed 3407 --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 3 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_cosbeta --seed 1111 --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 3 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_cosbeta --seed 1233 --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 3 1 --BETA 1 5 1 5
# 开始直接放飞
# python3 train.py --exp_type Ab_cosbeta --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 0 0 1 5

# imgsize
# python3 train.py --exp_type Ab_imgsz --cache --compile --img_size 600 960 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 8 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_imgsz --cache --compile --img_size 800 1280 --gradient_clip_val 4 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 10 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 8 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_imgsz --cache --compile --img_size 400 640 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_imgsz --cache --compile --img_size 300 480 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5

# pos 和 ori
# python3 train.py --exp_type Ab_pose --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot Quat -olt Cos --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_pose --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L2 -ot Quat -olt Cos --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_pose --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot Quat -olt CosDistance --ALPHA 5 1 --BETA 1 5 1 5
# python3 train.py --exp_type Ab_pose --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L2 -ot Quat -olt CosDistance --ALPHA 5 1 --BETA 1 5 1 5

# backbone
# ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
python3 train.py --exp_type Ab_backbone --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 8 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
# python3 train.py --exp_type Ab_backbone --cache --compile --img_size 600 960 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_075 -n TailNeck --head TokenHead --num_heads 8 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
# python3 train.py --exp_type Ab_backbone --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_150d -n TailNeck --head TokenHead --num_heads 8 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5
# ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9