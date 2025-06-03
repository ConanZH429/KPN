#!/bin/bash

# 第三章概率分布平滑
# python3 train.py --exp_type soft-prob --cache --compile --img_size 480 768 --epochs 300 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 5 --alpha 0.0 --neighbor 0 -olt CE --ALPHA 1 1 --BETA 1 5 0 0
# python3 train.py --exp_type soft-prob --cache --compile --img_size 480 768 --epochs 300 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 5 --alpha 0.1 --neighbor 2 -olt CE --ALPHA 1 1 --BETA 1 5 0 0

# 第三章数据增强
# 数据增强实验结果
# python3 train.py --exp_type data-aug --cache --compile --img_size 480 768 --epochs 300 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 5 --alpha 0 --neighbor 0 -olt CE --ALPHA 1 1 --BETA 1 5 0 0 --zr_p 0 --crop_pad_p 0 --drop_block_p 0 --album_p 0.0 --zr_p 0.0
# python3 train.py --exp_type data-aug --cache --compile --img_size 480 768 --epochs 300 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 5 --alpha 0 --neighbor 0 -olt CE --ALPHA 1 1 --BETA 1 5 0 0 --zr_p 0 --crop_pad_p 0 --drop_block_p 0 --album_p 0.1 --zr_p 0.0
# python3 train.py --exp_type data-aug --cache --compile --img_size 480 768 --epochs 300 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 5 --alpha 0 --neighbor 0 -olt CE --ALPHA 1 1 --BETA 1 5 0 0 --zr_p 0 --crop_pad_p 0.5 --drop_block_p 0 --album_p 0.1 --zr_p 0.0
# python3 train.py --exp_type data-aug --cache --compile --img_size 480 768 --epochs 300 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 5 --alpha 0 --neighbor 0 -olt CE --ALPHA 1 1 --BETA 1 5 0 0 --zr_p 0 --crop_pad_p 0.5 --drop_block_p 0.5 --album_p 0.1 --zr_p 0.0
# python3 train.py --exp_type data-aug --cache --compile --img_size 480 768 --epochs 300 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 5 --alpha 0 --neighbor 0 -olt CE --ALPHA 1 1 --BETA 1 5 0 0 --zr_p 0 --crop_pad_p 0.5 --drop_block_p 0.5 --album_p 0.1 --zr_p 0.5

# 第四章
# (1) 消融实验
# python3 train.py --exp_type soft-prob --cache --compile --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n DensAttFPN --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 5 --alpha 0.1 --neighbor 2 -olt CE --ALPHA 1 1 --BETA 1 5 0 0
# python3 train.py --exp_type soft-prob --cache --compile --gradient_clip_val 2 --img_size 480 768 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n DensAttFPN --att_type SSIA --head AvgPoolHead -pt Cart -plt L1 -ot DiscreteEuler -es 5 --alpha 0.1 --neighbor 2 -olt CE --ALPHA 1 1 --BETA 1 5 0 0
python3 train.py --exp_type disc_pose --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 4 --num_layers 8 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 5 1 --BETA 1 5 1 5

# 第五章
# head实验
# python3 train.py --exp_type headToken11 --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 1 --num_layers 1 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 4 1 --BETA 1 5 1 5
# python3 train.py --exp_type headToken41 --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 1 --num_layers 4 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 4 1 --BETA 1 5 1 5
# python3 train.py --exp_type headToken121 --cache --compile --img_size 480 768 --beta_cos --beta_epochs 400 --epochs 400 --batch_size 30 -b mobilenetv3_large_100 -n TailNeck --head TokenHead --num_heads 1 --num_layers 12 -pt DiscreteSpher -rs 1 -as 1 -plt CE -ot DiscreteEuler -es 1 -olt CE --ALPHA 4 1 --BETA 1 5 1 5

