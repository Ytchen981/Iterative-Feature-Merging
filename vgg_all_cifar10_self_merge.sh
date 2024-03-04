for max_ratio in 1.0
do
  for threshold in 0.001 0.01 0.03 0.05 0.07 0.1 0.12 0.14 0.15 0.18 0.2 0.22 0.24 0.25 0.26 0.28 0.3
  do
    for vgg_name in 19
    do
      CUDA_VISIBLE_DEVICES=6 python vgg_cifar10_self_weight_matching_v2.py --model_param /home/chenyiting/LMC_complexity/Model_Complexity/output/2023-09-21/model_b_vgg${vgg_name}_training.yml/params/model_epoch149.pt --max_ratio $max_ratio --threshold $threshold --vgg_name VGG${vgg_name}
    done
  done
done