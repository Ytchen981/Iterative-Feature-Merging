for max_ratio in 1.0
do
  for threshold in 0.01 0.03 0.05 0.07 0.1 0.12 0.14 0.15 0.18 0.2
  do
    for resnet_name in 18 34
    do
      CUDA_VISIBLE_DEVICES=1 python resnet${resnet_name}_imagenet_self_weight_matching.py --model_param /home/chenyiting/LMC_complexity/Model_Complexity/output/2023-09-05/resnet${resnet_name}_imagenet_torch_v1/resnet${resnet_name}.pth --max_ratio $max_ratio --threshold $threshold
    done
  done
done