python train_fsl.py  \
--max_epoch 200 \
--episodes_per_epoch 2000 \
--model_class DN4 \
--backbone_class Res12 \
--dataset MiniImageNet \
--way 5 \
--eval_way 5 \
--shot 1 \
--eval_shot 1 \
--query 15 \
--eval_query 15 \
--temperature 1 \
--lr 0.001 \
--lr_scheduler step \
--step_size 20 --gamma 0.5 \
--augment \
--gpu 1 \
--eval_interval 1 \
# --is_distill  \
# --teacher_backbone_class Res12 \
# --teacher_init_weights ./saves/initialization/miniimagenet/Res12-pre.pth \
# --kd_loss KD \
# --kd_weight 0.1 \
# --is_prune \
# --remove_ratio 0.5 \