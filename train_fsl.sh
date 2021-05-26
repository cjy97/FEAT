python train_fsl.py  \
--max_epoch 200 \
--model_class FEAT \
--backbone_class Res12 \
--dataset MiniImageNet \
--way 5 \
--eval_way 5 \
--shot 1 \
--eval_shot 1 \
--query 15 \
--eval_query 15 \
--balance 0.01 \
--temperature 64 \
--temperature2 64 \
--lr 0.0002 \
--lr_mul 10 \
--lr_scheduler step \
--step_size 40 --gamma 0.5 \
--gpu 2 \
--init_weights ./saves/initialization/miniimagenet/Res12-pre.pth \
--eval_interval 1 \
--use_euclidean

