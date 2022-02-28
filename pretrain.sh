python3.7 pretrain.py \
--lr 0.001 \
--batch_size 256 \
--max_epoch 500 \
--backbone_class Swin \
--schedule 350 400 440 460 480 \
--ngpu 3 \
--gamma 0.1
