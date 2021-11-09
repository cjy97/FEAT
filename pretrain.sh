python3.7 pretrain.py \
--lr 0.1 \
--batch_size 128 \
--max_epoch 500 \
--backbone_class Res12 \
--schedule 350 400 440 460 480 \
--ngpu 1 \
--gamma 0.1 \
--encoder_path ./saves/initialization/miniimagenet/Res12-pre.pth