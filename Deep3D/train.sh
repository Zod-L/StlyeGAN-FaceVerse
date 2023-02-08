# python  train.py --name=pretrain_toonme_head --flist=./datalist/toonme_caric/train/masks.txt \
# --flist_val=./datalist/toonme_caric/val/masks.txt --batch_size=256 --gpu_ids=0,1,2,3 \
# --n_epochs=5000 --save_epoch_freq=1000 --pretrained_name=test_pretrain \
# --data_root=../data/deep3d_data --cartoon_weight_shape=0.5  --cartoon_weight_tex=0.5 \
# --head_only=True --finetune=True --bfm_model=fuse_model_front_improve.mat --tex_from_im=False --epoch_count=0
 
python  train.py --name=pretrain_toonme_whole --flist=./datalist/toonme_caric/train/masks.txt \
--flist_val=./datalist/toonme_caric/val/masks.txt --batch_size=128 --gpu_ids=0,1 \
--n_epochs=5000 --save_epoch_freq=1000 --pretrained_name=pretrain_toonme_head \
--data_root=../data/deep3d_data --cartoon_weight_shape=0.5  --cartoon_weight_tex=0.5 \
--head_only=False --finetune=True --bfm_model=fuse_model_front_improve.mat --tex_from_im=False --epoch_count=0


