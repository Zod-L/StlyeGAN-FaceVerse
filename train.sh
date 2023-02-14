#CUDA_VISIBLE_DEVICES=4,5,6,7 \
#python train.py --outdir=./tex_out/grad/weight0 --gpus=8 --data=/home/liyi/data/faceverse_data/train/uv/fine/tex --cfg=stylegan2 --batch=128 --aug=noaug \
# --joint_train=False --uv_folder=/home/liyi/data/faceverse_data/train/uv/coarse/tex --use_noise=True --lap_weight=0 --smooth_weight=0 --mirror=True\


CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 \
python train.py --outdir=./deform_out/256-1024/grad_smooth_reall2_1_0_0 --gpus=6 \
--data=/home/liyi/data/faceverse_data/train/uv/fine/loc --cfg=stylegan2 --batch=48 --aug=noaug \
--joint_train=False --uv_folder=/home/liyi/data/faceverse_data/train/uv/coarse/loc --use_noise=False \
--lap_weight=1 --smooth_weight=0 --l2_weight=0 \
