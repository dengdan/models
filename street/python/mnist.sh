export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
export CUDA_VISIBLE_DEVICES=0

train_dir=/home/dengdan/temp_nfs/fsns-debug
rm -rf $train_dir
python vgsl_train.py --max_steps=100000000 --train_data=../data/train* \
  --train_dir=$train_dir 


