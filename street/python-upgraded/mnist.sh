export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
export CUDA_VISIBLE_DEVICES=0

cd python
train_dir=/tmp/fsns-debug
pdb vgsl_train.py --max_steps=100000000 --train_data=../data/train* \
  --train_dir=$train_dir 
exit
tensorboard --logdir=$train_dir


