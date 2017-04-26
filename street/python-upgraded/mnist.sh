#export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
export CUDA_VISIBLE_DEVICES=0


train_dir=/home/dengdan/temp_nfs/fsns-origin
python vgsl_eval.py --num_steps=1000 \
  --eval_data=../data/validation* \
  --decoder=../testdata/charset_size=134.txt \
  --eval_interval_secs=300 --train_dir=$train_dir --eval_dir=$train_dir/eval 
exit
python vgsl_train.py --max_steps=100000000 --train_data=../data/train* \
  --train_dir=$train_dir 
exit

#tensorboard --logdir=$train_dir


