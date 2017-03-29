export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
export CUDA_VISIBLE_DEVICES=1

dump_dir=/home/dengdan/temp_nfs/
name=fsns-fcn12s-soft
train_dir=$dump_dir/$name

python vgsl_train.py --max_steps=100000000 --train_data=../data/train* \
  --train_dir=$train_dir --proc_name=$name --gm=0.2

exit

python vgsl_eval.py --num_steps=1000 \
  --eval_data=/home/dengdan/github/models/street/data/validation* \
  --decoder=../testdata/charset_size=134.txt \
  --eval_interval_secs=300 --train_dir=$train_dir --eval_dir=$train_dir/eval \
  --proc_name=$name
tensorboard --logdir=$train_dir --port=9000

