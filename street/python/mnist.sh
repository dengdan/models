export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
export CUDA_VISIBLE_DEVICES=0

cd python
train_dir=/tmp/fsns-debug
rm -rf $train_dir
pdb vgsl_train.py --max_steps=100000000 --train_data=../data/train* \
  --train_dir=$train_dir 
exit
tensorboard --logdir=$train_dir

train_dir=/tmp/fsns
rm -rf $train_dir


python vgsl_train.py --max_steps=10000 --num_preprocess_threads=1 \
  --train_data=../data/fsns-00000-of-00001 \
  --initial_learning_rate=0.0001 --final_learning_rate=0.0001 \
  --train_dir=$train_dir 

python vgsl_eval.py --num_steps=256 --num_preprocess_threads=1 \
   --eval_data=../data/fsns-00000-of-00001 \
   --decoder=../testdata/charset_size=134.txt \
   --eval_interval_secs=300 --train_dir=/tmp/fsns --eval_dir=/tmp/fsns/eval
tensorboard --logdir=$train_dir


train_dir=/tmp/ctc
rm -rf $train_dir
python vgsl_train.py --model_str='1,32,0,1[S1(1x32)1,3 Lbx100]O1c105' \
  --max_steps=4096 --train_data=../data/arial-32-00000-of-00001 \
  --initial_learning_rate=0.001 --final_learning_rate=0.001 \
  --num_preprocess_threads=1 --train_dir=$train_dir &
python vgsl_eval.py --model_str='1,32,0,1[S1(1x32)1,3 Lbx100]O1c105' \
  --num_steps=256 --eval_data=../data/arial-32-00000-of-00001 \
  --num_preprocess_threads=1 --decoder=../testdata/arial.charset_size=105.txt \
  --eval_interval_secs=15 --train_dir=$train_dir --eval_dir=$train_dir/eval &
tensorboard --logdir=/tmp/ctc$train_dir

exit  
train_dir=/tmp/fixed
rm -rf $train_dir
python vgsl_train.py --model_str='8,16,0,1[S1(1x16)1,3 Lfx32 Lrx32 Lfx32]O1s12' \
  --max_steps=3072 --train_data=../data/numbers-16-00000-of-00001 \
  --initial_learning_rate=0.001 --final_learning_rate=0.001 \
  --num_preprocess_threads=1 --train_dir=$train_dir
python vgsl_eval.py --model_str='8,16,0,1[S1(1x16)1,3 Lfx32 Lrx32 Lfx32]O1s12' \
  --num_steps=256 --eval_data=../data/numbers-16-00000-of-00001 \
  --num_preprocess_threads=1 --decoder=../testdata/numbers.charset_size=12.txt \
  --eval_interval_secs=0 --train_dir=$train_dir --eval_dir=$train_dir/eval

exit
train_dir=/tmp/mnist
rm -rf $train_dir
python vgsl_train.py --model_str='16,0,0,1[Ct5,5,16 Mp3,3 Lfys32 Lfxs64]O0s12' \
  --max_steps=4024 --train_data=../data/mnist-sample-00000-of-00001 \
  --initial_learning_rate=0.001 --final_learning_rate=0.001 \
  --num_preprocess_threads=1 --train_dir=$train_dir
python vgsl_eval.py --model_str='16,0,0,1[Ct5,5,16 Mp3,3 Lfys32 Lfxs64]O0s12' \
  --num_steps=256 --eval_data=../data/mnist-sample-00000-of-00001 \
  --num_preprocess_threads=1 --decoder=../testdata/numbers.charset_size=12.txt \
  --eval_interval_secs=0 --train_dir=$train_dir --eval_dir=$train_dir/eval
  

