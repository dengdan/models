export PYTHONPATH=slim:$PYTHONPATH
python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=object_detection/ssd_text/ssd_text.config \
    --train_dir=/root/temp/no-use/object-detection
