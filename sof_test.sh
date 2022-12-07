if [ "$#" -eq 2 ]
then
    DATA_DIR=$1
    MODEL_DIR=$2
else
    DATA_DIR=../datasets/activity-net-v1.3
    MODEL_DIR=../../logs_tsm/test/models
fi

echo "Using data path: ${DATA_DIR} and model path: ${MODEL_DIR}"


python -u main_base.py actnet RGB --arch resnet50 --num_segments 16 --npb --exp_header X --stop_or_forward --use_conf_btw_blocks --block_rnn_list base conv_2 conv_3 conv_4 conv_5 --rescale_to 192 --accuracy_weight 0.95 --efficency_weight 0.05 --exp_decay --init_tau 0.000001 --use_gflops_loss --batch-size 24 -j 32 --gpus 0 1 2 3 --test_from ${MODEL_DIR}/ckpt.best.pth.tar --save_meta --cnt_log ../arnet_cnt_log --data_dir ${DATA_DIR}
