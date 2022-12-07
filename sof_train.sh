# Warmup
python main_base.py actnet RGB --arch resnet50 --num_segments 16 --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 40 80 --epochs 15 --repeat_batch 1 --batch-size 24 --dropout 0.5 --consensus_type=avg --eval-freq=1 --npb --gpus 0 1 2 3 --exp_header sofnet_res50_t16_epo15_192_lr.001 --rescale_to 192 -j 36 --data_dir ../datasets/activity-net-v1.3 --log_dir ../logs_tsm 

# Train all
python main_base.py actnet RGB --arch resnet50 --num_segments 16 --lr 0.001 --epochs 60  --batch-size 24 --repeat_batch 1 --eval-freq=1 --lr_steps 30 60 -j 32 --npb --gpus 0 1 2 3 --exp_header sofnet_neurips_test_16f_192 --stop_or_forward --use_conf_btw_blocks --rescale_to 192 --block_rnn_list base conv_2 conv_3 conv_4 conv_5 --accuracy_weight 0.95 --efficency_weight 0.05 --exp_decay --init_tau 5.0 --use_gflops_loss --save_meta --random_seed 1007 --data_dir ../datasets/activity-net-v1.3 --log_dir ../logs_tsm --model_paths ../logs_tsm/sofnet_res50_t16_epo15_192_lr.001/models/ckpt.best.pth.tar





