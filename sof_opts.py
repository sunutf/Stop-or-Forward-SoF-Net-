import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str)
parser.add_argument('modality', type=str, choices=['RGB', 'Flow'])
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--store_name', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="BNInception")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
# parser.add_argument('--loss_type', type=str, default="nll",
#                    choices=['nll'])
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--tune_from', type=str, default=None, help='fine-tune from checkpoint')
parser.add_argument('--use_transformer', default=False, action='store_true', help='using transformer module at the end of prediction')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')  # TODO(changed from 120 to 50)
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs="+",  # TODO(changed from [50,100] to [20,40])
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')  # TODO(changed from 5e-4 to 1e-4)
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')  # TODO(changed from None to 20)
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')  # TODO(changed from 5 to 1)

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--root_log', type=str, default='logs')
parser.add_argument('--root_model', type=str, default='checkpoint')

parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')

# ADAPTIVE RESEARCH HYPER-PARAMETERS
parser.add_argument('--exp_header', default="default", type=str, help='experiment header')
parser.add_argument('--rescale_to', default=192, type=int)

#  adaptive resolution and skipping (hardcoded version)
parser.add_argument('--shared_backbone', action='store_true', help="share same backbone weight")
parser.add_argument('--accuracy_weight', default=1., type=float)
parser.add_argument('--efficency_weight', default=0., type=float)
parser.add_argument('--show_pred', action='store_true')

# adaptive depth skipping(jhseon)
parser.add_argument('--hidden_dim', default=512, type=int, help="dimension for hidden state and cell state")
parser.add_argument('--routing_weight', default=0., type=float)
parser.add_argument('--stop_or_forward', action='store_true', help='Stop of Forward model')
parser.add_argument('--block_rnn_list', default=['conv_2'], type=str, nargs='+', help='adaptively select depth of model')
parser.add_argument('--use_conf_btw_blocks', action='store_true', help='compare conf btw block ')
parser.add_argument('--conf_weight', default=0., type=float)
parser.add_argument('--repeat_batch', default=1, type=int)
parser.add_argument('--random_ratio', default=50, type=int)
parser.add_argument('--use_weight_decay', action='store_true', help='weight decay')
parser.add_argument('--use_early_stop_inf', action='store_true', help='early_stop_inf')
parser.add_argument('--consensus_type', type=str, default='avg') 
parser.add_argument('--visual_log', type=str, default="")
parser.add_argument('--cnt_log', type=str, default="")  


# TODO(yue) multi-label cases (for activity-net-v1.3)
# Always provides (single + multi) mAPs. Difference is only in training
parser.add_argument('--loss_type', type=str, default="nll", choices=['nll', 'bce'])

parser.add_argument('--save_freq', default=10, type=int, help="freq to save network model weight")  # TODO(yue)
parser.add_argument('--model_paths', default=[], type=str, nargs="+", help='path to load models for backbones')
parser.add_argument('--policy_path', default="", type=str, help="path of the policy network")

# annealing
parser.add_argument('--exp_decay', action='store_true', help="type of annealing")
parser.add_argument('--init_tau', default=5.0, type=float, help="annealing init temperature")
parser.add_argument('--exp_decay_factor', default=-0.045, type=float, help="exp decay factor per epoch")


# try different losses for efficiency terms
parser.add_argument('--use_gflops_loss', action='store_true')  # TODO(yue) use flops as loss assignment
parser.add_argument('--head_loss_weight', type=float, default=1e-6)  # TODO(yue) punish to the high resolution selection
parser.add_argument('--frames_loss_weight', type=float, default=1e-6)  # TODO(yue) use num_frames as a loss assignment

# finetuning and testing
parser.add_argument('--base_pretrained_from', type=str, default='', help='for base model pretrained path')
parser.add_argument('--skip_training', action='store_true')  # TODO(yue) just doing eval
parser.add_argument('--freeze_policy', action='store_true')  # TODO(yue) fix the policy

# reproducibility
parser.add_argument('--random_seed', type=int, default=1007)

# for FCVID or datasets where eval is too heavy
parser.add_argument('--partial_fcvid_eval', action='store_true')
parser.add_argument('--partial_ratio', type=float, default=0.2)

parser.add_argument('--center_crop', action='store_true')
parser.add_argument('--random_crop', action='store_true')

parser.add_argument('--top_k', type=int, default=10)  # TODO can also use scsampler!

parser.add_argument('--test_from', type=str, default="")

parser.add_argument('--ada_crop_list', default=[], type=int, nargs="+", help='num of anchor points per scaling')

parser.add_argument('--save_meta', action='store_true')
parser.add_argument('--ablation', action='store_true')
parser.add_argument('--remove_all_base_0', action='store_true')
parser.add_argument('--save_all_preds', action='store_true')

parser.add_argument('--data_dir', type=str, default="../../datasets/activity-net-v1.3")
parser.add_argument('--log_dir', type=str, default="../../logs_tsm")
