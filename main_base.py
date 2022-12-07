import warnings

warnings.filterwarnings("ignore")
import os
import sys
import time
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import math
import shutil

from ops.dataset import TSNDataSet
from ops.models_sof import TSN_Sof
from ops.transforms import *
from sof_opts import parser
from ops import dataset_config
from ops.utils import AverageMeter, accuracy, cal_map, Recorder

from tensorboardX import SummaryWriter
from ops.my_logger import Logger
from ops.flops_table import get_gflops_params
from ops.utils import get_mobv2_new_sd

from os.path import join as ospj


def load_to_sd(model_dict, model_path, module_name, fc_name, resolution, apple_to_apple=False):
    if ".pth" in model_path:
        print("done loading\t%s\t(res:%3d) from\t%s" % ("%-25s" % module_name, resolution, model_path))
        sd = torch.load(model_path)['state_dict']
        if "module.block_cnn_dict.base.1.bias" in sd:
            print("Directly upload")
            return sd
        
        if apple_to_apple:
            del_keys = []
            if args.remove_all_base_0:
                for key in sd:
                    if "module.base_model_list.0" in key or "new_fc_list.0" in key or "linear." in key:
                        del_keys.append(key)

            if args.no_weights_from_linear:
                for key in sd:
                    if "linear." in key:
                        del_keys.append(key)

            for key in list(set(del_keys)):
                del sd[key]

            return sd

        replace_dict = []
        nowhere_ks = []
        notfind_ks = []

        for k, v in sd.items():  
            new_k = k.replace("base_model", module_name)
            new_k = new_k.replace("new_fc", fc_name)
            if new_k in model_dict:
                replace_dict.append((k, new_k))
            else:
                nowhere_ks.append(k)
        for new_k, v in model_dict.items():
            if module_name in new_k:
                k = new_k.replace(module_name, "base_model")
                if k not in sd:
                    notfind_ks.append(k)
            if fc_name in new_k:
                k = new_k.replace(fc_name, "new_fc")
                if k not in sd:
                    notfind_ks.append(k)
        if len(nowhere_ks) != 0:
            print("Vars not in ada network, but are in pretrained weights\n" + ("\n%s NEW  " % module_name).join(
                nowhere_ks))
        if len(notfind_ks) != 0:
            print("Vars not in pretrained weights, but are needed in ada network\n" + ("\n%s LACK " % module_name).join(
                notfind_ks))
        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)

        if "lite_backbone" in module_name:
            # TODO not loading new_fc in this case, because we are using hidden_dim
            if args.frame_independent == False:
                del sd["module.lite_fc.weight"]
                del sd["module.lite_fc.bias"]
        return {k: v for k, v in sd.items() if k in model_dict}
    else:
        print("skip loading\t%s\t(res:%3d) from\t%s" % ("%-25s" % module_name, resolution, model_path))
        return {}
    


def main():
    t_start = time.time()
    global args, best_prec1, num_class, use_ada_framework  # , model
    set_random_seed(args.random_seed)
    use_ada_framework = args.stop_or_forward

    if args.ablation:
        logger = None
    else:
        if not test_mode:
            logger = Logger()
            sys.stdout = logger
        else:
            logger = None

    num_class, args.train_list, args.val_list, args.root_path, prefix = dataset_config.return_dataset(args.dataset,
                                                                                                      args.data_dir)

    init_gflops_table()
    model = TSN_Sof(num_class, args.num_segments,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                fc_lr5=not (args.tune_from and args.dataset in args.tune_from),
                args=args)
    
    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.tune_from:
        print(("=> f/ine-tuning from '{}'".format(args.tune_from)))
        sd = torch.load(args.tune_from)
        sd = sd['state_dict']
        model_dict = model.state_dict()
        replace_dict = []
        for k, v in sd.items():
            if k not in model_dict and k.replace('.net', '') in model_dict:
                print('=> Load after remove .net: ', k)
                replace_dict.append((k, k.replace('.net', '')))
        for k, v in model_dict.items():
            if k not in sd and k.replace('.net', '') in sd:
                print('=> Load after adding .net: ', k)
                replace_dict.append((k.replace('.net', ''), k))

        for k, k_new in replace_dict:
            sd[k_new] = sd.pop(k)
        keys1 = set(list(sd.keys()))
        keys2 = set(list(model_dict.keys()))
        set_diff = (keys1 - keys2) | (keys2 - keys1)
        print('#### Notice: keys that failed to load: {}'.format(set_diff))
        if args.dataset not in args.tune_from:  # new dataset
            print('=> New dataset, do not load fc weights')
            sd = {k: v for k, v in sd.items() if 'fc' not in k}
        model_dict.update(sd)
        model.load_state_dict(model_dict)

        
    if args.stop_or_forward:
        if test_mode:
            print("Test mode load from pretrained model SoF")
            the_model_path = args.test_from
            if ".pth.tar" not in the_model_path:
                the_model_path = ospj(the_model_path, "models", "ckpt.best.pth.tar")
            model_dict = model.state_dict()
            sd = load_to_sd(model_dict, the_model_path, "foo", "bar", -1, apple_to_apple=True)
            model_dict.update(sd)
            model.load_state_dict(model_dict)
        elif args.base_pretrained_from != "":
            print("Adaptively load from pretrained whole SoF")
            model_dict = model.state_dict()
            sd = load_to_sd(model_dict, args.base_pretrained_from, "foo", "bar", -1, apple_to_apple=True)

            model_dict.update(sd)
            model.load_state_dict(model_dict)

        elif len(args.model_paths) != 0:
            print("Adaptively load from model_path_list SoF")
            model_dict = model.state_dict()
            for i, tmp_path in enumerate(args.model_paths):
                base_model_index = i
                new_i = i
                
                sd = load_to_sd(model_dict, tmp_path, "base_model_list.%d" % base_model_index, "new_fc_list.%d" % new_i, 224)
                
                model_dict.update(sd)
            model.load_state_dict(model_dict)
    
    cudnn.benchmark = True
    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)
    data_length = 1
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.train_list, num_segments=args.num_segments,
                   image_tmpl=prefix,
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       normalize,
                   ]), dense_sample=args.dense_sample,
                   dataset=args.dataset,
                   partial_fcvid_eval=args.partial_fcvid_eval,
                   partial_ratio=args.partial_ratio,
                   random_crop=args.random_crop,
                   center_crop=args.center_crop,
                   ada_crop_list=args.ada_crop_list,
                   rescale_to=args.rescale_to,
                   save_meta=args.save_meta),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        drop_last=True)  # prevent something not % n_GPU

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.val_list, num_segments=args.num_segments,
                   image_tmpl=prefix,
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=False),
                       ToTorchFormatTensor(div=True),
                       normalize,
                   ]), dense_sample=args.dense_sample,
                   dataset=args.dataset,
                   partial_fcvid_eval=args.partial_fcvid_eval,
                   partial_ratio=args.partial_ratio,
                   random_crop=args.random_crop,
                   center_crop=args.center_crop,
                   ada_crop_list=args.ada_crop_list,
                   rescale_to=args.rescale_to,
                   save_meta=args.save_meta
                   ),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    if not test_mode:
        exp_full_path = setup_log_directory(logger, args.log_dir, args.exp_header)
    else:
        exp_full_path = None

    if not args.ablation:
        if not test_mode:
            with open(os.path.join(exp_full_path, 'args.txt'), 'w') as f:
                f.write(str(args))
            tf_writer = SummaryWriter(log_dir=exp_full_path)
        else:
            tf_writer = None
    else:
        tf_writer = None

    map_record = Recorder()
    mmap_record = Recorder()
    prec_record = Recorder()
    best_train_usage_str = None
    best_val_usage_str = None

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        if not args.skip_training:
            set_random_seed(args.random_seed + epoch)
            adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)
            train_usage_str = train(train_loader, model, criterion, optimizer, epoch, logger, exp_full_path, tf_writer)
        else:
            train_usage_str = "No training usage stats (Eval Mode)"

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            set_random_seed(args.random_seed)
            mAP, mmAP, prec1, val_usage_str, val_gflops = validate(val_loader, model, criterion, epoch, logger,
                                                                   exp_full_path, tf_writer)

            # remember best prec@1 and save checkpoint
            map_record.update(mAP)
            mmap_record.update(mmAP)
            prec_record.update(prec1)

            if mmap_record.is_current_best():
                best_train_usage_str = train_usage_str
                best_val_usage_str = val_usage_str

            print('Best mAP: %.3f (epoch=%d)\t\tBest mmAP: %.3f(epoch=%d)\t\tBest Prec@1: %.3f (epoch=%d)' % (
                map_record.best_val, map_record.best_at,
                mmap_record.best_val, mmap_record.best_at,
                prec_record.best_val, prec_record.best_at))

            if args.skip_training:
                break

            if (not args.ablation) and (not test_mode):
                tf_writer.add_scalar('acc/test_top1_best', prec_record.best_val, epoch)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': prec_record.best_val,
#                 }, mmap_record.is_current_best(), exp_full_path)
                }, mmap_record.is_current_best(), exp_full_path, epoch+1)
    if use_ada_framework and not test_mode:
        print("Best train usage:")
        print(best_train_usage_str)
        print()
        print("Best val usage:")
        print(best_val_usage_str)

    print("Finished in %.4f seconds\n" % (time.time() - t_start))


def set_random_seed(the_seed):
    if args.random_seed >= 0:
        np.random.seed(the_seed)
        torch.manual_seed(the_seed)
        torch.cuda.manual_seed(the_seed)
        torch.cuda.manual_seed_all(the_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(the_seed)


def init_gflops_table():
    global gflops_table
    gflops_table = {}
    default_gflops_table = {}
    seg_len = -1
    resolution = args.rescale_to
    
    """get gflops of block even it not using"""
    default_block_list = ["base", "conv_2", "conv_3", "conv_4", "conv_5"]
    default_case_list = ["cnn", "rnn"]
    resize = int(args.rescale_to)
   
    default_gflops_table[str(args.arch) + "base"] = \
                get_gflops_params(args.arch, "base", num_class, resolution=resize, case="cnn", seg_len=seg_len)[0]
    default_gflops_table[str(args.arch) + "base" + "fc"] = \
                get_gflops_params(args.arch, "base_fc", num_class, resolution=resize, case="cnn", seg_len=seg_len)[0]
    for _block in default_block_list:
        for _case in default_case_list:
            default_gflops_table[str(args.arch) + _block + _case] = \
                get_gflops_params(args.arch, _block, num_class, resolution=resize, case=_case, hidden_dim = args.hidden_dim if _case is "rnn" else None, seg_len=seg_len)[0]
    
    print(default_gflops_table)

    """add gflops of unusing block to using block"""
    start = 0
    for using_block in args.block_rnn_list :
        gflops_table[str(args.arch) + using_block + "rnn"] = default_gflops_table[str(args.arch) + using_block + "rnn"]
        gflops_table[str(args.arch) + using_block + "cnn"] = 0
        index = default_block_list.index(using_block)
        for j in range(start, index+1):
            if j is 0:
                gflops_table[str(args.arch) + using_block + "cnn"] = default_gflops_table[str(args.arch) + "base"]
            else:
                gflops_table[str(args.arch) + using_block + "cnn"] += default_gflops_table[str(args.arch) + default_block_list[j] + "cnn"]
        start = index+1
    
    """get gflops of all pass block"""
    gflops_table[str(args.arch) + "basefc"] = default_gflops_table[str(args.arch) + "basefc"] 
    for last_block in range(start, len(default_block_list)):
        name = default_block_list[last_block]
        if name is not "base":
            gflops_table[str(args.arch) + "basefc"] += default_gflops_table[str(args.arch) + name + "cnn"] 

        
    print("gflops_table: from base to ")
    for k in gflops_table:
        print("%-20s: %.4f GFLOPS" % (k, gflops_table[k]))


def get_gflops_t_tt_vector():
    gflops_vec = []
    t_vec = []
    tt_vec = []

    if all([arch_name not in args.arch for arch_name in ["resnet", "mobilenet", "efficientnet", "res3d", "csn"]]):
        exit("We can only handle resnet/mobilenet/efficientnet/res3d/csn as backbone, when computing FLOPS")

    for using_block in args.block_rnn_list:
        gflops_lstm = gflops_table[str(args.arch) + str(using_block) + "rnn"]
        the_flops = gflops_table[str(args.arch) + str(using_block) + "cnn"] + gflops_lstm
        gflops_vec.append(the_flops)
        t_vec.append(1.)
        tt_vec.append(1.)
    
    the_flops = gflops_table[str(args.arch) + "basefc"]
    gflops_vec.append(the_flops)
    t_vec.append(1.)
    tt_vec.append(1.)
  
    return gflops_vec, t_vec, tt_vec #ex : (conv_2 skip, conv_3 skip, conv_4 skip, conv_5 skip, all_pass)


def cal_eff(r, all_policy_r):
    each_losses = []
    # TODO r N * T * (#which block exit, conv2/ conv_3/ conv_4/ conv_5/all)
    # r_loss : pass conv_2/ conv_3/ conv_4/ conv_5/ all
    gflops_vec, t_vec, tt_vec = get_gflops_t_tt_vector()
    t_vec = torch.tensor(t_vec).cuda()
    
    for i  in range(1, len(gflops_vec)):
        gflops_vec[i] += gflops_vec[i-1]
    total_gflops = gflops_vec[-1]

    for i in range(len(gflops_vec)):
        gflops_vec[i] = total_gflops - gflops_vec[i]
    gflops_vec[-1] += 0.00001

    if args.use_gflops_loss:
        r_loss = torch.tensor(gflops_vec).cuda()
    else:
        r_loss = torch.tensor([4., 2., 1., 0.5, 0.25]).cuda()[:r.shape[2]]
    

    loss = torch.sum(torch.mean(r, dim=[0, 1]) * r_loss)
    each_losses.append(loss.detach().cpu().item())
    
    return loss, each_losses    


def reverse_onehot(a):
    try:
        if args.stop_or_forward:
            return np.array(a.sum(axis=1), np.int32)
        else:
            return np.array([np.where(r > 0.5)[0][0] for r in a])
    except Exception as e:
        print("error stack:", e)
        print(a)
        for i, r in enumerate(a):
            print(i, r)
        return None

def confidence_criterion_loss(criterion, all_policy_r, feat_outs, target):
    # all_policy_r B,T,K-1,A
    # feat_outs B,T,(K-1)+1,#class
    policy_gt_loss = 0
    inner_acc_loss = 0
    _feat_outs = F.softmax(feat_outs, dim=-1)
    _target = target[:,0]
    total_cnt = 0.0
    total_acc_cnt = 0.0
    
    batch_size  = feat_outs.shape[0]
    time_length = feat_outs.shape[1]
    layer_cnt   = feat_outs.shape[2]
    
    for b_i in range(feat_outs.shape[0]):
        conf_outs = _feat_outs[b_i,:,:,_target[b_i]]
        diff_conf_l = []
        for k_i in range(1, layer_cnt):
            diff_conf_l.append(conf_outs[:,k_i] - conf_outs[:,k_i-1])
        
        target_pass_bool = torch.stack(diff_conf_l, dim=1) > 0  #T,K-1
        target_policy = torch.tensor(target_pass_bool, dtype=torch.long).cuda()
        
        for k_i in range(layer_cnt-1):
            total_cnt+=1.0
            policy_gt_loss += criterion(all_policy_r[b_i,:,k_i,:], target_policy[:,k_i])
    
    for t_i in range(time_length):
        for k_i in range(layer_cnt-1):
            total_acc_cnt +=1.0
            inner_acc_loss += criterion(feat_outs[:,t_i,k_i,:], _target)

    
    return policy_gt_loss/total_cnt, inner_acc_loss/total_acc_cnt

def get_criterion_loss(criterion, output, target):
    return criterion(output, target[:, 0])

def kl_categorical(p_logit, q_logit):
    import torch.nn.functional as F
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                         - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)

def compute_acc_eff_loss_with_weights(acc_loss, eff_loss, each_losses, epoch):
    acc_weight = args.accuracy_weight
    eff_weight = args.efficency_weight

    return acc_loss * acc_weight, eff_loss * eff_weight, [x * eff_weight for x in each_losses]

def compute_every_losses(r, all_policy_r, acc_loss, epoch):
    eff_loss, each_losses = cal_eff(r, all_policy_r)
    acc_loss, eff_loss, each_losses = compute_acc_eff_loss_with_weights(acc_loss, eff_loss, each_losses, epoch)
    return acc_loss, eff_loss, each_losses


def elastic_list_print(l, limit=8):
    if isinstance(l, str):
        return l

    limit = min(limit, len(l))
    l_output = "[%s," % (",".join([str(x) for x in l[:limit // 2]]))
    if l.shape[0] > limit:
        l_output += "..."
    l_output += "%s]" % (",".join([str(x) for x in l[-limit // 2:]]))
    return l_output


def compute_exp_decay_tau(epoch):
    return args.init_tau * np.exp(args.exp_decay_factor * epoch)


def get_policy_usage_str(r_list, act_dim):
    gflops_vec, t_vec, tt_vec = get_gflops_t_tt_vector()
    printed_str = ""
    rs = np.concatenate(r_list, axis=0)

    tmp_cnt = [np.sum(rs[:, :, iii] == 1) for iii in range(rs.shape[2])] #[#all #conv_2 #conv_3 #conv_4 #conv_5]
    tmp_total_cnt = rs.shape[0] * rs.shape[1]

    gflops = 0
    avg_frame_ratio = 0
    avg_pred_ratio = 0

    used_model_list = []
    reso_list = []

    prev_pass_cnt = tmp_total_cnt
    for action_i in range(rs.shape[2]):
        if action_i is 0:
            action_str = "pass%d (base) " % (action_i)
        else:
            action_str = "pass%d (%s)" % (action_i, args.block_rnn_list[action_i-1])

        usage_ratio = tmp_cnt[action_i] / tmp_total_cnt
        printed_str += "%-22s: %6d (%.2f%%)" % (action_str, tmp_cnt[action_i], 100 * usage_ratio)
        printed_str += "\n"

        gflops += usage_ratio * gflops_vec[action_i]
    
    avg_frame_ratio = usage_ratio * t_vec[-1]

    num_clips = args.num_segments
    printed_str += "GFLOPS: %.6f  AVG_FRAMES: %.3f " % (gflops, avg_frame_ratio * num_clips)

    return printed_str, gflops


def extra_each_loss_str(each_terms):
    loss_str_list = ["gf"]
    s = ""
    for i in range(len(loss_str_list)):
        s += " %s:(%.4f)" % (loss_str_list[i], each_terms[i].avg)
    return s


def get_current_temperature(num_epoch):
    if args.exp_decay:
        tau = compute_exp_decay_tau(num_epoch)
    else:
        tau = args.init_tau
    return tau

def get_average_meters(number):
    return [AverageMeter() for _ in range(number)]

def update_weights(epoch, acc, eff):
    if args.use_weight_decay:
        exp_decay_factor = np.log(float(acc)/0.8)/float(args.epochs)
        acc = 0.8 * np.exp(exp_decay_factor * epoch)
        eff = 1 - acc
    return acc, eff
    

def train(train_loader, model, criterion, optimizer, epoch, logger, exp_full_path, tf_writer):
    batch_time, data_time, losses, top1, top5 = get_average_meters(5)
    tau = 0
    if use_ada_framework:
        tau = get_current_temperature(epoch)
        if args.use_conf_btw_blocks:
            alosses, elosses, inner_alosses, policy_gt_losses, early_exit_losses = get_average_meters(5)
        else: 
            alosses, elosses = get_average_meters(2)
                      
        each_terms = get_average_meters(NUM_LOSSES)
        r_list = []

    meta_offset = -2 if args.save_meta else 0

    model.module.partialBN(not args.no_partialbn)
    model.train()

    end = time.time()
    print("#%s# lr:%.4f\ttau:%.4f" % (
        args.exp_header, optimizer.param_groups[-1]['lr'] * 0.1, tau if use_ada_framework else 0))
    
    accuracy_weight, efficiency_weight = update_weights(epoch, args.accuracy_weight, args.efficency_weight)
    
    accumulation_steps = args.repeat_batch
    total_loss = 0
    for i, input_tuple in enumerate(train_loader):
        data_time.update(time.time() - end) 

        target = input_tuple[1].cuda()
        target_var = torch.autograd.Variable(target)

        input = input_tuple[0]
        if args.stop_or_forward:
            input_var_list = [torch.autograd.Variable(input_item) for input_item in input_tuple[:-1 + meta_offset]]
            if args.use_conf_btw_blocks: 
                output, r, all_policy_r, feat_outs, early_stop_r, exit_r_t = model(input=input_var_list, tau=tau)
            else:
                output, r, all_policy_r, feat_outs, base_outs, _ = model(input=input_var_list, tau=tau)

            acc_loss = get_criterion_loss(criterion, output, target_var)

            acc_loss, eff_loss, each_losses = compute_every_losses(r, all_policy_r, acc_loss, epoch)
            acc_loss = acc_loss/args.accuracy_weight * accuracy_weight
            eff_loss = eff_loss/args.efficency_weight* efficiency_weight
            
            alosses.update(acc_loss.item(), input.size(0))
            elosses.update(eff_loss.item(), input.size(0))
            for l_i, each_loss in enumerate(each_losses):
                each_terms[l_i].update(each_loss, input.size(0))
                
            loss = acc_loss + eff_loss

            if args.use_conf_btw_blocks:
                policy_gt_loss, inner_aloss= confidence_criterion_loss(criterion, all_policy_r, feat_outs, target_var)
                policy_gt_loss = efficiency_weight * policy_gt_loss
                inner_aloss = accuracy_weight * inner_aloss
                inner_alosses.update(inner_aloss.item(), input.size(0))
                policy_gt_losses.update(policy_gt_loss.item(), input.size(0))
                loss = loss + policy_gt_loss + inner_aloss 

        else:
            input_var = torch.autograd.Variable(input)
            output = model(input=[input_var])
            loss = get_criterion_loss(criterion, output, target_var)
        
         # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss = loss / accumulation_steps
        loss.backward()
        if (i+1) % accumulation_steps == 0:
            if args.clip_gradient is not None:
                clip_grad_norm_(model.parameters(), args.clip_gradient)
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if use_ada_framework:
            r_list.append(r.detach().cpu().numpy())
          
        if i % args.print_freq == 0:
            print_output = ('Epoch:[{0:02d}][{1:03d}/{2:03d}] '
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            '{data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f}) '
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))  # TODO

            if use_ada_framework:
                roh_r = reverse_onehot(r[-1, :, :].detach().cpu().numpy())
                if args.use_conf_btw_blocks:
                    print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  i_a_l {inner_aloss.val:.4f} ({inner_aloss.avg:.4f})\t  p_g_l {p_g_loss.val:.4f} ({p_g_loss.avg:.4f})\tr {r} pick {pick}'.format(
                        aloss=alosses, eloss=elosses, inner_aloss=inner_alosses, p_g_loss=policy_gt_losses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                    )
                else:
                    print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  r {r} pick {pick}'.format(
                        aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                    )
                print_output += extra_each_loss_str(each_terms)
                
            if args.show_pred:
                print_output += elastic_list_print(output[-1, :].detach().cpu().numpy())
            print(print_output)

    if use_ada_framework:
        usage_str, gflops = get_policy_usage_str(r_list, len(args.block_rnn_list)+1)
        print(usage_str)

    if tf_writer is not None:
        tf_writer.add_scalar('loss/train', losses.avg, epoch)
        tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
        tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

    return usage_str if use_ada_framework else None


def validate(val_loader, model, criterion, epoch, logger, exp_full_path, tf_writer=None):
    batch_time, losses, top1, top5 = get_average_meters(4)
    tau = 0
    all_results = []
    all_targets = []
    all_local = {"TN":0, "FN":0, "FP":0, "TP":0}
    all_all_preds = []

    i_dont_need_bb = True

    if args.visual_log != '':
        try:
            if not(os.path.isdir(args.visual_log)):
               os.makedirs(ospj(args.visual_log))

            visual_log_path = args.visual_log
            if args.stop_or_forward :
                visual_log_txt_path = ospj(visual_log_path, "sof_visual_log.txt")
            else:
                visual_log_txt_path = ospj(visual_log_path, "visual_log.txt")
            visual_log = open(visual_log_txt_path, "w")


        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!")
                raise
                
    if args.cnt_log != '':
        try:
            if not(os.path.isdir(args.cnt_log)):
               os.makedirs(ospj(args.cnt_log))

            cnt_log_path = args.cnt_log
            if args.stop_or_forward :
                cnt_log_txt_path = ospj(cnt_log_path, "sof_cnt_log.txt")
            else:
                cnt_log_txt_path = ospj(cnt_log_path, "cnt_log.txt")
            cnt_log = open(cnt_log_txt_path, "w")
            input_result_dict = {}
            total_cnt_dict = {}
            target_dict = {}


        except OSError as e:
            if e.errno != errno.EEXIST:
                print("Failed to create directory!!!")
                raise

    if use_ada_framework:
        tau = get_current_temperature(epoch)
        if args.use_conf_btw_blocks:
            alosses, elosses, inner_alosses, policy_gt_losses, early_exit_losses = get_average_meters(5)
        else: 
            alosses, elosses = get_average_meters(2)
                      
        each_terms = get_average_meters(NUM_LOSSES)
        
        r_list = []
        if args.save_meta:
            name_list = []
            indices_list = []

    meta_offset = -1 if args.save_meta else 0

    # switch to evaluate mode
    model.eval()

    end = time.time()
    accuracy_weight, efficiency_weight = update_weights(epoch, args.accuracy_weight, args.efficency_weight)
    accumulation_steps = args.repeat_batch
    total_loss = 0
    with torch.no_grad():
        for i, input_tuple in enumerate(val_loader):
            target = input_tuple[1].cuda()
            input = input_tuple[0]

            # compute output
            if args.stop_or_forward:
                local_target = input_tuple[2].cuda()
                if args.use_conf_btw_blocks :
                    output, r, all_policy_r, feat_outs, early_stop_r, exit_r_t = model(input=input_tuple[:-1 + meta_offset], tau=tau)
                else:
                    output, r, all_policy_r, feat_outs, base_outs, _ = model(input=input_tuple[:-1 + meta_offset], tau=tau)
               
                acc_loss = get_criterion_loss(criterion, output, target)
                acc_loss, eff_loss, each_losses = compute_every_losses(r, all_policy_r, acc_loss, epoch)
                acc_loss = acc_loss/args.accuracy_weight * accuracy_weight
                eff_loss = eff_loss/args.efficency_weight* efficiency_weight
                          
                alosses.update(acc_loss.item(), input.size(0))
                elosses.update(eff_loss.item(), input.size(0))
                for l_i, each_loss in enumerate(each_losses):
                    each_terms[l_i].update(each_loss, input.size(0))

                loss = acc_loss + eff_loss
                if args.use_conf_btw_blocks:
                    policy_gt_loss, inner_aloss= confidence_criterion_loss(criterion, all_policy_r, feat_outs, target)
                    policy_gt_loss = efficiency_weight * policy_gt_loss
                    inner_aloss = accuracy_weight * inner_aloss
                    inner_alosses.update(inner_aloss.item(), input.size(0))
                    policy_gt_losses.update(policy_gt_loss.item(), input.size(0))
                    loss = loss + policy_gt_loss + inner_aloss
                        

            else:
                output = model(input=[input])
                loss = get_criterion_loss(criterion, output, target)
                
             # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
              
            if args.cnt_log != '':
                target_vals = target.cpu().numpy()
                output_vals = output.max(dim=1)[1].cpu().numpy()
                
                for i in range(len(target_vals)):
                    target_val = target_vals[i][0]
                    output_val = output_vals[i]
                    input_path = os.path.join(args.root_path, input_tuple[meta_offset-1][i])
                    
                    if input_path in input_result_dict:
                        if target_val == output_val:
                            input_result_dict[input_path] +=1
                        total_cnt_dict[input_path] +=1
                    else:
                        input_result_dict[input_path] = 1 if target_val == output_val else 0
                        total_cnt_dict[input_path] = 1
                        target_dict[input_path] = output_val
                   
                
            if args.visual_log != '':
                target_val = target.cpu().numpy()[0][0]
                output_val = output.max(dim=1)[1].cpu().numpy()[0]
                loc_target_val = local_target.cpu().numpy()[0]
                loc_output_val = r[:,:,-1].cpu().numpy()[0]
                
                input_path_list = list()
                image_tmpl='image_{:05d}.jpg'
                for seg_ind in input_tuple[meta_offset][0]:
                    input_path_list.append(os.path.join(args.root_path, input_tuple[meta_offset-1][0], image_tmpl.format(int(seg_ind))))
              
                if target_val == output_val :
                    print("True")
                    visual_log.write("\nTrue")
                else :
                    print("False")
                    visual_log.write("\nFalse")

                print('input path list')
                print(input_path_list[0])

                print('target')
                print(target_val)
                print('output')
                print(output_val)
                print('r')
                print('loc_target')
                print(loc_target_val)
                print('loc_output')
                print(loc_output_val)
                
                for i in range(1):
                    print(reverse_onehot(r[i, :, :].cpu().numpy()))

                #visual_log.write('\ninput path list: ')
                for i in range(len(input_path_list)):
                    visual_log.write('\n')
                    visual_log.write(input_path_list[i])

                visual_log.write('\n')
                visual_log.write(str(target_val))
                visual_log.write('\n')
                visual_log.write(str(output_val))
                visual_log.write('\n')
                visual_log.write(str(loc_target_val))
                visual_log.write('\n')
                visual_log.write(str(loc_output_val))
                visual_log.write('\n')
                
                for i in range(1):
                    visual_log.writelines(str(reverse_onehot(r[i, :, :].cpu().numpy())))
                visual_log.write('\n')

            all_results.append(output)
            all_targets.append(target)
            if args.stop_or_forward:
                total_loc = (local_target+2*r[:,:,-1]).cpu().numpy()# (0,1) + 2*(0,1) =? TN:0 FN:1 FP:2 TP:3
                all_local['TN'] += np.count_nonzero(total_loc == 0)
                all_local['FN'] += np.count_nonzero(total_loc == 1)
                all_local['FP'] += np.count_nonzero(total_loc == 2)
                all_local['TP'] += np.count_nonzero(total_loc == 3)

            if not i_dont_need_bb:
                for bb_i in range(len(all_bb_results)):
                    all_bb_results[bb_i].append(base_outs[:, bb_i])

            if args.save_meta and args.save_all_preds:
                all_all_preds.append(all_preds)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target[:, 0], topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
              
            if use_ada_framework:
                r_list.append(r.cpu().numpy())

            if i % args.print_freq == 0:
                print_output = ('Test: [{0:03d}/{1:03d}] '
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
                                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

                if use_ada_framework:
                    roh_r = reverse_onehot(r[-1, :, :].detach().cpu().numpy())
                    if args.use_conf_btw_blocks:
                        print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  i_a_l {inner_aloss.val:.4f} ({inner_aloss.avg:.4f})\t  p_g_l {p_g_loss.val:.4f} ({p_g_loss.avg:.4f})\tr {r} pick {pick}'.format(aloss=alosses, eloss=elosses, inner_aloss=inner_alosses, p_g_loss=policy_gt_losses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                        )
                    else:
                        print_output += '\n a_l {aloss.val:.4f} ({aloss.avg:.4f})\t e_l {eloss.val:.4f} ({eloss.avg:.4f})\t  r {r} pick {pick}'.format(
                            aloss=alosses, eloss=elosses, r=elastic_list_print(roh_r), pick = np.count_nonzero(roh_r == len(args.block_rnn_list)+1)
                        )
 
                    #TN:0 FN:1 FP:2 TP:3
                    print_output += extra_each_loss_str(each_terms)
                    print_output += '\n location TP:{}, FP:{}, FN:{} ,TN: {} \t'.format(
                        all_local['TP'], all_local['FP'], all_local['FN'], all_local['TN']
                    )
                print(print_output)

    mAP, _ = cal_map(torch.cat(all_results, 0).cpu(),
                     torch.cat(all_targets, 0)[:, 0:1].cpu())  
    mmAP, _ = cal_map(torch.cat(all_results, 0).cpu(), torch.cat(all_targets, 0).cpu()) 
    print('Testing: mAP {mAP:.3f} mmAP {mmAP:.3f} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(mAP=mAP, mmAP=mmAP, top1=top1, top5=top5, loss=losses))
    
    if not i_dont_need_bb:
        bbmmaps = []
        bbprec1s = []
        all_targets_cpu = torch.cat(all_targets, 0).cpu()
        for bb_i in range(len(all_bb_results)):
            bb_results_cpu = torch.mean(torch.cat(all_bb_results[bb_i], 0), dim=1).cpu()
            bb_i_mmAP, _ = cal_map(bb_results_cpu, all_targets_cpu)  
            bbmmaps.append(bb_i_mmAP)

            bbprec1, = accuracy(bb_results_cpu, all_targets_cpu[:, 0], topk=(1,))
            bbprec1s.append(bbprec1)

        print("bbmmAP: " + " ".join(["{0:.3f}".format(bb_i_mmAP) for bb_i_mmAP in bbmmaps]))
        print("bb_Acc: " + " ".join(["{0:.3f}".format(bbprec1) for bbprec1 in bbprec1s]))
    
    gflops = 0
    if use_ada_framework:
        usage_str, gflops = get_policy_usage_str(r_list, len(args.block_rnn_list)+1)
        print(usage_str)

    if tf_writer is not None:
        tf_writer.add_scalar('loss/test', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)

    if args.cnt_log != '':
        for k,v in input_result_dict.items():
            cnt_log.write(str(k))
            cnt_log.write(',')
            cnt_log.write(str(target_dict[k]))
            cnt_log.write(',')
            cnt_log.write(str(v))
            cnt_log.write(',')
            cnt_log.write(str(total_cnt_dict[k]))
            cnt_log.write('\n')
        cnt_log.close()
    
    if args.visual_log != '':
        visual_log.close()

    return mAP, mmAP, top1.avg, usage_str if use_ada_framework else None, gflops

def save_checkpoint(state, is_best, exp_full_path, epoch):
    torch.save(state, '{}/models/ckpt{:03}.pth.tar'.format(exp_full_path, epoch))
    if is_best:
        torch.save(state, '%s/models/ckpt.best.pth.tar' % (exp_full_path))

def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def setup_log_directory(logger, log_dir, exp_header):
    if args.ablation:
        return None

#     exp_full_name = "g%s_%s" % (logger._timestr, exp_header)
    exp_full_name = exp_header
    exp_full_path = ospj(log_dir, exp_full_name)
    if os.path.exists(exp_full_path):
        shutil.rmtree(exp_full_path)
    os.makedirs(exp_full_path)
    os.makedirs(ospj(exp_full_path, "models"))
    logger.create_log(exp_full_path, test_mode, args.num_segments, args.batch_size, args.top_k)
    return exp_full_path

if __name__ == '__main__':
    best_prec1 = 0
    num_class = -1
    use_ada_framework = False
    NUM_LOSSES = 10
    gflops_table = {}
    args = parser.parse_args()
    test_mode = (args.test_from != "")

    if test_mode:  # TODO test mode:
        print("======== TEST MODE ========")
        args.skip_training = True
    main()

