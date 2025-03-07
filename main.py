# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Adapted for AutoFocusFormer by Ziwen 2023

import os
import time
import argparse
import datetime
import numpy as np
import random
import copy

import torch
import torch.backends.cudnn as cudnn

from timm.loss import SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter, ModelEmaV2

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import cox_log_rank,load_checkpoint, save_checkpoint, auto_resume_helper, reduce_tensor, get_rank, init_distributed_mode, get_local_rank, get_world_size, NativeScalerWithGradNormCount

torch.backends.cuda.matmul.allow_tf32 = True
import pandas as pd
os.environ['TORCH_DISTRIBUTED_DEBUG'] = "INFO"
from sksurv.metrics import concordance_index_censored
def parse_option():
    parser = argparse.ArgumentParser('AutoFocusFormer training and evaluation script', add_help=True)
    parser.add_argument('--cfg', type=str, default="/workspace/aff_1layer/configs/aff_small.yaml",metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size per GPU")
    parser.add_argument('--epochs', type=int, help="number of epochs")
    parser.add_argument('--blr', type=float, help='base learning rate: absolute_lr = base_lr * total_batch_size / 512')
    parser.add_argument('--data-path', type=str,default="/workspace/TCGA_data/BLCA/patch_result_256/feature/h5_files",help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')
    parser.add_argument('--feature-dim', type=int,default=384, help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int,default=0,help='local rank for DistributedDataParallel')

    parser.add_argument("--nbhd_size", nargs='+', type=int,default=None,help='local rank for DistributedDataParallel')
    parser.add_argument("--cluster_size", nargs='+', type=int,default=None,help='local rank for DistributedDataParallel')
    parser.add_argument("--num_heads", nargs='+', type=int,default=None,help='local rank for DistributedDataParallel')
    parser.add_argument("--embed_dim", nargs='+', type=int,default=None,help='local rank for DistributedDataParallel')
    parser.add_argument("--drop_rate", type=float,default=None,help='local rank for DistributedDataParallel')
    parser.add_argument("--fold", type=int,default=None,help='local rank for DistributedDataParallel')
    parser.add_argument("--model_type", type=str,default=None,help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    return args

best_cin= 0
best_train_loss=0
best_train_acc=0

def main(config, logger):
    """
    Initializes all components needed for training, validates the resume checkpoint,
    and trains the model
    Args:
        config: CfgNode object, containing training and model configs
        logger: logger object for logging
    """

    # build dataloader
    data_loader_train, data_loader_val, mixup_fn = build_loader(config,config.DATA.DATA_PATH )

    # build model
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    print(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    # build loss scaler
    loss_scaler = NativeScalerWithGradNormCount(config)

    # build optimizer
    optimizer = build_optimizer(config, model)

    # build distributed model
    model_without_ddp = model
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)  # , find_unused_parameters=True)

    # print model param number and flop count
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    from ptflops import get_model_complexity_info
    # with torch.no_grad():
    #     macs, params = get_model_complexity_info(copy.deepcopy(model_without_ddp), (config.DATA.IN_CHANS, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), as_strings=True, print_per_layer_stat=False, verbose=True)
    # logger.info(f"macs {macs}, params {params}")

    # test model throughput
    # with torch.no_grad():
    #     throughput(config, data_loader_val, model, logger)
    torch.cuda.synchronize()
    if config.THROUGHPUT_MODE:
        return

    # build scheduler
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        # lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
        lr_scheduler = build_scheduler(config, optimizer, len(30) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # build criterion
    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    #     criterion = torch.nn.CrossEntropyLoss()
    #     # criterion=torch.nn.MSELoss()
    # else:
    #     # criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.MODEL.LABEL_SMOOTHING)
    criterion = torch.nn.CrossEntropyLoss()
    def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
        batch_size = len(Y)
        Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
        c = c.view(batch_size, 1).float() #censorship status, 0 or 1
        if S is None:
            S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
        # without padding, S(0) = S[0], h(0) = h[0]
        S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
        # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
        #h[y] = h(1)
        #S[1] = S(1) 
        uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
        neg_l = censored_loss + uncensored_loss
        loss = (1-alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        return loss
    criterion=nll_loss



    # resume from checkpoint (if applicable)
    max_accuracy = 0.0
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        # acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        acc1, acc5, loss = 0,0,0
        logger.info(f"Accuracy of the network: {acc1:.1f}%")
    #     if config.EVAL_MODE:
    #         return
    # EMA
    model_ema = None
    # if config.TRAIN.USE_EMA:
    #     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
    #     model_ema = ModelEmaV2(
    #         model_without_ddp, decay=config.TRAIN.EMA_DECAY, device=None)
    #     if config.MODEL.RESUME:
    #         load_checkpoint(config, model_ema, None, None, None, logger, use_ema=True)
    #         logger.info("Validating EMA checkpoint...")
    #         acc1, acc5, loss = validate(config, data_loader_val, model_ema.module, logger)
    #         logger.info(f"Accuracy of model ema: {acc1:.1f}%")

    # start training
    # num_epochs = config.TRAIN.EPOCHS + config.TRAIN.COOLDOWN_EPOCHS
    num_epochs=100
    logger.info("Start training")
    start_time = time.time()    


    result_csv_path=os.path.join(config.OUTPUT,"prediction.csv")
    df = None
    # df.to_csv(result_csv_path, index=False, header=False)
    
    # observed, predict, cencorship,c_index,pvalue_pred = validate(config, data_loader_val, model, logger,286)
    # global best_cin
    # print()
    # if isinstance(df, pd.DataFrame) is False:
    #     data = {'observed_{}'.format(str(286)):observed,
    #             'predict_{}'.format(str(286)): predict,
    #             'cencorship_{}'.format(str(286)): cencorship}
    #     df = pd.DataFrame(data)
    #     df.to_csv(result_csv_path, index=False, header=True)
    # return 

    for epoch in range(config.TRAIN.START_EPOCH, num_epochs):
        data_loader_train.sampler.set_epoch(epoch)


        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, logger, model_ema=model_ema, total_epochs=num_epochs)
        # if epoch>290:
        # save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema=model_ema, total_epochs=num_epochs)
        torch.cuda.synchronize()

        if epoch>=0 or epoch%40==0:
            observed, predict, cencorship,c_index,pvalue_pred = validate(config, data_loader_val, model, logger,epoch)
            global best_cin
            if c_index==best_cin or epoch==99:
                save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema=model_ema, total_epochs=num_epochs)

            
            if isinstance(df, pd.DataFrame) is False:
                data = {'observed_{}'.format(str(epoch)):observed,
                        'predict_{}'.format(str(epoch)): predict,
                        'cencorship_{}'.format(str(epoch)): cencorship}
                df = pd.DataFrame(data)
                df.to_csv(result_csv_path, index=False, header=True)
            else:
                df['observed_{}'.format(str(epoch))]=observed
                df['predict_{}'.format(str(epoch))] = predict
                df['cencorship_{}'.format(str(epoch))] = cencorship
                df.to_csv(result_csv_path, index=False, header=True)

            acc1, acc5, loss=0,0,0
            logger.info(f"Accuracy of the network: {acc1:.1f}%")
            max_accuracy = max(max_accuracy, acc1)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')
            # if model_ema is not None:
            #     ema_acc1, ema_acc5, ema_loss = validate(config, data_loader_val, model_ema.module, logger)
            #     logger.info(f"Accuracy of model ema: {ema_acc1:.1f}%")
        else:
            logger.info("Not at eval epoch yet!")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, logger, model_ema=None, total_epochs=None):
    """
    Trains the model for one epoch
    Args:
        config: CfgNode object, containing training and model configs
        model: the model to train
        criterion: the criterion for computing loss
        data_loader: torch.utils.data.DataLoader object
        optimizer: optimizer for training
        epoch: int, current epoch
        mixup_fn: mixup function used for mixup augmentation
        lr_scheduler: learning rate scheduler
        loss_scaler: loss scaler, used during mixed-precision training
        logger: logger object for logging
        model_ema (optional): EMA version of the model
        total_epochs (optional): int, total number of epochs
    """
    if total_epochs is None:
        total_epochs = config.TRAIN.EPOCHS
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()

    outputs_r=[]
    targets_r=[]
    

    for idx, (fe,pos, targets) in enumerate(data_loader):
        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)
        samples = [fe.cuda(),pos.cuda()]
        # with torch.no_grad():
        #     percent_to_keep = 1-config.DROP_RATE
        #     num_elements_to_keep = int(len(pos[0]) * percent_to_keep)
        #     random_indices = torch.randperm(len(pos[0]))[:num_elements_to_keep]
        #     fe=fe[:,random_indices]
        #     pos=pos[:,random_indices]

        label = targets[0].cuda().long()
        censorship= targets[2]
        censorship= torch.tensor(censorship).cuda().long()
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(samples)

            # print(torch.softmax(outputs,dim=1),targets)
            

        if config.TRAIN.ACCUMULATION_STEPS <= 1:
            ACCUMULATION_STEPS = 1
        else:
            ACCUMULATION_STEPS = config.TRAIN.ACCUMULATION_STEPS
        # loss = criterion(outputs, targets)
        output=output
        hazards = torch.sigmoid(output)
        S=torch.cumprod(1-hazards,dim=1)
        loss = criterion(hazards, S, label, censorship, alpha=0, eps=1e-7)
        if torch.isnan(loss).any() == True:
            optimizer.zero_grad()
            continue
        
        loss = loss / ACCUMULATION_STEPS
        total_loss = loss
        # grad_norm = loss_scaler(total_loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
        #                         parameters=model.parameters(), create_graph=False,
        #                         update_grad=(idx + 1) % ACCUMULATION_STEPS == 0)
        loss.backward()
        optimizer.step()

        outputs_r.append(output[0])
        targets_r.append(targets[0].cuda().long())

        if (idx + 1) % ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // ACCUMULATION_STEPS)
            # if model_ema is not None:
            #     model_ema.update(model)
        if loss_scaler.is_enabled():
            loss_scale_value = loss_scaler.state_dict()["scale"]
        else:
            loss_scale_value = 1.0

        torch.cuda.synchronize()

        loss_meter.update(loss * ACCUMULATION_STEPS, 1)
        # if grad_norm is not None:  # loss_scaler return None if not update
        #     norm_meter.update(grad_norm)#temp_delete
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()


        if idx % (config.PRINT_FREQ * ACCUMULATION_STEPS) == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{total_epochs}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
        del total_loss, output
        torch.cuda.empty_cache()
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    acc1, acc5 = accuracy(torch.vstack(outputs_r), torch.vstack(targets_r), topk=(1, 5))
    print(acc1.item())
    print(acc5.item())
    logger.info(f"EPOCH {epoch} acc {acc1.item()}")

    global best_train_loss,best_train_acc
    best_train_loss=min(loss_meter.avg,best_train_loss)
    best_train_acc=max(best_train_acc,acc1)
    logger.info(f' * best_train_loss {best_train_loss:.3f}')
    logger.info(f' * best_train_acc {best_train_acc:.3f}')
    
@torch.no_grad()
def validate(config, data_loader, model, logger,epoch):
    """
    Validates the accuracy of a model loaded with pre-trained checkpoint
    Args:
        config: CfgNode object, containing training and model configs
        data_loader: torch.utils.data.DataLoader object
        model: the model to validate
        logger: logger object for logging
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    end = time.time()

    from lifelines.utils import concordance_index
    observed_time=[]
    predicted_scores=[]
    event_occurred =[]
    hazards=[]
    labels=[]

    for idx, (fe,pos, target) in enumerate(data_loader):
        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)
        samples = [fe.cuda(),pos.cuda()]
        observed_time.append(float(target[1][0]))
        event_occurred.append(1-int(target[2][0]))
        # compute output
        # with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
        output = model(samples)

        target=target[0]
        # measure accuracy and record loss
        target=target.cuda().long()
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        output=output
        hazard=torch.sigmoid(output)
        hazards.append(hazard)
        S=torch.cumprod(1-torch.sigmoid(output).cpu(),dim=1)
        risk = -torch.sum(S, dim=1).cpu().item()
        predicted_scores.append(risk)
      

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'

                f'Mem {memory_used:.0f}MB')
    try:
        c_index = concordance_index_censored(np.array(event_occurred).astype(bool),np.array(observed_time),np.array(predicted_scores) ,tied_tol=1e-08)[0]
        pvalue_pred = cox_log_rank(np.array(predicted_scores), np.array(event_occurred), np.array(observed_time))
    except:
        c_index=0
        pvalue_pred=0
    logger.info(f' * cin {c_index:.3f}')
    logger.info(f' * pvalue_pred {pvalue_pred:.3f}')
    # return acc1_meter.avg, acc5_meter.avg, loss_meter.avg
    global best_cin
    if pvalue_pred<0.05:
        best_cin= max(c_index,best_cin)
    logger.info(f' * best_cin {best_cin:.3f}')

    return np.array(observed_time),np.array(predicted_scores),np.array(event_occurred),c_index,pvalue_pred


@torch.no_grad()
def throughput(config, data_loader, model, logger):
    
    """
    Computes the throughput of the model averaging over 30 steps
    Args:
        config: CfgNode object, containing training and model configs
        data_loader: torch.utils.data.DataLoader object
        model: the model to test
        logger: logger object for logging
    """
    model.eval()

    for idx, (fe,pos, _) in enumerate(data_loader):
        fe = fe.cuda(non_blocking=True)
        pos = pos.cuda(non_blocking=True)
        input=[fe,pos]
        batch_size = fe.shape[0]
        for i in range(50):
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                model(input)
        torch.cuda.synchronize()
        logger.info("throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                model(input)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


def run_all(config):
    """
    Run main() on all parallel gpus
    """

    # initialize distributed training and get the current GPU
    init_distributed_mode()
    config.defrost()
    # config.LOCAL_RANK = get_local_rank()
    config.freeze()

    seed = config.SEED + get_rank()
    print('Finished init distributed')
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    world_size = get_world_size()

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR
    linear_scaled_min_lr = config.TRAIN.BASE_LR/10
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # create output folder
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=get_rank(), name=f"{config.MODEL.NAME}")
    print('Logger created')
    if get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    import pykeops
    import tempfile
    with tempfile.TemporaryDirectory() as dirname:
        pykeops.set_build_folder(dirname)
        main(config, logger)


if __name__ == "__main__":
    args = parse_option()
    config = get_config(args)
    run_all(config)
