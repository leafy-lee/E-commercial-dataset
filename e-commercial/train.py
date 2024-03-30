# --------------------------------------------------------
# based on Swin Transformer
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np
import torchvision
import cv2


import torch
import torch.backends.cudnn as cudnn

from timm.utils import accuracy, AverageMeter
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from PIL import Image

from config import get_config
from models import build_model
from models import craft_utils, imgproc
from models.loss import KL_loss, Maploss
from models.metric import calCC, calKL
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, load_checkpoint_finetune, load_checkpoint_eval


try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    # parser.add_argument('--eval-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--finetune', help='finetune from checkpoint')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='./output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    
    # about dataset
    parser.add_argument('--dataset', type=str, default='imagenet', help='name of dataset')
    parser.add_argument('--datanum', type=int, default=972, help='num of dataset')
    parser.add_argument('--num_epoch', type=int, default=50, help='num of epoch')
    parser.add_argument('--head', type=str, default='denseNet_15layer', help='head')
    
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config, logger)
    # dataset_train, data_loader_train, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    model = torch.nn.DataParallel(model)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    if config.EVAL_MODE:
        print("test model")
        dataset_val, data_loader_val, _ = build_loader(config)
        load_checkpoint_eval(config, model_without_ddp, optimizer, logger)
        validate_article(config, dataset_train, model)
        return

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = Maploss()
    


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
            

    '''
    if config.MODEL.FINETUNE:
        print("FINETUNE")
        load_checkpoint_finetune(config, model_without_ddp, optimizer, lr_scheduler, logger)
    '''
    if config.MODEL.RESUME:
        print("resuming")
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        validate_article(config, data_loader_val, data_loader_train, model)
        # logger.info(f"loss of the network on the {len(dataset_val)} test images: {loss:.1f}%")
        if config.EVAL_MODE:
            print("eval mode")
            return

    
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            print("fake test first")
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)
            validate(config, data_loader_val, model, epoch)
            print("imgs saved")  
        print("traning %d epoch" % epoch)
        #data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        print("training %d epoch done" % epoch)
    

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    ocr_loss_meter = AverageMeter()
    attn_loss_meter = AverageMeter()
    saliency_loss_meter = AverageMeter()
    saliency_kl_meter = AverageMeter()
    saliency_cc_meter = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    cnt = 0
    test_loc =  config.OUTPUT + '/test/'
    if not os.path.exists(test_loc):
        os.makedirs(test_loc)
    test_map = []
    test_name = None
    cnt = 1
    
    for idx, (idx_name, samples, targets, ocr_target) in enumerate(data_loader):
    
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        
        outputs, attn_loss, ocr_out = model(samples, targets)
        # print(f"{ocr_out.shape}")
        test_map = outputs
        if test_name is None:
            test_name = idx_name
        
        targets = transforms.Resize(56)(targets)
        
        saliency_loss = KL_loss(outputs, targets)
        out1 = ocr_out[:, :, :, 0].cuda()
        out2 = ocr_out[:, :, :, 1].cuda()
        gah_label = ocr_target["gah_label"].resize_(out2.size()).cuda()
        gh_label = ocr_target["gh_label"].resize_(out1.size()).cuda()
        mask = ocr_target["mask"].cuda()
        ocr_loss = criterion(gh_label, gah_label, out2, out1, mask)
        
        # scale
        ocr_loss *= 3
        
        loss = saliency_loss + attn_loss + ocr_loss
        saliency_kl = calKL(targets, outputs, True)
        saliency_cc = calCC(targets, outputs, True)
        
        attn_loss_meter.update(attn_loss, targets.size(0))
        ocr_loss_meter.update(ocr_loss, targets.size(0))
        saliency_loss_meter.update(saliency_loss, targets.size(0))
        saliency_cc_meter.update(saliency_cc, targets.size(0))
        saliency_kl_meter.update(saliency_kl, targets.size(0))
        # print(saliency_loss , attn_loss,"here comes the bugs")
        # breakpoint()
        if len(attn_loss.shape) == 0:
            attn_loss = attn_loss
        else:
            attn_loss = sum(attn_loss)/4
        if config.TRAIN.START_EPOCH == epoch and cnt:
            print("displaying attnloss", attn_loss)
            cnt = 0
        # print("now adding", attn_loss)
        loss = saliency_loss + attn_loss * 0.2
        attn_loss_meter.update(attn_loss, targets.size(0))
        saliency_loss_meter.update(saliency_loss, targets.size(0))
        optimizer.zero_grad()

        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB\t'
                f'attn_loss {attn_loss_meter.val:.4f} ({attn_loss_meter.avg:.4f})\t'
                f'saliency_loss {saliency_loss_meter.val:.4f} ({saliency_loss_meter.avg:.4f})'      
                )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    # print(test_name,test_map.shape,"testing for epoch")
    test_map = np.ascontiguousarray(test_map.detach().cpu().numpy())
    test_map *= 255
    test_map = test_map.astype(np.uint8)
    # test_ans = np.ascontiguousarray(test_ans.detach().cpu().numpy())
    for i, maps in enumerate(test_map):
        name_i = test_name[i].item()
        path = test_loc + str(epoch) + f"_epoch/{name_i}.jpg"
        # ans_path = test_loc + str(epoch)+"_epoch_test_" + name_i
        re = cv2.imwrite(path, maps[0])
        #  re = cv2.imwrite(ans_path, test_ans[i][0])
        # print(path, "saved")
    print("saving batch_size images for browsing in epoch %d"%epoch)
    

@torch.no_grad()
def validate(config, data_loader, model, epoch):
    # waiting
    criterion = Maploss()
    saliency_loc = config.OUTPUT + f'/ans/{epoch}/new_saliency/'
    if not os.path.exists(saliency_loc):
        os.makedirs(saliency_loc)
    save_loc = config.OUTPUT + f'/ans/{epoch}/new_ocr/'
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    model.eval()
    print("save loc", save_loc, "saliency loc", saliency_loc)  
    klall = 0
    ccall = 0
    cnt = 0
    with torch.no_grad():
        for idx, (idx_name, images, target, ocr_target) in enumerate(data_loader):
            if epoch == 0 and cnt > 1:
                break
            idx_name = idx_name.tolist()          
            cnt += 1
            # print(idx_name, "3")
            images = images.cuda(non_blocking=True)
            targets = target.cuda(non_blocking=True)
            
            output, attn_loss, ocr_out = model(images, targets)
            
            target = transforms.Resize(56)(targets)
            loss = KL_loss(output, target)
            out1 = ocr_out[:, :, :, 0].cpu()
            out2 = ocr_out[:, :, :, 1].cpu()
            gah_label = ocr_target["gah_label"].resize_(out2.size())
            gh_label = ocr_target["gh_label"].resize_(out1.size())
            mask = ocr_target["mask"]
            
            ocr_loss = criterion(gh_label, gah_label, out2, out1, mask)
            target_metric = transforms.Resize(720)(targets)
            output_metric = transforms.Resize(720)(output)
            saliency_kl = calKL(target_metric, output_metric, False)
            saliency_cc = calCC(target_metric, output_metric, False)
            klall += saliency_kl
            ccall += saliency_cc
            
            saliency_map = np.array(F.interpolate(output, size=(720, 720), mode='bilinear', align_corners=False).cpu())
            saliency_map = np.ascontiguousarray(saliency_map)
            saliency_map *= 255
            saliency_maps = saliency_map.astype(np.uint8)
            
            for i, saliency_map in enumerate(saliency_maps):
                path = saliency_loc + str(idx_name[i]) + '.jpg'
                re = cv2.imwrite(path, saliency_map[0])
                
            ocr_outs = ocr_out.cpu()
            for i, ocr_out in enumerate(ocr_outs):
                score_text = ocr_out[:, :, 0].cpu().data.numpy()
                score_link = ocr_out[:, :, 1].cpu().data.numpy()
    
                boxes, polys = craft_utils.getDetBoxes(score_text, score_link, 0.7, 0.4, 0.4, False)
                for k in range(len(polys)):
                    if polys[k] is None: polys[k] = boxes[k]
                
                render_img = score_text.copy()
                render_img = np.hstack((render_img, score_link))
                ret_score_text = imgproc.cvt2HeatmapImg(render_img)
            
                image=cv2.imread("/mnt/hdd1/yifei/DATA/ECdata/ALLSTIMULI/"+str(idx_name[i]) + '.jpg')
                image=cv2.resize(image,(224,224))
                img = np.array(image)
                boxes = polys
                res_img_file = save_loc + str(idx_name[i]) + '.jpg'
                res_file = save_loc + str(idx_name[i]) + '.txt'
    
                with open(res_file, 'w') as f:
                    for j, box in enumerate(boxes):
                        poly = np.array(box).astype(np.int32).reshape((-1))
                        strResult = ','.join([str(p) for p in poly]) + '\r\n'
                        f.write(strResult)
                        poly = poly.reshape(-1, 2)
                        cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                        ptColor = (0, 255, 255)
                cv2.imwrite(res_img_file, img)
    logger.info("average kl, cc is %f, %f"%(klall/cnt, ccall/cnt))

    return

from torchvision import transforms
def validate_article(config, data_loader, model):
    # waiting
    criterion = Maploss()
    saliency_loc = config.OUTPUT + '/output/article_saliency/'
    if not os.path.exists(saliency_loc):
        os.makedirs(saliency_loc)
    model.eval()
    print("saliency loc", saliency_loc)
    times = 0
    cnt = 0
    print("testing")
    with torch.no_grad():
        for (name, images, _) in data_loader:
            # print(name, images.shape, _.shape)
            images = images.cuda(non_blocking=True)
            images = images.unsqueeze(0)
            st = time.time()
            
            output, __ = model(images, _)
            ed = time.time()
            print("time", ed-st)
            output = transforms.Resize((720,720))(output)
            
            # saliency_map = F.interpolate(outputs, size=(720, 720), mode='bilinear', align_corners=False).cpu().numpy()
            saliency_map = np.ascontiguousarray(output.detach().cpu().numpy())
            saliency_map *= 255
            saliency_maps = saliency_map.astype(np.uint8)
            tmp_time = ed-st
            times += len(saliency_maps)*tmp_time
            cnt += len(saliency_maps)
            # print(name, len(saliency_maps))
            
            name_i = name.split("/")[-1]
            path = saliency_loc + name_i
            print(type(saliency_map[0]),os.path.exists(saliency_loc))
            # embed()
            re = cv2.imwrite(path, saliency_map[0][0])
            print(path, "saved", re)
    print(times/cnt)
    print("images saved")


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    # torch.cuda.set_device(config.LOCAL_RANK)
    # torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier()

    # seed = config.SEED + dist.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR # * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR# * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR# * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    if True:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    # print(config)
    main(config)
