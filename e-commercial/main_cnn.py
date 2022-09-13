# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np
import torchvision
import cv2
from IPython import embed

# os.environ["CUDA_VISIBLE_DEVICES"] = '0, 2, 4, 7'

import torch
import torch.backends.cudnn as cudnn
#import torch.distributed as dist

from timm.utils import accuracy, AverageMeter
from torchvision import transforms
import torchvision
import torch.nn.functional as F

from config import get_config
from models import build_model_new
from models import craft_utils, imgproc
from models.loss import KL_loss, Maploss
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor

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
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')
    # parser.add_argument("--loss", type=str, required=True, help='debugging for different loss')

    # about dataset
    parser.add_argument('--dataset', type=str, default='imagenet', help='name of dataset')
    parser.add_argument('--head', type=str, default='denseNet_15layer', help='name of dataset')
    parser.add_argument('--datanum', type=int, default=972, help='num of dataset')
    
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model_new(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.DataParallel(model)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # criterion = KL_Loss()
    criterion = ""

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
        print("resuming")
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        validate(config, data_loader_val, model)
        # logger.info(f"loss of the network on the {len(dataset_val)} test images: {loss:.1f}%")
        if config.EVAL_MODE:
            print("eval mode")
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        print("traning %d epoch" % epoch)
        #data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler)
        print("training %d epoch done" % epoch)
        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

    validate(config, data_loader_val, model)
    print("imgs saved")

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
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    cnt = 0
    test_loc =  config.OUTPUT + '/test/'
    # print("saving in test_loc", test_loc)
    # if not os.path.exists(test_loc):
    #     os.makedirs(test_loc)
    for idx, (idx_name, samples, targets, ocr_target) in enumerate(data_loader):
        #print(samples.shape, targets.shape)
        # idx_name = idx_name.tolist()
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        outputs = model(samples, targets)
        # print("output.shape", outputs.shape)
        targets = transforms.Resize(56)(targets)
        # if epoch == 3 and cnt == 0:
        #     torchvision.utils.save_image(samples, test_loc+str(epoch)+"_epoch_"+str(cnt)+'_image_'+str(idx_name)+'.jpg')
        #     torchvision.utils.save_image(targets, test_loc+str(epoch)+"_epoch_"+str(cnt)+'_map_'+str(idx_name)+'.jpg')
        #cnt+=1
        loss = KL_loss(outputs, targets)
        # print("looking loss", idx, loss)
        # print("saliency mode", saliency_loss)
        # ocr_target = {key: ocr_target[key].cuda(non_blocking=True) for key in ocr_target}
        # if mixup_fn is not None:
        #    samples, targets = mixup_fn(samples, targets)

        # outputs, attn_loss, ocr_out, feature = model(samples, targets)
        
        # print("output.size", output.shape)
        # print("ocr_out.shape", ocr_out.shape)
        # outputs, attn_loss = model(samples, targets)
        # outputs = model(samples, targets)
        '''
        attn_loss_meter.update(attn_loss, targets.size(0))
        ocr_loss_meter.update(ocr_loss, targets.size(0))
        saliency_loss_meter.update(saliency_loss, targets.size(0))
        # print("detect loss from outside", saliency_loss, "\n", attn_loss, "\n", ocr_loss)
        
        if config.LOSS == 'ocr':
            ocr_out = model(samples, targets, config.LOSS)
            out1 = ocr_out[:, :, :, 0].cuda()
            out2 = ocr_out[:, :, :, 1].cuda()
            gah_label = ocr_target["gah_label"].resize_(out2.size())
            gh_label = ocr_target["gh_label"].resize_(out1.size())
            mask = ocr_target["mask"]
            ocr_loss = criterion[1](gh_label, gah_label, out2, out1, mask)
            print("ocr mode", ocr_loss)
            loss = ocr_loss
        elif config.LOSS == 'saliency':
            outputs = model(samples, targets, config.LOSS)
            targets = transforms.Resize(56)(targets)
            saliency_loss = criterion[0](outputs, targets)
            print("saliency mode", saliency_loss)
            loss = saliency_loss
        elif config.LOSS == 'attn':
            attn_loss = model(samples, targets, config.LOSS)
            print("attention mode", attn_loss)
            loss = attn_loss
        else:
            outputs, attn_loss, ocr_out, feature = model(samples, targets, config.LOSS)
            print("normal mode")
            loss = saliency_loss + attn_loss + ocr_loss
        '''
        optimizer.zero_grad()

        if config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(amp.master_params(optimizer))
        else:
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
                f'ocr_loss {ocr_loss_meter.val:.4f} ({ocr_loss_meter.avg:.4f})\t'
                f'saliency_loss {saliency_loss_meter.val:.4f} ({saliency_loss_meter.avg:.4f})'
                )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    

@torch.no_grad()
def validate(config, data_loader, model):
    # waiting
    saliency_loc = config.OUTPUT + '/ans/saliency/'
    if not os.path.exists(saliency_loc):
        os.makedirs(saliency_loc)
    model.eval()
    with torch.no_grad():
        for idx, (idx_name, images, target, ocr_target) in enumerate(data_loader):
            idx_name = idx_name.tolist()
            # print(idx_name, "3")
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            # torchvision.utils.save_image(images[0], saliency_loc + str(idx_name) + '_ori.jpg')
            #print(images)
            #non = target.nonzero()
            #print(non, " none \n")
            #print(target[non[0][0]][non[0][1]][non[0][2]][non[0][3]])
            # ocr_target = {key: ocr_target[key].cuda(non_blocking=True) for key in ocr_target}
            #output, attn_loss, ocr_out, feature = model(images, target)
            
            
            output = model(images, target)
            target = transforms.Resize(56)(target)
            loss = KL_loss(output, target)
            print("loss", idx_name, "validate", loss)
            saliency_map = np.array(F.interpolate(output, size=images.size()[2:], mode='bilinear', align_corners=False).cpu())
            saliency_map = np.ascontiguousarray(saliency_map)
            saliency_map *= 255
            saliency_maps = saliency_map.astype(np.uint8)
            # embed()
            for i, saliency_map in enumerate(saliency_maps):
                print("saving", saliency_loc + str(idx_name[i]) + '.jpg')
                path = saliency_loc + str(idx_name[i]) + '.jpg'
                re = cv2.imwrite(path, saliency_map[0])
            # print(re)
            '''
            ocr_out = ocr_out.cpu()
            images = images.cpu()
            score_text = ocr_out[0, :, :, 0].cpu().data.numpy()
            score_link = ocr_out[0, :, :, 1].cpu().data.numpy()

            boxes, polys = craft_utils.getDetBoxes(score_text, score_link, 0.7, 0.4, 0.4, False)
            # print(boxes.size())

            # boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
            # polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
            # print(boxes)
            for k in range(len(polys)):
                if polys[k] is None: polys[k] = boxes[k]
            
            render_img = score_text.copy()
            render_img = np.hstack((render_img, score_link))
            ret_score_text = imgproc.cvt2HeatmapImg(render_img)
            # output, attn_loss = model(images, target)
            # output = model(images, target)
            image=cv2.imread("/home/liyifei/experiment/EC/swin-transformer/ECdata/ALLSTIMULI/"+str(idx_name) + '.jpg')
            image=cv2.resize(image,(896,896))
            img = np.array(images)
            boxes = polys
            # make result file list
            # filename, file_ext = os.path.splitext(os.path.basename(img_file))
            # dirname=/temp_disk2/home/leise/ali/chineseocr1/CRAFT-Reimplementation-master/result/img/
            # result directory
            # res_file = '/temp_disk2/leise/ali/CRAFT-Reimplementation-master/data/result1/txt/' + "res_" + filename + '.txt'
            # res_img_file = dirname + "res_" + filename + '.jpg'
            if not os.path.exists('/home/liyifei/experiment/EC/swin-transformer/output/ans/ocr/box/'):
                os.makedirs('/home/liyifei/experiment/EC/swin-transformer/output/ans/ocr/box/')
                os.makedirs('/home/liyifei/experiment/EC/swin-transformer/output/ans/ocr/anchor/')
            res_img_file = r'/home/liyifei/experiment/EC/swin-transformer/output/ans/ocr/box/' + str(idx_name) + '.jpg'
            res_file = r'/home/liyifei/experiment/EC/swin-transformer/output/ans/ocr/anchor/' + str(idx_name) + '.txt'
            # if not os.path.isdir(dirname):
            # os.mkdir(dirname)

            with open(res_file, 'w') as f:
                for i, box in enumerate(boxes):
                    poly = np.array(box).astype(np.int32).reshape((-1))
                    strResult = ','.join([str(p) for p in poly]) + '\r\n'
                    print("str result", strResult)
                    f.write(strResult)
                    poly = poly.reshape(-1, 2)
                    cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
                    ptColor = (0, 255, 255)
                    # if verticals is not None:
                    # if verticals[i]:
                    # ptColor = (255, 0, 0)

                    # if texts is not None:
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # font_scale = 0.5
                    # cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                    # cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

            # Save result image
            # img=cv2.addWeighted(ret_score_text,0.5,img,0.5,0)
            cv2.imwrite(res_img_file, img)
            '''

        # print('time:%s' % ((T2 - T1)*1000))
        # print('cc:%s' %np.mean(self.cc))

    return


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
