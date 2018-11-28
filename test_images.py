import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.sa_gan_l2h_unet import InpaintRUNNet, InpaintSADirciminator
from models.sa_gan import InpaintSANet, InpaintSADirciminator
from models.loss import SNDisLoss, SNGenLoss, ReconLoss, PerceptualLoss, StyleLoss
from util.logger import TensorBoardLogger
from util.config import Config
from data.inpaint_dataset import InpaintDataset
from util.evaluation import AverageMeter
from util.util import load_consistent_state_dict
from models.vgg import vgg16_bn

from evaluation import metrics
from PIL import Image
import pickle as pkl
import numpy as np
import logging
import time
import sys
import os

# python train inpaint.yml
config = Config(sys.argv[1])
logger = logging.getLogger(__name__)
time_stamp = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_dir = 'model_logs/test_{}_{}'.format(time_stamp, config.LOG_DIR)
result_dir = 'result_logs/{}'.format(config.MODEL_RESTORE[:config.MODEL_RESTORE.find('/')])
#tensorboardlogger = TensorBoardLogger(log_dir)
cuda0 = torch.device('cuda:{}'.format(config.GPU_IDS[0]))
cuda1 = torch.device('cuda:{}'.format(config.GPU_IDS[1]))
cpu0 = torch.device('cpu')
TRAIN_SIZES = ((64,64),(128,128),(256,256))
SIZES_TAGS = ("64x64", "128x128", "256x256")

def logger_init():
    """
    Initialize the logger to some file.
    """
    logging.basicConfig(level=logging.INFO)

    logfile = 'logs/{}_{}.log'.format(time_stamp, config.LOG_DIR)
    fh = logging.FileHandler(logfile, mode='w')
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def img2photo(imgs):
    return ((imgs+1)*127.5).transpose(1,2).transpose(2,3).detach().cpu().numpy()

def validate(nets, loss_terms, opts, dataloader, epoch, network_type, devices=(cuda0,cuda1), batch_n="whole_test_show"):
    """
    validate phase
    """
    netD, netG = nets["netD"], nets["netG"]
    ReconLoss, DLoss, PercLoss, GANLoss, StyleLoss = loss_terms['ReconLoss'], loss_terms['DLoss'], loss_terms["PercLoss"], loss_terms["GANLoss"], loss_terms["StyleLoss"]
    optG, optD = opts['optG'], opts['optD']
    device0, device1 = devices
    netG.to(device0)
    netD.to(device0)
    netG.eval()
    netD.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = {"g_loss":AverageMeter(),"p_loss":AverageMeter(), "s_loss":AverageMeter(), "r_loss":AverageMeter(), "whole_loss":AverageMeter(), "d_loss":AverageMeter()}

    netG.train()
    netD.train()
    end = time.time()
    val_save_dir = os.path.join(result_dir, "val_{}_{}".format(epoch, batch_n if isinstance(batch_n, str) else batch_n+1))
    val_save_real_dir = os.path.join(val_save_dir, "real")
    val_save_gen_dir = os.path.join(val_save_dir, "gen")
    val_save_comp_dir = os.path.join(val_save_dir, "comp")
    for size in SIZES_TAGS:
        if not os.path.exists(os.path.join(val_save_real_dir, size)):
            os.makedirs(os.path.join(val_save_real_dir, size))
        if not os.path.exists(os.path.join(val_save_gen_dir, size)):
            os.makedirs(os.path.join(val_save_gen_dir, size))
        if not os.path.exists(os.path.join(val_save_comp_dir, size)):
            os.makedirs(os.path.join(val_save_comp_dir, size))
    info = {}
    t = 0
    for i, (ori_imgs, ori_masks) in enumerate(dataloader):
        data_time.update(time.time() - end)
        pre_imgs = ori_imgs
        pre_complete_imgs = (pre_imgs / 127.5 - 1)

        for s_i, size in enumerate(TRAIN_SIZES):

            masks = ori_masks['val']
            masks = F.interpolate(masks, size)
            masks = (masks > 0).type(torch.FloatTensor)
            imgs = F.interpolate(ori_imgs, size)
            if imgs.size(1) != 3:
                print(t, imgs.size() )
            pre_inter_imgs = F.interpolate(pre_complete_imgs, size)

            imgs, masks, pre_complete_imgs, pre_inter_imgs = imgs.to(device0), masks.to(device0), pre_complete_imgs.to(device0), pre_inter_imgs.to(device0)
            #masks = (masks > 0).type(torch.FloatTensor)

            #imgs, masks = imgs.to(device), masks.to(device)
            imgs = (imgs / 127.5 - 1)
            # mask is 1 on masked region
            # forward
            if network_type == 'l2h_unet':
                recon_imgs = netG(imgs, masks, pre_complete_imgs, pre_inter_imgs, size)
            elif network_type == 'l2h_gated':
                recon_imgs = netG(imgs, masks, pre_inter_imgs)
            elif network_type == 'sa_gated':
                recon_imgs, _ = netG(imgs, masks)
            complete_imgs = recon_imgs * masks + imgs * (1 - masks)


            pos_imgs = torch.cat([imgs, masks, torch.full_like(masks, 1.)], dim=1)
            neg_imgs = torch.cat([recon_imgs, masks, torch.full_like(masks, 1.)], dim=1)
            pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

            pred_pos_neg = netD(pos_neg_imgs)
            pred_pos, pred_neg = torch.chunk(pred_pos_neg,  2, dim=0)


            g_loss = GANLoss(pred_neg)

            r_loss = ReconLoss(imgs, recon_imgs, recon_imgs, masks)

            imgs, recon_imgs, complete_imgs = imgs.to(device1), recon_imgs.to(device1), complete_imgs.to(device1)
            p_loss = PercLoss(imgs, recon_imgs) + PercLoss(imgs, complete_imgs)
            s_loss = StyleLoss(imgs, recon_imgs) + StyleLoss(imgs, complete_imgs)
            p_loss, s_loss = p_loss.to(device0), s_loss.to(device0)
            imgs, recon_imgs, complete_imgs = imgs.to(device0), recon_imgs.to(device0), complete_imgs.to(device0)

            whole_loss = r_loss + p_loss #g_loss + r_loss

            # Update the recorder for losses
            losses['g_loss'].update(g_loss.item(), imgs.size(0))
            losses['r_loss'].update(r_loss.item(), imgs.size(0))
            losses['p_loss'].update(p_loss.item(), imgs.size(0))
            losses['s_loss'].update(s_loss.item(), imgs.size(0))
            losses['whole_loss'].update(whole_loss.item(), imgs.size(0))

            d_loss = DLoss(pred_pos, pred_neg)
            losses['d_loss'].update(d_loss.item(), imgs.size(0))
            pre_complete_imgs = complete_imgs
            # Update time recorder
            batch_time.update(time.time() - end)


            # Logger logging


            #if t < config.STATIC_VIEW_SIZE:
            print(i, size)
            real_img = img2photo(imgs)
            gen_img = img2photo(recon_imgs)
            comp_img = img2photo(complete_imgs)

            real_img = Image.fromarray(real_img[0].astype(np.uint8))
            gen_img = Image.fromarray(gen_img[0].astype(np.uint8))
            comp_img = Image.fromarray(comp_img[0].astype(np.uint8))
            real_img.save(os.path.join(val_save_real_dir, SIZES_TAGS[s_i], "{}.png".format(i)))
            gen_img.save(os.path.join(val_save_gen_dir, SIZES_TAGS[s_i], "{}.png".format(i)))
            comp_img.save(os.path.join(val_save_comp_dir, SIZES_TAGS[s_i], "{}.png".format(i)))

            end = time.time()


def main():
    logger_init()
    dataset_type = config.DATASET
    batch_size = config.BATCH_SIZE

    # Dataset setting
    logger.info("Initialize the dataset...")
    val_dataset = InpaintDataset(config.DATA_FLIST[dataset_type][1],\
                                    {mask_type:config.DATA_FLIST[config.MASKDATASET][mask_type][1] for mask_type in ('val',)}, \
                                    resize_shape=tuple(config.IMG_SHAPES), random_bbox_shape=config.RANDOM_BBOX_SHAPE, \
                                    random_bbox_margin=config.RANDOM_BBOX_MARGIN,
                                    random_ff_setting=config.RANDOM_FF_SETTING)
    val_loader = val_dataset.loader(batch_size=1, shuffle=False,
                                        num_workers=1)
    #print(len(val_loader))

    ### Generate a new val data

    logger.info("Finish the dataset initialization.")

    # Define the Network Structure
    logger.info("Define the Network Structure and Losses")
    whole_model_path = 'model_logs/{}'.format(config.MODEL_RESTORE)
    nets = torch.load(whole_model_path)
    netG_state_dict, netD_state_dict = nets['netG_state_dict'], nets['netD_state_dict']
    if config.NETWORK_TYPE == "l2h_unet":
        netG = InpaintRUNNet(n_in_channel=config.N_CHANNEL)
        netG.load_state_dict(netG_state_dict)

    elif config.NETWORK_TYPE == 'sa_gated':
        netG = InpaintSANet()
        load_consistent_state_dict(netG_state_dict, netG)
        #netG.load_state_dict(netG_state_dict)

    netD = InpaintSADirciminator()
    netVGG = vgg16_bn(pretrained=True)


    #netD.load_state_dict(netD_state_dict)
    logger.info("Loading pretrained models from {} ...".format(config.MODEL_RESTORE))

    # Define loss
    recon_loss = ReconLoss(*(config.L1_LOSS_ALPHA))
    gan_loss = SNGenLoss(config.GAN_LOSS_ALPHA)
    perc_loss = PerceptualLoss(weight=config.PERC_LOSS_ALPHA,feat_extractors = netVGG.to(cuda1))
    style_loss = StyleLoss(weight=config.STYLE_LOSS_ALPHA, feat_extractors = netVGG.to(cuda1))
    dis_loss = SNDisLoss()
    lr, decay = config.LEARNING_RATE, config.WEIGHT_DECAY
    optG = torch.optim.Adam(netG.parameters(), lr=lr, weight_decay=decay)
    optD = torch.optim.Adam(netD.parameters(), lr=4*lr, weight_decay=decay)
    nets = {
        "netG":netG,
        "netD":netD,
        "vgg":netVGG
    }

    losses = {
        "GANLoss":gan_loss,
        "ReconLoss":recon_loss,
        "StyleLoss":style_loss,
        "DLoss":dis_loss,
        "PercLoss":perc_loss

    }
    opts = {
        "optG":optG,
        "optD":optD,

    }
    logger.info("Finish Define the Network Structure and Losses")

    # Start Training
    logger.info("Start Validation")

    validate(nets, losses, opts, val_loader,0 , config.NETWORK_TYPE,devices=(cuda0,cuda1))

if __name__ == '__main__':
    main()
