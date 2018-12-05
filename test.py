import torch
from evaluation import metrics
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images'))

args = parser.parse_args()
path1, path2 = args.path
#path1:real data, path2:generated data
# inception_score, std_is = None, None
# inception_score, std_is = metrics['is'](path1,cuda=False )
# print("True IS Mean:{}, IS STD:{}".format(inception_score2, std_is2))
# inception_score2, std_is2 = metrics['is'](path2, cuda=False )
# print("Generated IS Mean:{}, IS STD:{}".format(inception_score2, std_is2))
#fid_score = None
#ssim_score = None
fid_score = metrics['fid']([path1, path2],cuda=True)
print("Generated FID:{}".format(fid_score))
ssim_score = metrics['ssim']([path1, path2])
print("Generated SSIM:{}".format(ssim_score))
psnr_score = metrics['psnr']([path1, path2])
print("Generated PSNR:{}".format(psnr_score))
meanl1_error = metrics['meanl1']([path1, path2])
print("Generated Mean L1:{}".format(meanl1_error))
