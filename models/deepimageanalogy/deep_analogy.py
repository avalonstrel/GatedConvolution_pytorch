import torch
import torchvision.transforms as transforms
from patchmatch.PatchMatchOrig import PatchMatch
from feat_extractor import multi_layer_vgg19_bn, multi_layer_vgg19
from PIL import Image
import numpy as np
import cv2
import sys
SHAPE = (224, 224)
PSIZE = 3
ALPHA = [0.1, 0.6, 0.7, 0.8, 0.9]
kappa = 300
tau = 0.05
transforms_fun = transforms.Compose([transforms.Resize(SHAPE), transforms.ToTensor()])


def torch2numpy(*feats):
    np_feats = []
    for feat in feats:
        np_feats.append(feat[0].transpose(0,1).transpose(1,2).detach().numpy())
    return np_feats

def numpy2torch(*feats):
    torch_feats = []
    for feat in feats:
        torch_feats.append(torch.tensor(feat.transpose(2,0,1)).unsqueeze(0))
    return torch_feats

def deep_image_analogy(img_a, img_bp, multi_feat_extractor):
    feats_a, feats_bp = multi_feat_extractor(img_a), multi_feat_extractor(img_bp)
    n_feats = len(feats_a)
    feats_ap, feats_b = [feats_a[n_feats-1].clone()], [feats_bp[n_feats-1].clone()]
    l = n_feats - 1
    print("Patch Match Initialize...")
    pmab = PatchMatch(*torch2numpy(feats_a[l], feats_ap[n_feats-l-1],  feats_b[n_feats-l-1], feats_bp[l]), PSIZE)
    pmba = PatchMatch(*torch2numpy(feats_b[n_feats-l-1], feats_bp[l],  feats_a[l], feats_ap[n_feats-l-1]), PSIZE)
    print("Patch Match Initialize Finish")
    for l in range(n_feats-1, -1, -1):
        # Compute the mapping functions
        # NNF Search
        # the lth layer
        print("Layer {}, propagation.".format(l))
        pmab.propagate(), pmba.propagate()
        #print(pmab.nnf)
        if l > 0:
            # Reconstruction
            # the l-1(th) layer
            print("Layer {}, reconstruction.".format(l))
            feats_ap.append(reconstruction(feats_a[l-1], feats_bp[l], pmab, multi_feat_extractor.get_layer(l-1), l))
            feats_b.append(reconstruction(feats_bp[l-1], feats_a[l], pmba, multi_feat_extractor.get_layer(l-1), l))

            # NNF Upsampling
            feat_a, feat_ap, feat_b, feat_bp = feats_a[l-1], feats_ap[n_feats-l],  feats_b[n_feats-l], feats_bp[l-1]
            pmab.upsample_update(*torch2numpy(feat_a, feat_ap, feat_b, feat_bp))
            pmba.upsample_update(*torch2numpy(feat_b, feat_bp, feat_a, feat_ap))


    return pmab, pmba


def reconstruction(feat_a_pre, feat_bp, pmab, layer, l):
    """
    Reconstruct F_AP^[L-1] from F_A^[L-1], F_BP^[L] and  \phi_[a->b]^L, CNN_[L-1]^[L]
    """
    re_feat_bp_pre = deconvolve(feat_bp, pmab, layer, tuple(feat_a_pre.size()))
    w_a_pre = weight_from_feat(feat_a_pre) * ALPHA[l-1]
    feat_ap_pre = weighted_blend(feat_a_pre, w_a_pre, re_feat_bp_pre)
    return feat_ap_pre

def weight_from_feat(feat):
    """
    Compute a weight map from feature
    """
    feat2 = torch.pow(feat, 2)
    m_a = 1 / (1 + torch.exp(-kappa*feat2/feat2.max()  - tau))

    return m_a


def weighted_blend(feat_a_pre, w_a_pre, re_feat_bp_pre):
    """
    Compute the l-1 th Recovered by weighted blend
    Params:
        feat_a_pre(torch.Tensor): l-1 layer feat_a F_A^[L-1]
        w_a_pre(torch.Tensor): W_A^[L-1]
        re_feat_bp_pre(torch.Tensor): R_BP^[L-1]
    Return:
        F_AP^[L-1]
    """
    return feat_a_pre*w_a_pre + re_feat_bp_pre*(1-w_a_pre)

def deconvolve(feat_bp, pmab, layer, shape):
    """
    Return the (i-1)th layer recovered feature from ith layer feature and \phi and ith CNN layer
    Params:
        feat_bp(torch.Tensor): F_BP^L
        pmab(PatchMatch): \phi_[a->b]^L
        layer(torch.Tensor): CNN_[L-1]^L
    Return:
        R_BP^[L-1]
    """
    feat_bp = torch2numpy(feat_bp)[0]
    print(feat_bp.shape)
    feat_bp_a = numpy2torch(pmab.reconstruct_avg(feat_bp))[0]
    re_feat_bp = torch.nn.Parameter(torch.zeros(*shape))

    opt = torch.optim.Adam([re_feat_bp], lr=1e-3, weight_decay=0)
    iters = 1000
    for i in range(iters):
        opt.zero_grad()
        out_re_feat_bp = layer(re_feat_bp)
        loss = torch.sum(torch.pow(out_re_feat_bp-feat_bp_a, 2))
        loss.backward()
        opt.step()
    print("loss term:",loss.item())
    #print(re_feat_bp)
    return re_feat_bp.detach()

def save_image(name, array):
    img = Image.fromarray(array.astype(np.uint8))
    img.save(name)

def main():
    multi_feat_extractor = multi_layer_vgg19_bn(pretrained=True)
    img_a = Image.open(sys.argv[1]).convert("RGB")
    img_bp = Image.open(sys.argv[2]).convert("RGB")

    img_a, img_bp = transforms_fun(img_a).unsqueeze(0), transforms_fun(img_bp).unsqueeze(0)

    pmab, pmba = deep_image_analogy(img_a, img_bp, multi_feat_extractor)
    img_a = torch2numpy(img_a)[0]*255
    img_bp = torch2numpy(img_bp)[0]*255
    save_image("re_img_a.png", img_a)
    save_image("re_img_bp.png", img_bp)
    img_ap = pmab.reconstruct_avg(img_bp)
    img_b = pmba.reconstruct_avg(img_a)

    save_image("img_b.png", img_b)
    save_image("img_ap.png", img_ap)

if __name__ == '__main__':
    main()
