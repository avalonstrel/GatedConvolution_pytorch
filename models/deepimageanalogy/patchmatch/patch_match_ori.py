import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
SHAPE = (224,224)
PSIZE = 3
#cuda0 = torch.device("cuda:0")
cpu = torch.device("cpu")
device = cpu

def feat_patch_distance(feat1, feat2, feat1_, feat2_, pos, pos_p, psize):
    """
    Return a distance
    """
    y, x = pos
    yp, xp = pos_p
    #print(pos, pos_p)
    return torch.sum(torch.pow(feat1[y:y+psize, x:x+psize,:]-feat2[yp:yp+psize, xp:xp+psize,:],2)) +\
        torch.sum(torch.pow(feat1_[y:y+psize, x:x+psize,:]-feat2_[yp:yp+psize, xp:xp+psize,:],2))

def improve_guess(pos, pos_new_f, best_pos_f, best_dist, feat1, feat2, feat1_, feat2_, psize=2):
    """
    Return the best b position and corresponding distance
    Params:
        pos(2):position for update
        pos_f(2): new_position in b for update
        best_pos_f(2): best pos now
        best_dist(1): best distance now
        feat*(H*W*C): features
    """
    #print(best_dist)
    new_dist = feat_patch_distance(feat1, feat2, feat1_, feat2_, pos, pos_new_f, psize)
    if new_dist < best_dist:
        return pos_new_f, new_dist
    else:
        return best_pos_f, best_dist


def propagation(pos, change, f, dist_f, feat1, feat2, feat1_, feat2_, eff_shape, psize=2):
    """
    Batch Propagation in patch match.
    Params:
        pos(torch.Tensor:2): batch of position
        change(torch.Tensor:2): direction for propagation
        f(torch.Tensor:H*W*2): a \phi_a->b function represented by a tensor relative position
        dist_f(torch.Tensor:H*W): a \phi_a->b function represented by a tensor min dist
        feat*(torch.Tensor:C*H*W): batch features
    Return best_f(torch.Tensor:H*W*2) best_dist_f(torch.Tensor:H*W)
    """
    y,x = pos
    ew, eh = eff_shape
    best_pos_f = f[y, x]
    best_dist = dist_f[y, x]

    # make the change variable
    ychange, xchange = change
    # Batch pos adding up_change new 2
    if  abs(x - xchange) < ew and x - xchange >= 0:
        yp, xp = f[y, x-xchange]
        xp = xp + xchange
        if xp < ew and xp >= 0:
            best_pos_f, best_dist = improve_guess(pos, (yp, xp), best_pos_f, best_dist, feat1, feat2, feat1_, feat2_, psize)

    if abs(y - ychange) < eh and y-ychange >= 0:
        yp, xp = f[y-ychange, x]
        yp = yp + ychange
        if yp < eh and yp >=0:
            best_pos_f, best_dist = improve_guess(pos, (yp, xp), best_pos_f, best_dist, feat1, feat2, feat1_, feat2_, psize)

    #f, dist_f = update_f_dist_f(pos, f, dist_f, best_pos_f, best_dist)

    return best_pos_f, best_dist

def random_search(pos, f, dist_f, best_pos_f, best_dist, feat1, feat2, feat1_, feat2_,\
                    eff_shape, alpha=0.5, psize=2):
    """
    Batch Random Search For patch Match
    """
    r = [eff_shape[0], eff_shape[1]]
    eh, ew = eff_shape
    while r[0] >= 1 and r[1] >= 1:
        best_y, best_x = best_pos_f
        #end_time = time.time()
        xmin, xmax = max(best_x-r[1], 0), min(best_x+r[1]+1, ew)
        ymin, ymax = max(best_y-r[0], 0), min(best_y+r[0]+1, eh)
        pos_random_f = (ymin+np.random.randint(0, ymax-ymin), xmin+np.random.randint(0, xmax-xmin))#(torch.tensor([ymin, xmin]) + torch.rand(2)*torch.tensor([ymax-ymin, xmax-xmin])).type(torch.LongTensor)
        #print("Random Search: Random Pos Time:{}".format(time.time() - end_time))
        #end_time = time.time()
        best_pos_f, best_dist = improve_guess(pos, pos_random_f, best_pos_f, best_dist, feat1, feat2, feat1_, feat2_)
        #print("Random Search: Improve Time:{}".format(time.time() - end_time))

        r = [int(alpha*r[0]), int(alpha*r[1])]

    return best_pos_f, best_dist

def initialize_direction(i, ae_shape):
    if (i) % 2 == 1:
        change = [-1,-1]
        start = [ae_shape[0]-1, ae_shape[1]-1]
        end = [-1, -1]
    else:
        change = [1, 1]
        start = [0, 0]
        end = [ae_shape[0], ae_shape[1]]
    return change, start, end

def get_effective_shape(img_shape, psize):
    return (int(img_shape[0]-psize+1), int(img_shape[1]-psize+1))

def deep_patch_match(feat1, feat2, feat1_, feat2_, img_shape, psize=2, iteration=5, alpha=0.5):
    """
    A deep patch match method based on two pairs data. Formulated in Deep Image Analogy
    Original version only use img1 and img2
    Params: img1(torch.Tensor):  shape C*H*W
    """
    assert feat1.size() == feat2.size() == feat1_.size() == feat2_.size()
    eff_shape = get_effective_shape(img_shape, psize)
    #eff_shape = get_effective_shape(img_shape, psize)
    feat1, feat2, feat1_, feat2_ = feat1.to(device), feat2.to(device), feat1_.to(device), feat2_.to(device)
    # initialization
    f = torch.zeros(*((img_shape)+(2,)), device=device, dtype=torch.int32)
    dist_f = torch.zeros(*img_shape, device=device)
    for y in range(eff_shape[0]):
        for x in range(eff_shape[1]):
            pos = (y,x)
            pos_f = (np.random.randint(0,eff_shape[0]),np.random.randint(0,eff_shape[1]))
            f[y, x] = torch.tensor(pos_f, device=device).type(torch.LongTensor)
            dist_f[y, x] = feat_patch_distance(feat1, feat2, feat1_, feat2_, pos, pos_f, psize)

    for i in range(iteration):
        print("Iteration {}: Running".format(i+1))
        change, start, end = initialize_direction(i, eff_shape)
        print('start:{}, end:{}, change:{}'.format(start, end, change))
        ori_time = end_time = time.time()
        for y in range(int(start[0]), int(end[0]), int(change[0])):
            for x in range(int(start[1]), int(end[1]), int(change[1])):
                pos = (y,x)
                best_pos_f, best_dist = propagation(pos, change, f, dist_f, feat1, feat2, feat1_, feat2_, eff_shape, psize)
                best_pos_f, best_dist = random_search(pos, f, dist_f, best_pos_f, best_dist, feat1, feat2, feat1_, feat2_, eff_shape, psize=psize)
                f[y,x] = torch.tensor(best_pos_f, device=device, dtype=torch.int32)
                dist_f[y,x] = best_dist

        re_img1 = reconstruct_avg(feat2, f, psize=PSIZE)
        save_img(re_img1, "epoch_{}_re_test.png".format(i))
        print("Iteration {}: Finishing Time : {}".format(i+1, time.time()-ori_time))
    return f

def reconstruct_avg(feat2, f, psize=2):
    """
    Reconstruct another batch feat1 from batch feat2 by f
    Params:
        feat2(torch.Tensor:shape (C*H*W)): feature 2
        f(torch.Tensor:shape (H*W*2)): f : 1->2
    """
    #assert feat.size()[2:] == f.size(H)
    print(feat2.size())
    feat1 = torch.zeros_like(feat2)

    for y in range(feat2.size(0)):
        for x in range(feat2.size(1)):
            yp, xp = f[y,x]
            #print(yp,xp)
            batch_feat = feat2[yp:yp+psize,xp:xp+psize, :]
            feat1[y,x,:] = feat2[yp,xp,:] #batch_feat.reshape(psize*psize, feat2.size(2)).transpose(0,1).mean(dim=1)

    return feat1[:feat1.size(0)-psize, :feat1.size(1)-psize, :]

def img_padding(img, psize):
    """
    Input C*H*W
    """
    img = np.array(img)
    h, w, c = img.shape
    new_img = torch.zeros(h+psize, w+psize, c)
    new_img[:h, :w, :] = torch.tensor(img)
    return new_img/255

def save_img(img, name):
    #print(img)
    img = img.numpy()*255
    #print(img.shape)
    #print(value, sep=' ', end='n', file=sys.stdout, flush=False)
    img = Image.fromarray(img.astype(np.uint8))
    img.save(name)

def test():
    end_time = time.time()
    for i in range(1000):
        a = torch.tensor(np.random.rand(2))
    print("NP Random:{}".format(time.time()-end_time))
    end_time = time.time()
    for i in range(1000):
        torch.rand(2)
    print("Torch Random:{}".format(time.time()-end_time))

def main():
    transforms_fun = transforms.Compose([transforms.Resize(SHAPE)])
    img1 = transforms_fun(Image.open('bike_a.png'))
    img2 = transforms_fun(Image.open('bike_b.png'))
    img1_ = transforms_fun(Image.open('bike_a.png'))
    img2_ = transforms_fun(Image.open('bike_b.png'))
    #img1, img2, img1_, img2_ = reshape_test(img1),reshape_test(img2),reshape_test(img1_),reshape_test(img2_)
    img1, img2, img1_, img2_ = img_padding(img1, PSIZE),img_padding(img2, PSIZE),img_padding(img1_, PSIZE),img_padding(img2_, PSIZE)
    f = deep_patch_match(img1, img2, img1_, img2_, img_shape=(SHAPE[0]+PSIZE, SHAPE[1]+PSIZE), psize=PSIZE, iteration=10, alpha=0.5)
    re_img1 = reconstruct_avg(img2, f, psize=PSIZE)
    save_img(re_img1, "re_test.png")

if __name__ == '__main__':
    main()
