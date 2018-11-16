"""
The Patchmatch Algorithm. The actual algorithm is a nearly
line to line port of the original c++ version.
The distance calculation is different to leverage numpy's vectorized
operations.

This version uses 4 images instead of 2.
You can supply the same image twice to use patchmatch between 2 images.

"""
import os
package_directory = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as drv
import numpy
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
from pycuda.compiler import SourceModule
import cv2

from PIL import Image

class PatchMatch(object):
    def __init__(self, a, aa, b, bb, patch_size):
        """
        Initialize Patchmatch Object.
        This method also randomizes the nnf , which will eventually
        be optimized.
        """
        assert a.shape == b.shape == aa.shape == bb.shape, "Dimensions were unequal for patch-matching input"
        print("called")
        self.A = a.copy(order='C')
        self.B = b.copy(order='C')
        self.AA = aa.copy(order='C')
        self.BB = bb.copy(order='C')
        self.patch_size = patch_size
        self.nnf = np.zeros(shape=(self.A.shape[0], self.A.shape[1],2)).astype(np.int32)  # the nearest neighbour field
        self.nnd = np.random.rand(self.A.shape[0], self.A.shape[1]).astype(np.float32)   # the distance map for the nnf
        self.initialise_nnf()

    def initialise_nnf(self):
        """
        Set up a random NNF
        Then calculate the distances to fill up the NND
        :return:
        """
        self.nnf = self.nnf.transpose((2, 0, 1))
        self.nnf[0] = np.random.randint(self.B.shape[1], size=(self.A.shape[0], self.A.shape[1]))
        self.nnf[1] = np.random.randint(self.B.shape[0], size=(self.A.shape[0], self.A.shape[1]))
        self.nnf = self.nnf.transpose((1, 2, 0))
        self.nnf = self.nnf.copy("C")

    def reconstruct_image(self, img_a):
        """
        Reconstruct image using the NNF and img_a.
        :param img_a: the patches to reconstruct from
        :return: reconstructed image
        """
        final_img = np.zeros_like(img_a)
        size = self.nnf.shape[0]
        scale = img_a.shape[0] // self.nnf.shape[0]
        for i in range(size):
            for j in range(size):
                x, y = self.nnf[i, j]
                if final_img[scale * i:scale * (i + 1), scale * j:scale * (j + 1)].shape == img_a[scale * y:scale * (y + 1), scale * x:scale * (x + 1)].shape:
                    final_img[scale * i:scale * (i + 1), scale * j:scale * (j + 1)] = img_a[scale * y:scale * (y + 1), scale * x:scale * (x + 1)]
        return final_img

    def upsample_update(self, a, aa, b, bb):
        assert a.shape == b.shape == aa.shape == bb.shape, "Dimensions were unequal for patch-matching input"

        self.A = a.copy(order='C')
        self.B = b.copy(order='C')
        self.AA = aa.copy(order='C')
        self.BB = bb.copy(order='C')
        self.nnf = self.upsample_nnf(a.shape[:2]).copy("C")
        


    def upsample_nnf(self, shape):
        """
        Upsample NNF based on size. It uses nearest neighbour interpolation
        :param size: INT size to upsample to.

        :return: upsampled NNF
        """

        temp = np.zeros((self.nnf.shape[0], self.nnf.shape[1], 3))

        for y in range(self.nnf.shape[0]):
            for x in range(self.nnf.shape[1]):
                temp[y][x] = [self.nnf[y][x][0], self.nnf[y][x][1], 0]

        img = np.zeros(shape=(shape[0], shape[1], 2), dtype=np.int)
        small_size = self.nnf.shape
        aw_ratio = ((shape[1]) // small_size[1])
        ah_ratio = ((shape[0]) // small_size[0])
        print(aw_ratio, ah_ratio)
        temp = cv2.resize(temp, None, fx=ah_ratio, fy=aw_ratio, interpolation=cv2.INTER_NEAREST)

        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                pos = temp[i, j]
                img[i, j] = pos[0] * ah_ratio, pos[1] * aw_ratio

        return img


    def reconstruct_avg(self, img, patch_size=5):
        """
        Reconstruct image using average voting.
        :param img: the image to reconstruct from. Numpy array of dim H*W*3
        :param patch_size: the patch size to use

        :return: reconstructed image
        """

        final = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):

                dx0 = dy0 = patch_size // 2
                dx1 = dy1 = patch_size // 2 + 1
                dx0 = min(j, dx0)
                dx1 = min(img.shape[0] - j, dx1)
                dy0 = min(i, dy0)
                dy1 = min(img.shape[1] - i, dy1)

                patch = self.nnf[i - dy0:i + dy1, j - dx0:j + dx1]

                lookups = np.zeros(shape=(patch.shape[0], patch.shape[1], img.shape[2]), dtype=np.float32)

                for ay in range(patch.shape[0]):
                    for ax in range(patch.shape[1]):
                        x, y = patch[ay, ax]
                        #print(x, y)
                        lookups[ay, ax] = img[y, x]

                if lookups.size > 0:
                    value = np.mean(lookups, axis=(0, 1))
                    final[i, j] = value

        return final


    def visualize(self):
        """
        Get the NNF visualisation
        :return: The RGB Matrix of the NNF
        """
        nnf = self.nnf

        img = np.zeros((nnf.shape[0], nnf.shape[1], 3), dtype=np.uint8)

        for i in range(nnf.shape[0]):
            for j in range(nnf.shape[1]):
                pos = nnf[i, j]
                img[i, j, 0] = int(255 * (pos[0] / self.B.shape[1]))
                img[i, j, 2] = int(255 * (pos[1] / self.B.shape[0]))

        return img




    def propagate(self, iters=2, rand_search_radius=500):
        """
        Optimize the NNF using PatchMatch Algorithm
        :param iters: number of iterations
        :param rand_search_radius: max radius to use in random search
        :return:
        """
        mod = SourceModule(open(os.path.join(package_directory,"patchmatch.cu")).read(),no_extern_c=True)
        patchmatch = mod.get_function("patch_match")

        rows = self.A.shape[0]
        cols = self.A.shape[1]
        channels = np.int32(self.A.shape[2])
        nnf_t = np.zeros(shape=(rows,cols),dtype=np.uint32)
        threads = 20

        def get_blocks_for_dim(dim,blocks):
            #if dim % blocks ==0:
            #    return dim//blocks
            return dim// blocks +1
        patchmatch(
            drv.In(self.A),
            drv.In(self.AA),
            drv.In(self.B),
            drv.In(self.BB),
            drv.InOut(self.nnf),
            drv.InOut(nnf_t),
            drv.InOut(self.nnd),
            np.int32(rows),
            np.int32(cols),
            channels,
            np.int32(self.patch_size),
            np.int32(iters),
            np.int32(8),
            np.int32(rand_search_radius),
        block=(threads,threads,1),
        grid=(get_blocks_for_dim(rows,threads),
              get_blocks_for_dim(cols,threads)))
