"""
Date: 2024,3,28
"""
from scipy import ndimage as ndi
import numpy as np
import skimage
from skimage import io, filters
import matplotlib.pyplot as plt

def show_3_im(img1, img2, img3, name=None):
    fig, (ax, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 12))
    ax.imshow(img1)
    ax1.imshow(img2)
    ax2.imshow(img3)
    if name: plt.savefig(r'.\quant_fig\{}'.format(name))
    plt.show()

def show_ims(*args):
    num = len(args)
    fig, axs = plt.subplots(1, num)
    if num == 1:
        axs.imshow(args[0])
    else:
        for i, ax in enumerate(axs):
            ax.imshow(args[i])
    plt.show()

def thresh_finder(x, y, extra_bias=0):
    """
    Aim to use this function to find the thresh where the hist gram just leave the plateau.
    """
    max_index = np.argmax(y)
    differ = [y[index] - y[index+1] for index in range(max_index, len(y)-1)]
    bias = np.argmax(differ)
    return x[np.argmax(y)+bias+extra_bias]


def remove_large_objects(im, max_size=1000, connectivity=1):
    out = im.copy()
    selem = ndi.generate_binary_structure(im.ndim, connectivity)
    ccs = np.zeros_like(im, dtype=np.int32)
    ndi.label(im, selem, output=ccs)
    component_sizes = np.bincount(ccs.ravel())
    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0
    return out

def show(img, **kwargs):
    if 'name' in kwargs: print(kwargs['name'])
    skimage.io.imshow(img)
    skimage.io.show()


def brightfield_segmentation(im_bf,
                             gauss_sigma=30,
                             truncate=0.35,
                             dark_thresh=20000,
                             light_thresh=1000,
                             disk_radius=2,
                             name=None):
    """
    im_bf,process the brightfiled image
    """
    im_bg = filters.gaussian(im_bf, sigma=5)
    im = im_bf-im_bg
    im = im > 4 / 255
    total_area = sum(sum(im))
    return im, total_area

def bright_segmentation(im_bf):
    """
    process the green_fluorescence images
    :param im_bf:
    :return:
    """
    im_bg = filters.gaussian(im_bf, sigma=30)
    im_no_bg = im_bf - im_bg
    hist_bin = skimage.exposure.histogram(im_no_bg)
    hist, bins = hist_bin
    thresh = thresh_finder(bins, hist, extra_bias=6)
    res = im_no_bg > thresh
    total_area = sum(sum(res))
    return res, total_area


def signal_segmentation(im_sig,
                        gauss_sigma=5,
                        sig_thresh=100,
                        min_size=5,
                        name=None):
    im_bg = skimage.filters.gaussian(im_sig, sigma=gauss_sigma)
    im_no_bg = im_sig - im_bg
    hist_bin = skimage.exposure.histogram(im_no_bg)
    hist, bins = hist_bin
    thresh = thresh_finder(bins, hist, extra_bias=12)
    # Determine the dark areas
    sig_thresh = im_no_bg > thresh
    total_area = sum(sum(sig_thresh))
    return sig_thresh, total_area


def in_vitro_quantification(im_bf,
                            im_sig,
                            bf_gauss_sigma=30,
                            truncate=0.35,
                            dark_thresh=10000,
                            light_thresh=3000,
                            disk_radius=2,
                            sig_gauss_sigma=5,
                            sig_thresh=1000,
                            min_size=5,
                            photo_name=None):

    # green fluorescence images
    brightfield_areas, total_area = brightfield_segmentation(im_bf, bf_gauss_sigma, truncate,
                                                             dark_thresh, light_thresh, disk_radius, name=photo_name)
    # bright filed images
    # brightfield_areas, total_area = bright_segmentation(im_bf)
    signal_areas, signal_total_area = signal_segmentation(im_sig, sig_gauss_sigma,
                                                          sig_thresh, min_size, name=photo_name)
    show_3_im(brightfield_areas, im_sig, signal_areas, name=photo_name)
    original_sig = signal_areas*im_sig
    total_brightness = np.sum(np.sum(original_sig))

    return total_area, signal_total_area, total_brightness
