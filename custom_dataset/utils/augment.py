'''Original code author: Li Hua'''
import numpy as np
import cv2 as cv
import random
import torch
from PIL import Image
from matplotlib import cm

# Constant
MAX_VALUE = 255
GAUSSIAN_SIGMA_THRES = 4

ALL_COLOR_MAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
				  'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                  'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                  'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                  'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                  'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                  'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
                  'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                  'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                  'twilight', 'twilight_shifted', 'hsv',
                  'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                  'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b','tab20c',
                  'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                  'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                  'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                  'turbo', 'nipy_spectral', 'gist_ncar']

COLOR_TEXTURE_PARAMS = {
    'bg': {'p_black': 0.2, 'p_single': 0.4, 'p_single_grad': 0.1, 'p_multi_grad': 0.1, 'p_colormap': 0.2, 'p_n': [0.8, 0.2]},
    'color': {'p_black': 0.1, 'p_single': 0.2, 'p_single_grad': 0.2, 'p_multi_grad': 0.2, 'p_colormap': 0.3, 'p_n': [0.8, 0.2]},
    'border': {
        'p_border': 0.8,
        'min_width': 0.01, 'max_width': 0.08, 
        'min_gaussian': 0.0, 'max_gaussian': 0.1,
        'p_n': [0.8, 0.1, 0.1],
        'color_single': {'p_black': 0.0, 'p_single': 0.6, 'p_single_grad': 0.2, 'p_multi_grad': 0.1, 'p_colormap': 0.1, 'p_n': [0.8, 0.2]},
        'color_multi': {'p_black': 0.0, 'p_single': 0.8, 'p_single_grad': 0.2, 'p_multi_grad': 0.0, 'p_colormap': 0.0, 'p_n': [0.8, 0.2]},
    },
    'shift': {
        'p_shift': 0.8,
        'min_distance': 0.01, 'max_distance': 0.08, 
        'min_gaussian': 0.0, 'max_gaussian': 0.1,
        'p_n': [0.8, 0.1, 0.1],
        'color_single': {'p_black': 0.3, 'p_single': 0.3, 'p_single_grad': 0.2, 'p_multi_grad': 0.1, 'p_colormap': 0.1, 'p_n': [0.8, 0.2]},
        'color_multi': {'p_black': 0.3, 'p_single': 0.5, 'p_single_grad': 0.2, 'p_multi_grad': 0.0, 'p_colormap': 0., 'p_n': [0.8, 0.2]},
    }
}

############################################################
##########          helper functions              ##########
############################################################

def get_random_color(n=1, max_value=MAX_VALUE):
    return np.round(np.random.rand(n,3)*max_value)

def get_random_sim_color(ref_color, max_value=MAX_VALUE):
    if random.random() < ref_color.mean()/max_value: # darker
        return ref_color*random.random()
    else: # lighter
        w = random.random()
        return w*ref_color + (1-w)*np.ones(3,)*max_value

'''
img: np array (size, size, 3), [0, 255]
color: [b, g, r]
apply color to gray scale image
'''
def apply_color_img(img, color_img):
    res = np.zeros_like(img)
    res[..., :3] = color_img
    res[..., 3] = img[..., 3]
    return res

def apply_color(img, color):
	color = color.reshape(-1,)
	color_img = np.stack((np.ones(img.shape[:2])*color[0], np.ones(img.shape[:2])*color[1], np.ones(img.shape[:2])*color[2]),axis=2)
	return apply_color_img(img, color_img)

'''
color: (b, g, r)
'''
def create_horizontal_gradient(h, w, start_color, end_color):
	gradient = np.zeros((h,w,3), np.uint8)
	gradient[:,:,:] = np.linspace(start_color, end_color, w, dtype=np.uint8)
	return gradient
'''
colors: (n_color, 3)
'''
def create_horizontal_gradient_multi(h, w, colors):
	gradient = np.zeros((h,w,3), np.uint8)
	n_color = colors.shape[0]
	ws = [w // (n_color-1)] * (n_color-1)
	ws[-1] += w - sum(ws)
	ws_cumsum = [0] + np.cumsum(ws).tolist()
	for i in range(n_color-1):
		gradient[:,ws_cumsum[i]:ws_cumsum[i+1],:] = np.linspace(colors[i], colors[i+1], ws[i], dtype=np.uint8)
	return gradient
'''
'''
def create_horizontal_colormap(h, w, colormap='rainbow', max_value=MAX_VALUE):
	colors = eval('cm.'+colormap)(np.arange(w)/w)
	gradient = np.zeros((h,w,3), np.uint8)
	gradient[:,:,:] = colors[:,:3]*max_value
	return gradient
'''
only for square
rot in degree
'''
def create_angled(h, func, PARAMS, rot=0):
	gradient = func(h*2, h*2, PARAMS)
	gradient = np.array(Image.fromarray(gradient).rotate(rot, resample=Image.Resampling.BILINEAR, expand=True))
	a, b = np.abs(np.sin(rot/180*np.pi))*h*2, np.abs(np.cos(rot/180*np.pi))*h*2
	start = int(np.ceil(a*b/(a+b)))
	gradient = gradient[start:-start, start:-start]
	gradient = cv.resize(gradient, (h, h))
	return gradient

'''
img: np array (size, size, 3), [0, 255]
color: [b, g, r]
assume img black front & white background, expand black border into color
'''
def create_bordered_img(img, color, width=1, gaussian=None):
	kernel = np.ones((int(2*width+1), int(2*width+1)), dtype=np.uint8)
	img_ = cv.erode(src=img, kernel=kernel)#, iterations=1)
	img_ = apply_color(img_, color)
	if gaussian and gaussian > 0:
		img_ = cv.GaussianBlur(img_, (gaussian*2+1, gaussian*2+1), 0)
	return img_
'''
colors: (n_color, 3) from inside to outside
widths: (n_color, ) from small to large
'''
def create_bordered_img_multi(img, colors, widths, gaussian=None):
	img_ = create_bordered_img(img, color=colors[-1], width=widths[-1])
	for color, width in zip(colors[::-1][1:], widths[::-1][1:]):
		img__ = create_bordered_img(img, color=color, width=width)
		img_ = overlay(img_, img__)
	if gaussian and gaussian > 0:
		img_ = cv.GaussianBlur(img_, (gaussian*2+1, gaussian*2+1), 0)
	return img_

'''
img: np array (size, size, 3), [0, 255]
color: [b, g, r]
assume white background
'''
def create_shifted_img(img, color, distance=None, degree=45, bgcolor=[MAX_VALUE,MAX_VALUE,MAX_VALUE,0], gaussian=None):
	if not distance:
		distance = int(img.shape[0]*0.02)
	a, b = np.sin(degree/180*np.pi)*distance, np.cos(degree/180*np.pi)*distance
	img_ = Image.fromarray(img).rotate(0, translate=(int(np.round(a)),int(np.round(b))), fillcolor=tuple(bgcolor))
	img_ = apply_color(np.array(img_), color)
	if gaussian and gaussian > 0:
		img_ = cv.GaussianBlur(img_, (gaussian*2+1, gaussian*2+1), 0)
	return img_
'''
colors: (n_color, 3) from near to distant
distances: (n_color, ) from small to large
'''
def create_shifted_img_multi(img, colors, distances=None, degree=45, bgcolor=[MAX_VALUE,MAX_VALUE,MAX_VALUE,0], gaussian=None):
	if not distances:
		distances = [int(img.shape[0]*0.02), int(img.shape[0]*0.04), int(img.shape[0]*0.06)]
	img_ = create_shifted_img(img, color=colors[-1], distance=distances[-1], degree=degree, bgcolor=bgcolor)
	for color, distance in zip(colors[::-1][1:], distances[::-1][1:]):
		img__ = create_shifted_img(img, color=color, distance=distance)
		img_ = overlay(img_, img__)
	if gaussian and gaussian > 0:
		img_ = cv.GaussianBlur(img_, (gaussian*2+1, gaussian*2+1), 0)
	return img_

############################################################
##########          end helper functions          ##########
############################################################

'''
ref_color: (b, g, r)
'''
def get_colored_img(img, p_black, p_single, p_single_grad, p_multi_grad, p_colormap=None, p_n=[0.8, 0.2]):

    p = random.random()
    h, w, _ = img.shape
    single = True

    if p < p_black:
        return img, True # return original fontimg

    elif p < p_black + p_single:
        rndcolor = get_random_color(n=1)
        color = np.tile(rndcolor, (h,w,1)).astype(np.uint8)
    
    elif p < p_black + p_single + p_single_grad:
        rndcolor = get_random_color(n=1)
        colors = np.stack((rndcolor, get_random_sim_color(rndcolor)), axis=0)
        gradient = create_angled(h, create_horizontal_gradient_multi, (colors), rot=random.random()*360)
        color, single = gradient, False

    elif p < p_black + p_single + p_single_grad + p_multi_grad:
        colors = get_random_color(n=random.choices(range(2, len(p_n)+2),weights=p_n,k=1)[0])
        gradient = create_angled(h, create_horizontal_gradient_multi, (colors), rot=random.random()*360)
        color, single = gradient, False

    else:
        color_map = random.sample(ALL_COLOR_MAPS, 1)[0]
        gradient = create_angled(h, create_horizontal_colormap, (color_map), rot=random.random()*360)
        color, single = gradient, False

    return apply_color_img(img, color), single

'''
single: if get_colored_img result is single-colored (or single grad)
'''
def get_bordered_img(img, single, p_border, min_width, max_width, min_gaussian, max_gaussian, 
                        color_single, color_multi, p_n=[0.8, 0.1, 0.1]):
    p = random.random()
    h, _, _ = img.shape
    if p > p_border:
        return np.zeros_like(img)

    min_width, max_width = int(np.round(h*min_width)), int(np.round(h*max_width))
    min_gaussian, max_gaussian = int(np.round(h*min_gaussian)), int(np.round(h*max_gaussian))
    n_border = random.choices(range(1, len(p_n)+1), weights=p_n, k=1)[0]
    widths = sorted(np.abs(np.random.randn(n_border))/GAUSSIAN_SIGMA_THRES * (max_width-min_width) + min_width)
    gaussian = np.abs(np.random.randn(1))/GAUSSIAN_SIGMA_THRES * (max_gaussian-min_gaussian) + min_gaussian
    border_ref = create_bordered_img_multi(img, np.array([[0,0,0]]), widths, int(np.round(gaussian)))
    if single:
        border, _ = get_colored_img(border_ref, **color_single)
    else:
        border, _ = get_colored_img(border_ref, **color_multi)
    return border

'''
single: if get_colored_img result is single-colored (or single grad)
'''
def get_shifted_img(img, single, p_shift, min_distance, max_distance, min_gaussian, max_gaussian, 
                        color_single, color_multi, p_n=[0.8, 0.1, 0.1]):
    p = random.random()
    h, _, _ = img.shape
    if p > p_shift:
        return np.zeros_like(img)

    min_distance, max_distance = int(np.round(h*min_distance)), int(np.round(h*max_distance))
    min_gaussian, max_gaussian = int(np.round(h*min_gaussian)), int(np.round(h*max_gaussian))
    n_shift = random.choices(range(1, len(p_n)+1), weights=p_n, k=1)[0]
    distances = sorted(np.abs(np.random.randn(n_shift))/GAUSSIAN_SIGMA_THRES * (max_distance-min_distance) + min_distance)
    gaussian = np.abs(np.random.randn(1))/GAUSSIAN_SIGMA_THRES * (max_gaussian-min_gaussian) + min_gaussian
    shift_ref = create_shifted_img_multi(img, np.array([[0,0,0]]), distances, degree=random.random()*360, gaussian=int(np.round(gaussian)))
    if single:
        shift, _ = get_colored_img(shift_ref, **color_single)
    else:
        shift, _ = get_colored_img(shift_ref, **color_multi)
    return shift

'''
lay img2 on top of img1, filter out white color in img2 based on ref
img1, img2, ref: np array (size, size, 3), [0, 255]
'''
def overlay(imgarr1, imgarr2):
    img1 = Image.fromarray(imgarr1)
    img2 = Image.fromarray(imgarr2)
    res = Image.alpha_composite(img1, img2)
    res = np.array(res)
    return res

def aug_single_ch(img, seed, aug_bg=False, color_texture_params=COLOR_TEXTURE_PARAMS):
    # fix seed
    random.seed(seed)
    np.random.seed(seed)

    # render layers
    imarr = np.array(img)
    color, single = get_colored_img(imarr, **color_texture_params['color'])
    border = get_bordered_img(imarr, single, **color_texture_params['border'])
    shift = get_shifted_img(imarr, single, **color_texture_params['shift'])
    
    # merge layers
    if aug_bg:
        background = get_colored_img(np.ones_like(imarr)*MAX_VALUE, **color_texture_params['bg'])[0]
    else:
        background = np.zeros_like(imarr)
    res = overlay(background, shift)
    res = overlay(res, border)
    res = overlay(res, color)
    res = Image.fromarray(res)
	
    # fill white bg
    res_w = Image.new(mode='RGBA', size=res.size, color=(MAX_VALUE, MAX_VALUE, MAX_VALUE, MAX_VALUE))
    res_w.paste(res, (0, 0), res)
    res_w = res_w.convert("RGB")

    # get alpha mask
    _, _, _, alpha = res.split()
    a_mask = np.array(alpha, dtype=np.float32) / MAX_VALUE
    a_mask = torch.from_numpy(a_mask).unsqueeze(0)

    return res, res_w, a_mask

def aug_chs(chs, aug_seed, aug_bg=False):
    
    augs = []
    aug_ws = []
    a_masks = []
    for ch in chs:
        aug, aug_w, a_mask = aug_single_ch(ch, aug_seed, aug_bg)
        augs.append(aug)
        aug_ws.append(aug_w)
        a_masks.append(a_mask)

    a_mask = torch.stack(a_masks, dim=0)

    return augs, aug_ws, a_mask

def disturb_single_ch(ch, dist_seed, p_dist=0.5):
    # fix seed
    random.seed(dist_seed)
    np.random.seed(dist_seed)

    arr = np.array(ch)
    if random.random() < p_dist: # gaussian blur
        ksize = random.choice([3, 5, 7, 11])
        arr = cv.GaussianBlur(arr, (ksize, ksize), 10.0)
    if random.random() < p_dist: # salt and pepper noise
        noise = np.random.normal(0, 20, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    if random.random() < p_dist: # downsample
        scale = random.uniform(0.05, 0.5)
        h, w = arr.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        arr_small = cv.resize(arr, new_size, interpolation=cv.INTER_LINEAR)
        arr = cv.resize(arr_small, (w, h), interpolation=cv.INTER_LINEAR)
    if random.random() < p_dist: # rotation
        angle = random.uniform(-45, 45)
        h, w = arr.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        arr = cv.warpAffine(arr, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)

    return Image.fromarray(arr)

def disturb_chs(chs, dist_seed):

    dists = []
    for ch in chs:
        dist = disturb_single_ch(ch, dist_seed)
        dists.append(dist)

    return dists