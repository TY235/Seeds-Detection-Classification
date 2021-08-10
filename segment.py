import cv2
from matplotlib import pyplot as plt
import time
from multiprocessing import Pool

#PARAMETERS AND CONSTANTS
DEBUG_FLAG = 0
DARK_THRESHOLD = 80 #threshold value in grayscale for "dark" pixel
MASK_SIZE = 21 #must be odd number
IN_SEED_THRESHOLD = 100 #number of dark pixels to indicate entering seed region
OUT_SEED_THRESHOLD = 15 #number of dark pixels to indicate exiting seed region

#calculate number of dark pixels in each column/row
#mode=1 for horizontal, 2 for vertical
def calc_dark_pixel(img, mode):
    if mode == 1:
        size = img.shape[1]
    else:
        size = img.shape[0]
    dark_pixel = [0] * size

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] < DARK_THRESHOLD:
                if mode == 1:
                    dark_pixel[x] += 1
                else:
                    dark_pixel[y] += 1

    #debug
    if DEBUG_FLAG == 1:
        print("dark_pixel", dark_pixel)
        
    return dark_pixel

def apply_mask(arr, mask_size):
    arr_masked = [0] * len(arr)

    #first mask application
    for col in range(mask_size+1):
        arr_masked[int(mask_size/2)+1] += arr[col]
    #calculate subsequent column masked values O(N)
    for col in range(int(mask_size/2)+2, len(arr) - int(mask_size/2)):
        arr_masked[col] = arr_masked[col-1] - arr[col-int(mask_size/2)+1] + arr[col+int(mask_size/2)]
    #average out masked values
    for col in range(len(arr)):
        arr_masked[col] = int(arr_masked[col]/mask_size)

    #debug
    if DEBUG_FLAG == 1:
        print("arr_masked", arr_masked)
        
    return arr_masked

#find the edges of seeds from dark_pixel
def find_seed_edges(dark_pixel):
    seed_edges = []
    in_seed = False
    for i in range(len(dark_pixel)):
        if in_seed and dark_pixel[i] < OUT_SEED_THRESHOLD:
            seed_edges.append(i)
            in_seed = False
        elif not in_seed and dark_pixel[i] > IN_SEED_THRESHOLD:
            seed_edges.append(i)
            in_seed = True

    #debug
    if DEBUG_FLAG == 1:
        print("seed_edges", seed_edges)

    return seed_edges

#find cutting points
def find_cutting_points(seed_edges):
    cutting_points = []

    #find midpoints of spaces between seeds
    i = 1
    while i < len(seed_edges) - 1:
        left_seed_right = seed_edges[i]
        right_seed_left = seed_edges[i+1]
        cutting_points.append(int((left_seed_right + right_seed_left) / 2))
        i += 2

    #find starting cut point for first seed and ending cut point for last seed
    first_seed_left = seed_edges[0] - (cutting_points[0] - seed_edges[1])
    last_seed_right = seed_edges[-1] + (seed_edges[-2] - cutting_points[-1])
    cutting_points = [first_seed_left] + cutting_points + [last_seed_right]

    #debug
    if DEBUG_FLAG == 1:
        print("cutting_points", cutting_points)

    return cutting_points

#cut seeds horizontally according to cutting points
def seed_separation_horizontal(img, cutting_points):
    seed_imgs_notrim = []
    i = 0
    for i in range(len(cutting_points)-1):
        seed_left = cutting_points[i]
        seed_right = cutting_points[i+1]
        seed_img = img[:, seed_left:seed_right, :]

        seed_imgs_notrim.append(seed_img)

    return seed_imgs_notrim

#trim top and bottom of the seed image
def trim_seed_vertical(img):
    #use blue channel only for grayscale
    img_gs = img[:, :, 0]

    dark_pixel_vertical = calc_dark_pixel(img_gs, 2)
    dark_pixel_vertical_masked = apply_mask(dark_pixel_vertical, MASK_SIZE)
    seed_edges_vertical = find_seed_edges(dark_pixel_vertical_masked)

    seed_img_width = img.shape[1]
    seed_midpoint = (seed_edges_vertical[0] + seed_edges_vertical[1]) / 2
    seed_top = int(seed_midpoint - seed_img_width/2)
    seed_bottom = int(seed_midpoint + seed_img_width/2)
    seed_img_trimmed = img[seed_top:seed_bottom, :, :]

    return seed_img_trimmed

def segment_whole_img(img, seed_num, writepath):

    #crop out top and bottom 15% of the whole image
    img_height = img.shape[0]
    img = img[int(img_height * 0.15):img_height-int(img_height * 0.15), :, :]

    #use blue channel only for grayscale
    img_gs = img[:, :, 0]

    #separate each seed horizontally
    dark_pixel_horizontal = calc_dark_pixel(img_gs, 1) 
    dark_pixel_horizontal_masked = apply_mask(dark_pixel_horizontal, MASK_SIZE)
    seed_edges_horizontal = find_seed_edges(dark_pixel_horizontal_masked)
    cutting_points_horizontal = find_cutting_points(seed_edges_horizontal)
    seed_imgs_notrim = seed_separation_horizontal(img, cutting_points_horizontal)

    #trim each separated seed image
    seed_imgs = []
    for seed in seed_imgs_notrim:
        seed_imgs.append(trim_seed_vertical(seed))

    return seed_imgs
    
#create training and testing sets of individual seed images
#
#group numbers:
#1 - GoodSeed testing set
#2 - GoodSeed training set
#3 - BadSeed testing set
#3 - BadSeed training set
def segment(group):
    seed_num = 0
    if group == 1:
        num_start = 0
        num_end = 19
        readpath = "seed/GoodSeed"
        filename = "GoodSeed{0}.jpg"
        writepath = "seedsegment/GoodSeed/test"
    elif group == 2:
        num_start = 20
        num_end = 109
        readpath = "seed/GoodSeed"
        filename = "GoodSeed{0}.jpg"
        writepath = "seedsegment/GoodSeed/train"
    elif group == 3:
        num_start = 0
        num_end = 19
        readpath = "seed/BadSeed"
        filename = "BadSeed{0}.jpg"
        writepath = "seedsegment/BadSeed/test"
    elif group == 4:
        num_start = 20
        num_end = 104
        readpath = "seed/BadSeed"
        filename = "BadSeed{0}.jpg"
        writepath = "seedsegment/BadSeed/train"
        
    for i in range(num_start, num_end+1):
        start_time = time.time()
        print("Segmenting " + filename.format(i))
        img = cv2.imread(readpath + '/' + filename.format(i), 1)
        
        seed_imgs = segment_whole_img(img, seed_num, writepath)
        
        for seed_img in seed_imgs:
            cv2.imwrite(writepath + '/seed{0}.jpg'.format(seed_num), seed_img)
            seed_num += 1
            
        print("Time taken for " + filename.format(i) + ": {0:2f}s".format(time.time() - start_time))

if __name__ == '__main__':
    with Pool(4) as p:
        print(p.map(segment, list(range(1, 5))))
    print("Press enter to close the program")
    input()
