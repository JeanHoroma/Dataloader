import os, sys
import ReadDatFile
import config_LandUseSampling_inria as cfg
# import cv2 as cv
import numpy as np
from PIL import Image
import random
from shutil import copyfile
from multiprocessing import Pool
import collections
import time
import glob
import cv2
import shutil
import re


# this program will create take image lists as INPUT found in config_LandUseSamplilng
# it will generate small images 1024 x 1024 for the CNN to use (for instance, random crop)
# it could generate 256x256, but it has no Data Augmentation capabilities.
Image.MAX_IMAGE_PIXELS = None  # remove max image size limit
file_types = (".DAT", ".tif", ".tiff")

# list of folders to be generated if they are missing
folders = ['train_frames/img', 'train_masks/img',
           'val_frames/img', 'val_masks/img',
           'test_frames/img', 'test_masks/img',
           'frames', 'masks',   'generated/masks', 'generated/frames',
            'generated/output/masks', 'generated/output/frames']

# creates missing folder if necessary
for folder in folders:
    os.makedirs(cfg.PATH_OUTPUT + folder, exist_ok=True)

# INPUT of all images, unprocessed, RGB, Raster -mask-, DSM, OSM in .TIF format
# generates the list of names available in each directory
n = os.listdir(cfg.PATH_INPUT_RGB)  # List of testing images
m = os.listdir(cfg.PATH_INPUT_R)  # List of testing mask
#o = os.listdir(cfg.PATH_INPUT_DSM)  # List of testing DSM
#p = os.listdir(cfg.PATH_INPUT_OSM)  # list open street map files
# validates if there are missing files

#if len(n) != len(p):
#    a = len(n)
#    print('image files', a)
#    #b = len(o)
#    #print('dsm files', b)
#    c = len(m)
#    print('ground truth files', c)
#    #d = len(p)
#    #print('ground truth files', d)
#    print(
#        'Number of files does not match:  {} RGB files, {} dsm files,  {} Labelled files, {} open street map files'.format(
#            a, b, c, d))
    #sys.exit()  # stop program

# sort all list in order...
n.sort()
m.sort()
#o.sort()
#p.sort()
# cue for the number of files
print('image', len(n))
print('mask', len(m))
#print('dsm', len(o))
#print('osm', len(p))
small_number = 0
small_number_r = 0
small_number_d = 0
small_number_o = 0

# generates ALL small images (with overlap), see config file for details
for i in range(len(n)):
    # for i in range(2):
    image = Image.open(cfg.PATH_INPUT_RGB + '/' + n[i])

    gt = Image.open(cfg.PATH_INPUT_R + '/' + m[i])

   # dsm = Image.open(cfg.PATH_INPUT_DSM + '/' + o[i])

    #osm = Image.open(cfg.PATH_INPUT_OSM + '/' + p[i])

    for (x, y, resultat) in ReadDatFile.sliding_window(np.asarray(image), cfg.STEP, (cfg.winW, cfg.winH)):
        if resultat.shape[0] != cfg.winH or resultat.shape[1] != cfg.winW:
            continue

        img_save = Image.fromarray(resultat)
        location = cfg.PATH_OUTPUT_IMAGE + 'ID_' + str(small_number) + '.tif'

        img_save.save(location)
        small_number += 1

    for (x, y, resultat_r) in ReadDatFile.sliding_windowBW(np.asarray(gt), cfg.STEP, (cfg.winW, cfg.winH)):
        if resultat_r.shape[0] != cfg.winH or resultat_r.shape[1] != cfg.winW:
            continue

        img_save = Image.fromarray(np.array(resultat_r/255., dtype=np.uint8))
        location1 = cfg.PATH_OUTPUT_MASK + 'ID_' + str(small_number_r) + '.tif'

        img_save.save(location1)
        small_number_r += 1

    #for (x, y, resultat_d) in ReadDatFile.sliding_windowBW(np.asarray(dsm), cfg.STEP, (cfg.winW, cfg.winH)):
    #    if resultat_d.shape[0] != cfg.winH or resultat_d.shape[1] != cfg.winW:
    #        continue
    #    # save files in tiff 8 bit format for Tensor easy loading
    #    # normalize image
    #    min_val = np.amin(resultat_d)
    #    #temp = np.asarray(resultat_d - min_val),
    #    temp = np.asarray(resultat_d - min_val, dtype=np.uint8).copy()
    #    img_save1 = Image.fromarray(temp)
    #    #location2 = cfg.PATH_OUTPUT_DSM + 'ID_' + str(small_number_d)
    #    location2 = cfg.PATH_OUTPUT_DSM + 'ID_' + str(small_number_d) + '.tif'
    #    img_save1.save(location2)
    #    #np.save(location2, resultat_d)
    #    small_number_d += 1
    #
    #for (x, y, resultat_o) in ReadDatFile.sliding_windowBW(np.asarray(osm), cfg.STEP, (cfg.winW, cfg.winH)):
    #    if resultat_o.shape[0] != cfg.winH or resultat_o.shape[1] != cfg.winW:
    #        continue
    #    img_save1 = Image.fromarray(resultat_o)
    #    location3 = cfg.PATH_OUTPUT_OSM + 'ID_' + str(small_number_o) + '.tif'
    #
    #    img_save1.save(location3)
    #    small_number_o += 1

all_frames = os.listdir(cfg.PATH_OUTPUT_IMAGE)
all_masks = os.listdir(cfg.PATH_OUTPUT_MASK)
#all_dsm = os.listdir(cfg.PATH_OUTPUT_DSM)
#all_osm = os.listdir(cfg.PATH_OUTPUT_OSM)

all_frames.sort()
all_masks.sort()
#all_dsm.sort()
#all_osm.sort()

# shuffle files with same seed
random.seed(1234)
random.shuffle(all_frames)
random.seed(1234)
random.shuffle(all_masks)
#random.seed(1234)
#random.shuffle(all_dsm)
#random.seed(1234)
#random.shuffle(all_osm)

train_split = int(0.8 * len(all_frames))
val_split = int(0.9 * len(all_frames))

# frames
train_frames = all_frames[:train_split]
val_frames = all_frames[train_split:val_split]
test_frames = all_frames[val_split:]
# Ground truth masks
train_masks = all_masks[:train_split]
val_masks = all_masks[train_split:val_split]
test_masks = all_masks[val_split:]
# DSM images
#rain_dsm = all_dsm[:train_split]
#al_dsm = all_dsm[train_split:val_split]
#est_dsm = all_dsm[val_split:]
# OSM images
#rain_osm = all_osm[:train_split]
#al_osm = all_osm[train_split:val_split]
#est_osm = all_osm[val_split:]
#

def add_frames(dir_name, image):
    # print(image)
    # print(dir_name)
    copyfile(cfg.PATH_OUTPUT_IMAGE + image, cfg.PATH_OUTPUT + '{}'.format(dir_name) + '/' + image)


def add_masks(dir_name, image):
    copyfile(cfg.PATH_OUTPUT_MASK + image, cfg.PATH_OUTPUT + '{}'.format(dir_name) + '/' + image)


#def add_dsm(dir_name, image):
#    copyfile(cfg.PATH_OUTPUT_DSM + image, cfg.PATH_OUTPUT + '{}'.format(dir_name) + '/' + image)
#
#
#def add_osm(dir_name, image):
#    copyfile(cfg.PATH_OUTPUT_OSM + image, cfg.PATH_OUTPUT + '{}'.format(dir_name) + '/' + image)
#

frame_folders = [(train_frames, 'train_frames/img'), (val_frames, 'val_frames/img'),
                 (test_frames, 'test_frames/img')]

mask_folders = [(train_masks, 'train_masks/img'), (val_masks, 'val_masks/img'),
                (test_masks, 'test_masks/img')]

#dsm_folders = [(train_masks, 'train_dsm/img'), (val_masks, 'val_dsm/img'),
#               (test_masks, 'test_dsm/img')]
#
#osm_folders = [(train_masks, 'train_osm/img'), (val_masks, 'val_osm/img'),
#               (test_masks, 'test_osm/img')]
#
# add frames
for folder in frame_folders:
    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_frames, name, array))

# Add masks
for folder in mask_folders:
    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_masks, name, array))

# Add osm
#for folder in osm_folders:
#    array = folder[0]
#    name = [folder[1]] * len(array)
#
#    list(map(add_osm, name, array))
#
# Add dsm
#for folder in dsm_folders:
#    #image1 = []
#    #for i in range(len(folder[0])):
#    #    image1.append(re.sub(r'tif', r'npy', folder[0][i]))
#
#    array = folder[0]
#    #array = image1
#    name = [folder[1]] * len(array)
#
#    list(map(add_dsm, name, array))
#
# ####################################### IMGAE AUGMENTATION #########################################

#
#TH_INPUT_RGB = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/frames'
#ground truth mask
#TH_INPUT_R = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/masks'
#DSM
#ATH_INPUT_DSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/dsm'
# OSM
#ATH_INPUT_OSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/osm'
#
#TH_OUTPUT = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/'
#TH_OUTPUT_MASK = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/masks/'
#TH_OUTPUT_IMAGE = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/frames/'
#ATH_OUTPUT_DSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/dsm/'
#ATH_OUTPUT_OSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/osm/'
#
#= os.listdir(PATH_INPUT_RGB)  # List of testing images
#= os.listdir(PATH_INPUT_R)  # List of testing mask
# = os.listdir(PATH_INPUT_DSM)  # List of testing DSM
# = os.listdir(PATH_INPUT_OSM)  # List of testing OSM
#
#r_classe = 18
#output_list_x3 = []
#output_list_x6 = []
#start numbering images value
#ount = 11520
#me.sleep(2)
#unt = len(n)
#int('count',count)
#
#f rot(list_frames, list_mask, list_dsm, list_osm, count):
#  print('saving augmented frames')
#  for g in range(len(list_frames)):
#      frames = Image.open(list_frames[g])
#      masks = Image.open(list_mask[g])
#      dsm = Image.open(list_dsm[g])
#      osm = Image.open(list_osm[g])
#      #print('augmented image count val', count)
#      frames.rotate(90).save(
#          '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/frames/' + 'ID_' + str(
#              count) + '.tif')
#      #dsm.rotate(90).save(
#      #    '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/dsm/' + 'ID_' + str(
#      #        count) + '.tif')
#      masks.rotate(90).save(
#          '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/masks/' + 'ID_' + str(
#              count) + '.tif')
#      #osm.rotate(90).save(
#      #    '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/osm/' + 'ID_' + str(
#      #        count) + '.tif')
#      count += 1
#      frames.rotate(180).save(
#          '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/frames/' + 'ID_' + str(
#              count) + '.tif')
#      #dsm.rotate(180).save(
#      #    '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/dsm/' + 'ID_' + str(
#      #        count) + '.tif')
#      masks.rotate(180).save(
#          '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/masks/' + 'ID_' + str(
#              count) + '.tif')
#      #sm.rotate(180).save(
#      #   '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/osm/' + 'ID_' + str(
#      #       count) + '.tif')
#      count += 1
#      frames.rotate(270).save(
#          '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/frames/' + 'ID_' + str(
#              count) + '.tif')
#      #dsm.rotate(270).save(
#      #    '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/dsm/' + 'ID_' + str(
#      #        count) + '.tif')
#      masks.rotate(270).save(
#          '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/masks/' + 'ID_' + str(
#              count) + '.tif')
#      #osm.rotate(270).save(
#      #    '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/osm/' + 'ID_' + str(
#      #        count) + '.tif')
#      count += 1
#

#def f(i):
#    # image = np.load(PATH_INPUT_RGB + '/' + n[i])
#    # print('image size',imagePath[i], np.asarray(image).shape)
#    # print('image name', n[i])
#
#    # gt = np.load(PATH_INPUT_R + '/' + m[i])
#    # img = Image.fromarray(gt)
#    img = Image.open(PATH_INPUT_R + '/' + m[i])
#    #img_d = Image.open(PATH_INPUT_DSM + '/' + m[i])
#    img_rgb = Image.open(PATH_INPUT_RGB + '/' + m[i])
#    #img_osm = Image.open(PATH_INPUT_OSM + '/' + m[i])
#    gt = np.array(img)
#
#    vect = collections.Counter(gt.flatten())
#
#    if vect[1] >= 1000:  # buildings
#        # print('class 1  pixels {} in image {}'.format(vect[1], m[i]))
#        img_rgb.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/frames/' + n[i])
#        #img_d.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/dsm/' + o[i])
#        #img_osm.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/osm/' + p[i])
#        img.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/masks/' + m[i])
#
#    elif vect[4] >= 100:  # piscine
#        # print('class 1  pixels {} in image {}'.format(vect[1], m[i]))
#        img_rgb.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/frames/' + n[i])
#        #img_d.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/dsm/' + o[i])
#        #img_osm.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/osm/' + p[i])
#        img.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/masks/' + m[i])
#
#    #elif vect[6] >= 1000000:
#    #    img_rgb.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/frames/' + n[i])
#    #    img_d.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/dsm/' + o[i])
#    #    img_osm.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/osm/' + p[i])
#    #    img.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/masks/' + m[i])
#    #    # return n[i]
#    #
#    #elif vect[10] >= 100000:
#    #    img_rgb.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/frames/' + n[i])
#    #    img_d.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/dsm/' + o[i])
#    #    img_osm.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/osm/' + p[i])
#    #    img.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/masks/' + m[i])
#    #    # return n[i]
#    #
#    #elif vect[17] >= 500000:
#    #    img_rgb.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/frames/' + n[i])
#    #    img_d.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/dsm/' + o[i])
#    #    img_osm.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/osm/' + p[i])
#    #    img.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/masks/' + m[i])
#    #
#    #elif vect[8] >= 20000:
#    #    img_rgb.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/frames/' + n[i])
#    #    img_d.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/dsm/' + o[i])
#    #    img_osm.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/osm/' + p[i])
#    #    img.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/masks/' + m[i])
#    #
#    #
#    #elif vect[9] >= 20000:
#    #    img_rgb.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/frames/' + n[i])
#    #    img_d.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/dsm/' + o[i])
#    #    img_osm.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/osm/' + p[i])
#    #    img.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/masks/' + m[i])
#    #
#    #
#    #elif vect[11] >= 2000000:
#    #    img_rgb.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/frames/' + n[i])
#    #    img_d.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/dsm/' + o[i])
#    #    img_osm.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/osm/' + p[i])
#    #    img.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/masks/' + m[i])
#    #
#    #elif vect[13] >= 200000:
#    #    img_rgb.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/frames/' + n[i])
#    #    img_d.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/dsm/' + o[i])
#    #    img_osm.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/osm/' + p[i])
#    #    img.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/masks/' + m[i])
#    #
#    #elif vect[14] >= 200000:
#    #    img_rgb.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/frames/' + n[i])
#    #    img_d.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/dsm/' + o[i])
#    #    img_osm.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/osm/' + p[i])
#    #    img.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/masks/' + m[i])
#    #
#    #elif vect[16] >= 200000:
#    #    img_rgb.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/frames/' + n[i])
#    #    img_d.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/dsm/' + o[i])
#    #    img_osm.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/osm/' + p[i])
#    #    img.save('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/masks/' + m[i])
#
#    else:
#        pass
#
#    vect.clear()
#
#
#
#starttime = time.time()
## processes = []
#
## MULTI-PROCESSING FOR IMAGE STAT CALCULATION (ALL cpu BY DEFAULT)
#pool = Pool()
#pool.map(f, [i for i in range(len(m))])
#pool.close()
#
# for i in range(20):
#    p = multiprocessing.Process(target=f, args=(i,))
#    processes.append(p)
#    p.start()

# for process in processes:
#    process.join()
#GEN_PATH_MASK = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/masks/*.tif'
#GEN_PATH_IMAGE = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/frames/*.tif'
#GEN_PATH_DSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/dsm/*.tif'
#GEN_PATH_OSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/osm/*.tif'
## test
#
#gen_n = sorted(glob.glob(GEN_PATH_IMAGE))  # List of testing images
#gen_m = sorted(glob.glob(GEN_PATH_MASK))  # List of testing mask
#gen_o = sorted(glob.glob(GEN_PATH_DSM))  # List of testing DSM
#gen_p = sorted(glob.glob(GEN_PATH_OSM))  # List of testing DSM
#gen_m.sort()
#gen_n.sort()
#gen_o.sort()
#gen_p.sort()
## print('affiche liste mask',gen_m)
#print('nbre fichier image', len(gen_n))
#print('nbre fichier dsm', len(gen_o))
#print('nbre fichier osm', len(gen_p))
#print('nbre fichier mask', len(gen_m))
## kernel3 = np.ones((3, 3), np.uint8)
## kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
## kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
#
## for i in range(len(gen_m)):
## images_dsm_osm_masks = list(zip(gen_n[0:10], gen_o[0:10], gen_m[0:10], gen_p[0:10]))
#
## print('..................................',images_dsm_osm_masks[0])
## print(np.asarray(images_dsm_osm_masks).shape)
#
#
## images = [[[(Image.open(x)) for y in x] for x in z] for z in images_dsm_osm_masks]
## images = [[[print(x) for y in x] for x in z] for z in images_dsm_osm_masks]
## images = [[np.asarray(Image.open(y)) for y in x] for x in images_dsm_osm_masks]
## images = [[[np.asarray(Image.open(y)) for y in x] for x in z] for z in tess]
## print(np.array(images).shape)
## print('len', len(images))
## print('images',images[0][0])
#
#rot(gen_n, gen_m, gen_o, gen_p, count)
#
#
#print('saving and moving augmented files')
#
## move augmented files into processing folder
#
#for filename in glob.glob(os.path.join('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/dsm', '*.*')):
#    shutil.copy(filename, '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_dsm/img')
#
#for filename in glob.glob(os.path.join('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/osm', '*.*')):
#    shutil.copy(filename, '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_osm/img')
#
#for filename in glob.glob(os.path.join('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/frames', '*.*')):
#    shutil.copy(filename, '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_frames/img')
#
#for filename in glob.glob(os.path.join('/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/generated/output/masks', '*.*')):
#    shutil.copy(filename, '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/train_masks/img')
#
#print('Processing data augmentation time {} seconds'.format(time.time() - starttime))
#