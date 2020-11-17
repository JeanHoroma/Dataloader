from datetime import datetime

time_stamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")

#(winW, winH) = (256, 256)
#STEP = 256

####################  binary label INRIA Database   #####################################
#PATH_INPUT_RGB = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/train/frames_originals'
# ground truth
#PATH_INPUT_R = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/train/masks_originals'

####################  binary label INRIA Database   #####################################
#PATH_OUTPUT = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/train/'
#PATH_OUTPUT_MASK = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/train/masks/'
#PATH_OUTPUT_IMAGE = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/train/frames/'



####################3 images segementee pour MRC  -- images sources --
PATH_INPUT_RGB = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/frames_originals'
# ground truth mask
PATH_INPUT_R = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/masks_originals'
# DSM
#PATH_INPUT_DSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/dsm'
# DSM -- open street map
#PATH_INPUT_OSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/osm'

######################  test ###################
#PATH_INPUT_RGB = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/DEBUG/frames'
# ground truth mask
#PATH_INPUT_R = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/DEBUG/masks'
# DSM
#PATH_INPUT_DSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/DEBUG/dsm'
# DSM -- open street map
#PATH_INPUT_OSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/DEBUG/osm'


####################3 images segementee pour MRC  -- images crop 1024x1024 pour  --
(winW, winH) = (256, 256)
STEP = 192
PATH_OUTPUT = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/256_img_size/'
PATH_OUTPUT_MASK = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/256_img_size/masks/'
PATH_OUTPUT_IMAGE = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/256_img_size/frames/'
#PATH_OUTPUT_DSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/dsm/'
#PATH_OUTPUT_OSM = '/media/DATA/DATA_LANDUSE/10.03__IMAGES__TUILES_EXTRAITES/1024img/osm/'









# PATH_OUTPUT_MASK = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/train/masks/'
# PATH_OUTPUT_IMAGE = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/train/frames/'
# PATH_OUTPUT_MASK = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/1024_img_size/masks/'
# PATH_OUTPUT_IMAGE = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/1024_img_size/frames/'
# PATH_OUTPUT = '/media/DATA/DATA_LANDUSE/INRIA/AerialImageDataset/1024_img_size/'


