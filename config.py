DATA_DIR = "./data/VOC2012"

DIRECTORY_ANNOTATIONS = 'Annotations/'

DIRECTORY_IMAGES = 'JPEGImages/'

MODEL_DIR = './checkpoints/'

EXAMPLE_PER_FIEL = 200

WEIGHT_DIR = './pretrained/YOLO.ckpt'

VOC_LABELS = {
    'aeroplane': (0, 'Vehicle'),
    'bicycle': (1, 'Vehicle'),
    'bird': (2, 'Animal'),
    'boat': (3, 'Vehicle'),
    'bottle': (4, 'Indoor'),
    'bus': (5, 'Vehicle'),
    'car': (6, 'Vehicle'),
    'cat': (7, 'Animal'),
    'chair': (8, 'Indoor'),
    'cow': (9, 'Animal'),
    'diningtable': (10, 'Indoor'),
    'dog': (11, 'Animal'),
    'horse': (12, 'Animal'),
    'motorbike': (13, 'Vehicle'),
    'person': (14, 'Person'),
    'pottedplant': (15, 'Indoor'),
    'sheep': (16, 'Animal'),
    'sofa': (17, 'Indoor'),
    'train': (18, 'Vehicle'),
    'tvmonitor': (19, 'Indoor'),
}

IMG_SIZE = 448

BATCH_SIZE = 16

CELL_SIZE = 7

CLASSES = 20

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0

BOXES_PER_CELL = 2
ALPHA = 0.1
