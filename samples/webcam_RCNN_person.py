


























import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import datetime
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)



# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

print("webcam")
cap = cv2.VideoCapture(0)
t_old = datetime.datetime.now()
drawing = np.zeros((256,256,3),np.uint8)
im = np.zeros((256,256,3),np.uint8)

print("start    ")

while(True):

    t_now = datetime.datetime.now()
    #print("time", t_now- t_old)
    datetime.datetime.now()
    ret, frame = cap.read()

    now_frame = cv2.resize(frame,(256,256))
    mask = np.zeros((256,256,3),np.uint8)
    results = model.detect([now_frame], verbose=1)
    r = results[0]

    """
    visualize.display_instances(now_frame, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])

    """
   
    print(r['class_ids'])
    mask = np.zeros((256,256,3),np.uint8)
    for i in range(len(r['class_ids'])):
        if (r['class_ids'][i]==1):
            mask[:,:,0] = mask[:,:,0] + r['masks'][:,:,i] 
            mask[:,:,1] = mask[:,:,1] + r['masks'][:,:,i] 
            mask[:,:,2] = mask[:,:,2] + r['masks'][:,:,i] 

    
    drawing = im *(np.ones((256,256,3),np.uint8)- mask)
    #drawing = np.uint8(drawing * 0.999)

    im = now_frame * mask +drawing

    #im_final =  np.uint8(im * 0.8) + np.uint8(now_frame * 0.2)
    im_final = im
    cv2.imshow('frame',cv2.resize(im_final,(1000,1000)))
    t_old = t_now

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

















