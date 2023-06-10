import tensorflow as tf
from pprint import pprint

from train_support import *
from model import *

from datetime import datetime
import sys
import os
import cv2

import argparse

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Network")
    parser.add_argument("--image_dir", type=str, default='',
                        help="Path to the directory containing the input images.")
    parser.add_argument("--output_dir", type=str, default='',
                        help="Path to the directory where the program will store output image.")
    parser.add_argument("--output_vid_path", type=str, default='',
                        help="Path to store the op video")
    parser.add_argument("--restore_from", type=str, default='',
                        help="Where restore model parameters from.")

    return parser.parse_args()

def infer_step(X):
    '''
    Infer one minibatch
    '''
    Y_pred = model(X)
    return Y_pred

if __name__ == '__main__':
    print('###################################')
    args = get_arguments()
    restore_from = args.restore_from # './checkpoints/ex2/max_val_iou_coef/max_val_iou_coef_22'
    image_dir  = args.image_dir #'/media/saket/Elements/test/Nymble/frames_KT_25/'   # '/media/saket/Elements/datasets/handpalm/hand_seg/egohos_dataset/val/images'
    output_dir = args.output_dir #'/media/saket/Elements/test/Nymble/frames_KT_25_op_pcex2/'

    image_shape = [224, 224]
    ## Create model and model trainer
    model = unet_model(output_channels=1, shape=image_shape + [3,])
    model.load_weights(restore_from)
    trainInfer = TrainInferSegmodelBinary(model, image_shape)
    
    # Compile the model
    trainInfer.compile_for_test(infer_step)

    # load checkpoint
    # trainInfer.load_checkpoint(restore_from)

    # inference
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]

    for input_image_path in input_image_paths:
        print('########', os.path.basename(input_image_path))
        output_image_path = os.path.join(output_dir, os.path.basename(input_image_path))

        image = cv2.cvtColor(cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        output_image = trainInfer.infer(image)

        # Apply the overlay on the base image using the mask
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(output_image_path, output_image)

    if args.output_vid_path != '':
        os.system(f'ffmpeg -i {os.path.join(output_dir, "%05d.png")} {args.output_vid_path}')
    else:
        os.system(f'ffmpeg -i {os.path.join(output_dir, "%05d.png")} {output_dir if output_dir[-1] != "/" else output_dir[:-1]}.mp4')

