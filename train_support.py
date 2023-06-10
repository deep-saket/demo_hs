import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import neptune.new as neptune
from matplotlib import pyplot as plt
import cv2
import tensorflow.keras.backend as K

from datetime import datetime
import sys
import os
import glob
import copy

from utils import *
from train_infer import *

CONTENT_LAYERS = ['block5_conv2']

class TrainInferSegmodelBinary(TrainInfer):   
    def train_iter(self, data_dict, iter):
        '''
        This function is what's executed every train epoch
        '''
        minibatch_X = data_dict['images']
        minibatch_Y = data_dict['labels']  
        minibatch_W = data_dict['weights']          

        ## train the model for one iteration
        temp_cost, Y_pred = self.train_step(minibatch_X, minibatch_Y, minibatch_W, self.model, self.compute_cost, self.optimizer)

        if iter in [1, 10, 100, 1000, 100000, 30000] or iter % 1000 == 0:
            heatmap = Y_pred.numpy()
            binarized_map = (heatmap > 0.4).astype(np.uint8)
            binarized_map = binarized_map[0, :, :, 0] * 255

            if not os.path.isdir('logs/train'):
                os.makedirs('logs/train')

            # print(np.max(heatmap), np.min(heatmap), np.mean(heatmap))
            # print(np.unique(binarized_map))
            # binarized_map = binarized_map[0, :, :, 0]
            # cv2.imwrite('b.png', binarized_map*255)

            # print((minibatch_Y[0].astype(np.uint8) * 255).shape)
            cv2.imwrite(f'logs/train/t{iter}_ip.png', (minibatch_X.numpy()[0] * 255).astype(np.uint8))
            cv2.imwrite(f'logs/train/t{iter}.png', minibatch_Y.numpy()[0].astype(np.uint8) * 255)
            cv2.imwrite(f'logs/train/t{iter}_bm.png', binarized_map.astype(np.uint8))


        return temp_cost

    def val_iter(self, data_dict, iter):
        '''
        This function is what's executed every val epoch
        '''
        iter_metric = {f'val_{k}' : 0 for k, v in self.eval_metrics.items()}
        minibatch_X = data_dict['images']
        minibatch_Y = data_dict['labels']
        minibatch_W = data_dict['weights'] 

        if not isinstance(minibatch_Y, np.ndarray):
            minibatch_Y = minibatch_Y.numpy()

        ## calculate cost and Y_pred
        temp_cost, Y_pred = self.test_step(minibatch_X, minibatch_Y, minibatch_W, self.model, self.compute_cost)

        if iter % 100 == 0:
            heatmap = Y_pred.numpy()
            binarized_map = (heatmap > 0.4).astype(np.uint8)
            binarized_map = binarized_map[0, :, :, 0] * 255

            # print(np.max(heatmap), np.min(heatmap), np.mean(heatmap))
            # print(np.unique(binarized_map))
            # binarized_map = binarized_map[0, :, :, 0]
            # cv2.imwrite('b.png', binarized_map*255)

            if not os.path.isdir('logs/val'):
                os.makedirs('logs/val')
            # print((minibatch_Y[0].astype(np.uint8) * 255).shape)
            cv2.imwrite(f'logs/val/{iter}.png', minibatch_Y[0].astype(np.uint8) * 255)
            cv2.imwrite(f'logs/val/{iter}_bm.png', binarized_map.astype(np.uint8))

        ## calculate metrics
        for kmetric, vmetric in self.eval_metrics.items():            
            minibatch_Y = minibatch_Y.astype(np.float32)
            iter_metric[f'val_{kmetric}'] += vmetric(Y_pred, minibatch_Y)

        return temp_cost, iter_metric

    def infer(self, image):
        ## pre process
        h, w, _ = image.shape
        input_image = cv2.resize(image, self.image_shape)
        input_image = input_image / 255.
        input_image = input_image[None, :, :, :]

        ## infer
        Y_pred = self.infer_step(input_image)

        ## post process
        heatmap = Y_pred.numpy()
        binarized_map = (heatmap > 0.99).astype(np.uint8)
        binarized_map = binarized_map[0, :, :, 0] * 255
        print('##################', np.sum((heatmap > 0.99).astype(np.uint8)), np.sum((heatmap > 0.5).astype(np.uint8)))

        ## overlay mask
        binarized_map = cv2.resize(binarized_map, (w, h), cv2.INTER_NEAREST)
        mask = np.dstack((binarized_map, np.zeros_like(binarized_map), np.zeros_like(binarized_map)))

        # Apply the overlay on the base image using the mask
        output_image = cv2.addWeighted(image, 1, mask, 0.7, 0, dtype=cv2.CV_8U)

        return output_image
