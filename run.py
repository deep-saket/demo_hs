import neptune.new as neptune

import tensorflow as tf
from pprint import pprint

from train_support import *
from dataset import *
# from train import *
from model import *
from eval import *
from callbacks import *

from datetime import datetime
import sys
import os
from arguments import *

def train_step(X, Y, W, model, compute_cost, optimizer):
    '''
    Train one minibatch
    '''
    with tf.GradientTape() as tape:
        Y_pred = model(X)
        cost = compute_cost(Y, Y_pred, W) 

    gradient = tape.gradient(cost, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    return cost, Y_pred

def test_step(X, Y, W, model, compute_cost):
    '''
    Infer one minibatch
    '''
    Y_pred = model(X)
    cost = compute_cost(Y, Y_pred, W)

    return cost, Y_pred

if __name__ == '__main__':
    print('###################################')
    args = get_arguments()

    ## Hyperparameters and variables
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    start_epoch = args.start_epoch
    dataset_name = args.dataset_name
    train_dataset_dir = args.train_dataset_dir
    val_dataset_dir = args.val_dataset_dir
    optimizer_name = args.optimizer_name
    ex_name = args.ex_name
    checkpoint_filepath = os.path.join(args.checkpoint_filepath, ex_name) + '/ckpt'
    restore_from = args.restore_from
    iteration_per_epoch = args.iteration_per_epoch
    val_iteration_per_epoch = args.val_iteration_per_epoch
    input_shape = args.input_shape.split(',')
    image_shape = [int(x) for x in input_shape]
    num_classes = args.num_classes
    log_dir = os.path.join(args.checkpoint_filepath, ex_name)
    print('Hyperparameters initialized')    

    ## Load datasets
    train_dataset = SegmentationDataset(train_dataset_dir, batch_size, 2, 
                                        image_size=image_shape, weights={1:1.5},
                                        hflip=True, vflip=True, rotate=True, blur=True, noise=True
                                        ) 
    dev_dataset = SegmentationDataset(val_dataset_dir, batch_size, 2, 
                                      image_size=image_shape, weights={1:1.5},
                                      hflip=True, vflip=True, rotate=True, blur=True, noise=True) 

    ## Create model and model trainer
    model = unet_model(output_channels=1, shape=image_shape + [3,])
    model.load_weights(restore_from)
    trainInfer = TrainInferSegmodelBinary(model)

    # Define Loss
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Define evaluation metrics
    eval_dict = {
        'dice_coef' : dice_coef,
        'iou_coef' : iou_coef
    }

    # Compile the model
    trainInfer.compile(train_step, test_step, bce, lr, optimizer_name, eval_dict)

    # load checkpoint
    # trainInfer.load_checkpoint(restore_from)

    # Create callbacks
    basis_dict = {
        'val_dice_coef' : 'max',
        'val_iou_coef' : 'max',
        'val_loss' : 'min'        
    }
    callbacks = [
        CkptCallback(log_dir, basis_dict, skip_from_start=0, frequency=1),
        TensorboardCallback(log_dir, basis_dict, image_shape = image_shape + [3]),
        # TFLiteCallback(log_dir, basis_dict, skip_from_start=4, frequency=5, on_epoch_end=False)
    ]

    # Train the model
    trainInfer.train(train_dataset, dev_dataset, batch_size, epochs,                             
                    callbacks = callbacks, start_epoch = start_epoch, 
                     show_results=1)
