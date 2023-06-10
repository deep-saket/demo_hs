import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from matplotlib import pyplot as plt

from datetime import datetime
import sys
import os
import glob
from tensorflow.keras.losses import *
import numpy as np
from pprint import pprint

class TrainInfer:
    '''
    This class hepls in training.

    Args ::
        model -- tf.keras.nn.Model        
    '''
    def __init__(self, model, image_shape) -> None:
        self.model = model
        self.OPTIMIZER_DICT = {
                    "Adadelta" : tf.keras.optimizers.Adadelta, 
                    "Adagrad" : tf.keras.optimizers.Adagrad, 
                    "Adam" : tf.keras.optimizers.Adam, 
                    "Adamax" : tf.keras.optimizers.Adamax, 
                    "Ftrl" : tf.keras.optimizers.Ftrl, 
                    "Nadam" : tf.keras.optimizers.Nadam, 
                    "SGD" : tf.keras.optimizers.SGD 
        }
        self.OPTIMIZER_ARG = {
                            "Adadelta" : {'rho': 0.95, 'epsilon' : 1e-07, 'name' : 'Adadelta'}, 
                            "Adagrad" : tf.keras.optimizers.Adagrad, 
                            "Adam" : {'beta_1' : 0.9, 'beta_2' : 0.999, 'epsilon' : 1e-07, 'amsgrad' : False, 'name' : 'Adam'},
                            "Adamax" : tf.keras.optimizers.Adamax, 
                            "Ftrl" : tf.keras.optimizers.Ftrl, 
                            "Nadam" : tf.keras.optimizers.Nadam, 
                            "SGD" : tf.keras.optimizers.SGD 
        }
        self.is_compiled = False
        self.is_implemented_train_iter = True
        self.is_implemented_val_iter = True

        self.image_shape = image_shape
        
        ## allow gpu growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        gpus = ['gpu:'+gpu.name[-1] for gpu in gpus]
        print(f'GPUs : {gpus}')

    def get_optimizer_arg(self, optimizer_name):
        '''
        Returns all the optimizer arguments
        Arguments --
            optimizer_name -- string | one of ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "SGD"]
        '''
        return self.OPTIMIZER_ARG[optimizer_name]
    
    def create_optimizer(self, optimizer_fnc, lr, other_params = {}):
        '''
        Creates one of the optimizers present in tf.keras.optimizers and returns it.

        Args --
            optimizer_func -- function | optimizer's creation function
            lr -- float or float tensor or learning_rate_function | learning rate
            other_params -- dict | default {} | contains all the arguments needed to create the optimizer
        Return --
            other_params -- tf.keras.optimizers.*
        '''
        other_params = other_params.values()
        optimizer = optimizer_fnc(lr) #, *other_params)

        return optimizer
    
    def compile(self, train_step, test_step, compute_cost, lr, optimizer_name, eval_metrics = {},
                weighted_cost=False):
        '''
        Prepares the model for training and returns True on success.

        Args ::
            train_step -- function | train for 1 iteration | takes 6 arguments 
                                            minibatch_X -- feature minibatch
                                            minibatch_Y -- label minibatch
                                            model -- CV model
                                            compute_cost -- cost function
                                            optimizer -- optimizer
            test_step -- function | test for 1 iteration | takes 5 arguments 
                                            minibatch_X -- feature minibatch
                                            minibatch_Y -- label minibatch
                                            model -- CV model
                                            compute_cost -- cost function
            compute_cost - function | calculates loss | takes 3 arguments
                                            Y - gt labels
                                            Y_pred - predicted labels
                                            vgg -- vgg model if use_vgg is true
            lr -- float or float tensor or learning_rate_function | learning rate
            optimizer_name -- string | one of ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "SGD"]
            eval_metrics -- python dict | default {} | dict of metrics to be evaluated |
                                            the dict should contain eval step function as values and 
                                            metric name as keys
            weighted_cost -- boolean | default False | cost is supplied with weight, when True
        '''
        self.train_step = train_step
        self.test_step = test_step
        self.compute_cost = compute_cost
        self.weighted_cost = weighted_cost

        ## creating optimizer
        optimizer_arg_default = self.OPTIMIZER_ARG[optimizer_name] if optimizer_name in self.OPTIMIZER_ARG.keys() else {}

        optimizer_arg = self.get_optimizer_arg(optimizer_name)
        pprint(optimizer_arg)

        for k, v in optimizer_arg.items():
            if k in optimizer_arg_default.keys():
                optimizer_arg_default[k] = v
        
        self.optimizer_arg = optimizer_arg_default
    
        if optimizer_name not in self.OPTIMIZER_DICT.keys():
            print(f'Invalid optimizer option')
            print(f'Optimizer should be one of  : {self.OPTIMIZER_DICT.keys()}')


        self.lr = lr
        self.optimizer = self.create_optimizer(self.OPTIMIZER_DICT[optimizer_name], lr, optimizer_arg)
        self.eval_metrics = eval_metrics
        self.is_compiled = True

    def compile_for_test(self, infer_step):
        self.infer_step = infer_step

    def load_checkpoint(self, checkpoint_path, optimizer = True, wights_only = False):
        '''
        checkpoint_path - str | default '' | specifies the saved checkpoint path to restore
                                            the model from
        '''
        if wights_only:
            self.model.load_weights(checkpoint_path)

        checkpoint = None
        if self.is_compiled and optimizer and not wights_only:
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, net=self.model)
        else:
            checkpoint = tf.train.Checkpoint(net=self.model)
        if checkpoint_path != '' and not wights_only:
            if os.path.exists(checkpoint_path):
                print(f'Restoring checkpoint from {checkpoint_path}', end='\r')
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

    def train_iter(self, data_dict, iter, **kwds):
        self.is_implemented_train_iter = False
        pass
    
    def val_iter(self, data_dict, iter, **kwds):
        self.is_implemented_val_iter = False
        pass

    def train(self, train_dataset, dev_dataset, batch_size, epochs,                             
                    callbacks = [], steps_per_epoch = None, validation_steps = None,
                    start_epoch=0, dataset_name = '', loss_name = '',
                    model_name = '', show_results=-1):
        '''
        This function gets executed on executing the script.
        
        Args ::
            train_dataset -- Training set
            dev_dataset -- Dev set            
            batch_size -- int | number of batches used
            epochs -- int or int tensor | number of epochs            
            callbacks - list | default [] | list of callbacks. keras.callbacks.Callback instances
            steps_per_epoch - int | default None | when given, runs given number of iterations per epoch
            validation_steps - int | default None | when given, runs given number of val iterations per epoch
            start_epoch -- int | default 0 | starting epoch            
            dataset_name -- str | default '' | name of the dataset used in trainiing
            loss_name -- str | default '' | name of the loss(es) used in trainiing
            model_name -- str | default '' | name of the model used in trainiing
            show_results -- int | default -1 | if set between 0 to epochs; computes
                                        metrics and displayes results from dev set
                                        in that intervals
        '''    
        if not self.is_compiled:
            print('[INFO] Please call TrainInfer.compile() before calling TrainInfer.train()')
            return
        if not self.is_implemented_train_iter:
            print('[INFO] Please override TrainInfer.train_iter before calling TrainInfer.train()')
            return
        if not self.is_implemented_val_iter:
            print('[INFO] Please override TrainInfer.val_iter before calling TrainInfer.train()')
            return
        ## hyperparameters
        lr = self.lr
        PARAMS = {
                'start-lr' : lr,
                'batch-size' : batch_size,
                'dataset-name' : dataset_name,
                'loss-name' : loss_name,
                'model-name' : model_name
        }

        ## initialize data loader
        n_minibatches = train_dataset.count_minibatches()
        n_minibatches_dev = dev_dataset.count_minibatches()

        print(f'Total number of training examples = {train_dataset.m}')         
        print(f'Start epoch - {start_epoch} | End epoch - {start_epoch + epochs}')
        print(f'Number of minibatches in training set - {n_minibatches}')
        print('Starting training...')

        logs = {}
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = n_minibatches
        elif steps_per_epoch < 0:
            print(f'[INFO] steps_per_epoch can not be -ve, but found {steps_per_epoch}')  
            return
        if validation_steps is None or validation_steps == 0:
            validation_steps = n_minibatches_dev  
        elif validation_steps < 0:
            print(f'[INFO] validation_steps can not be -ve, but found {validation_steps}')  
            return
        for epoch in range(start_epoch, start_epoch+epochs):
            costs = []
            dev_costs = []
            dev_metric = {f'val_{k}' : [] for k, v in self.eval_metrics.items()}
            minibatch_cost = 0
            dev_minibatch_cost = 0
            dev_minibatch_metric = {f'val_{k}' : 0 for k, v in self.eval_metrics.items()}
            dev_epoch_metric = {f'val_{k}' : 0 for k, v in self.eval_metrics.items()}

            ## iterate over minibatches
            for iteration in range(steps_per_epoch):
                step = (iteration + 1) + (epoch * steps_per_epoch)

                ## fetch one minibatch
                data_dict = train_dataset.get_data()            

                temp_cost = self.train_iter(data_dict, step)
                minibatch_cost += temp_cost            

                if iteration > 0:
                    sys.stdout.write("\033[K")
                print(f'{iteration + 1}/{steps_per_epoch} minibatches processed | {step} iterations | cost - {temp_cost}', end='\r')
                
                step_lr = lr(step) if not isinstance(lr, float) else lr            

                ## update model in callbacks
                for callback in callbacks:
                    callback.model = self.model   
                

            ## track cost
            costs.append(minibatch_cost) # /len(minibatch_cost))
            minibatch_cost = 0
            sys.stdout.write("\033[K")
            print(f'Training set cost after {epoch} epochs =  {costs[-1]}')

            ## evaluate if show_result in greater than 0 and after every show_result epochs
            if show_results > 0 and epoch % show_results == 0:
                ## iterate over dev set
                for iteration in range(validation_steps):
                    step = (iteration + 1) + (epoch * validation_steps)
                    ## fetch one minibatch
                    data_dict = dev_dataset.get_data()                               
                    temp_cost, iter_metric = self.val_iter(data_dict, step)

                    dev_minibatch_cost += temp_cost
                    for kmetric, vmetric in iter_metric.items():
                        dev_minibatch_metric[kmetric] += vmetric / validation_steps

                    if iteration > 0:
                        sys.stdout.write("\033[K")
                    print(f'{iteration + 1}/{validation_steps} minibatches processed | dev cost - {temp_cost}', end='\r')
                
                ## track cost and PSNR
                dev_costs.append(dev_minibatch_cost) # /len(minibatch_cost))
                for kmetric, vmetric in self.eval_metrics.items():
                    dev_metric[f'val_{kmetric}'].append(dev_minibatch_metric[f'val_{kmetric}'])
                    dev_epoch_metric[f'val_{kmetric}'] = dev_metric[f'val_{kmetric}'][-1]
                sys.stdout.write("\033[K")
                print(f'Dev set cost after {epoch} epochs =  {dev_costs[-1]}') #'| PSNR = {dev_psnr[-1]}')

                ## epoch end callbacks
                logs = {'loss' : costs[-1],
                    'val_loss' : dev_costs[-1]}
                for key, value in  dev_epoch_metric.items():
                    logs[key] = value

                for callback in callbacks:
                    callback.on_epoch_end(epoch, logs)
            else:
                ## epoch end callbacks
                logs = {'loss' : costs[-1],
                    'val_loss' : dev_costs[-1]}
                for key, value in  dev_epoch_metric.items():
                    logs[key] = value

                for callback in callbacks:
                    callback.on_epoch_end(epoch, logs)

        ## train end callbacks
        for callback in callbacks:
            callback.on_train_end(logs)
