import os
import numpy as np
import tensorflow as tf
# from tflite_runtime.interpreter import Interpreter
import argparse
import time
import cv2

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--input_image_path", type=str, 
                        help="Path to the (directory) input image to be inferred")
    parser.add_argument("--output_image_path", type=str, 
                        help="Path (to the directory) where output image to be stored")
    parser.add_argument("--tflite_path", type=str, default='',
                        help="Path to restore the ckpt from") 


    return parser.parse_args()


def normalize_input(input_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    '''
    Normalizes the input image across channels given mean and sd

    Arguments ::
        input_image -- ndarray | input RGB image
        mean -- list of floats | mean values of all 3 channels (i.e. r, g and b)
        std -- list of floats | standard deviation of all three channels
    Returns ::
        normalized_input_image -- ndarray | normalized input image
    '''
    normalized_input_image = (input_image - np.array(mean).reshape(1, 1, len(mean))) / np.array(std).reshape(1, 1, len(std))

    return normalized_input_image

def norm(in_image, std=[0.229, 0.224, 0.225], mean=[0.485, 0.456, 0.406]):
    '''
    This function normalizes the input image
    
    Args ::
        in_image -- ndarray | input image
        std -- list | default [0.229, 0.224, 0.225] | standard deviation of the channels
        mean -- list default [0.485, 0.456, 0.406] | mean of the channels
    Return ::
    	norm_image -- ndarray | normalized image
    '''
    norm_image = np.zeros_like(in_image)
    for c in [0, 1, 2]:
        norm_image[c] = (in_image[c] - mean[c]) / std[c]
        
    return norm_image

def preprocess_input(input_image_path, shape=None, normalize=True, dtype=np.float32):
    '''
    Reads and preprocesses the input image.

    Arguments ::
        input_image_path -- str | path where input image is stored
        shape -- tupple | shape to resize the input image in the form (w, h) | default None
                    | if None, does not resize the input image
        normalize -- bool | if set True, normalizes the input image
                        | default False
        dtype -- datatype of the input to the model | default np.float32
    Returns ::
        input_image -- ndarray | preprocessed input image
    '''
    if dtype == np.int8:
        return cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)[np.newaxis, :, :, :].transpose(0, 3, 1, 2).astype(dtype)
    
    input_image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
    
    gr_image =  cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    input_image = np.ones_like(input_image) * gr_image
    
    if np.max(input_image) > 2:
        input_image = input_image / 255.

    if shape != None:
        input_image = cv2.resize(input_image, shape)

    # input_image = (input_image)[np.newaxis, :, :, :].transpose(0, 3, 1, 2).astype(dtype)
    input_image = input_image[np.newaxis, :, :, :].astype(np.float32) # .transpose(0, 3, 1, 2).astype(np.float32)

    return input_image

def infer_uno(interpreter, input_details, output_details, input_image_path, output_image_path):
    '''
    This function infers one example.
    
    Arguments ::
        interpreter -- tflite interpreter | torch model with pretrained weights
        input_details -- dict | input details to the interpreter
        output_details -- dict | output details of the interpreter
        input_image_path -- str | input image path
        output_image_path -- srt | path where output image will be saved
    '''
    print(f'Processing {input_image_path}')
    
    ## 1. load the input image
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    input_image_tensor = preprocess_input(input_image_path, shape=(width, height), dtype=input_details[0]['dtype'])
    
    i = np.ones(input_details[0]['shape']).astype(np.float32)
    ## 2. infer
    start_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_image_tensor)
    interpreter.invoke()
    
    heatmap = interpreter.get_tensor(output_details[0]['index'])

    binarized_map = (heatmap > 0.4).astype(np.uint8)
    binarized_map = binarized_map[0, :, :, 0] * 255

    print(np.unique(heatmap))
    # binarized_map = binarized_map[0, :, :, 0]
    # cv2.imwrite('b.png', binarized_map*255)

    
    cv2.imwrite(output_image_path, binarized_map.astype(np.uint8))
    # cv2.imwrite(output_image_path + '5.png', binarized_map[0, 0] * 255)    
    
    print(f'{input_image_path} >> output saved to >> {output_image_path}')
    
def infer(model, input_details, output_details, input_image_dir, output_image_dir):
    '''
    This function infers from a directory full of images
    
    Arguments ::
        model -- torch.nn.Module | torch model with pretrained weights
        input_details -- dict | input details to the interpreter
        output_details -- dict | output details of the interpreter
        input_image_path -- str | input image dir
        output_image_path -- srt | dir where output image will be saved
    '''
    for input_image_name in os.listdir(input_image_dir):
        input_image_path = os.path.join(input_image_dir, input_image_name)
        output_image_path = os.path.join(output_image_dir, input_image_name)
        
        infer_uno(model, input_details, output_details, input_image_path, output_image_path)

if __name__ == '__main__':    
    ## 1. arguments
    args = get_arguments()

    ## 2. create model
    interpreter = tf.lite.Interpreter(args.tflite_path) 

    ## 3. Get model details
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ## 4. call the infer function
    if os.path.isdir(args.input_image_path):
        if not os.path.exists(args.output_image_path):
            os.makedirs(args.output_image_path)
        infer(interpreter, input_details, output_details, args.input_image_path, args.output_image_path)    
    elif os.path.isfile(args.input_image_path):
        infer_uno(interpreter, input_details, output_details, args.input_image_path, args.output_image_path)
        