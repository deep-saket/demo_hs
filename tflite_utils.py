import concurrent.futures
import collections
import dataclasses
import hashlib
import itertools
import json
import math
import os
import pathlib
import random
import re
import string
import time
import urllib.request

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import requests
import tqdm

import tensorflow as tf

def save_tflites(model, exp_name, epoch):
  if not os.path.isdir(f'{exp_name}/{epoch}'):
    os.makedirs(f'{exp_name}/{epoch}')

  model.save(f'{exp_name}/{epoch}/HandSeg')

  # Convert the model
  converter = tf.lite.TFLiteConverter.from_saved_model(f'{exp_name}/{epoch}/HandSeg') # path to the SavedModel directory
  tflite_model = converter.convert()

  # Save the model.
  with open(f'{exp_name}/{epoch}/HandSeg.tflite', 'wb') as f:
    f.write(tflite_model)

def save_tflites_from_saved_models(saved_model_dir, tflite_path):
  # Convert the model
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
  tflite_model = converter.convert()

  # Save the model.
  with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

def save_tflites_ckpt(model, ckpt_path, tflite_path):
  ## model.set_training(False)
  ## checkpoint = tf.train.Checkpoint(net=model)
  ## if ckpt_path != '':
  ##      if os.path.exists(ckpt_path):
  ##          print(f'Restoring checkpoint from {ckpt_path}', end='\r')
  ##          checkpoint.restore(tf.train.latest_checkpoint(ckpt_path))
  
  model.load_weights(ckpt_path)

  model.save(f'{ckpt_path}/HandSeg')

  # Convert the model
  converter = tf.lite.TFLiteConverter.from_saved_model(f'{ckpt_path}/HandSeg') # path to the SavedModel directory
  tflite_model = converter.convert()

  # Save the model.
  with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
  
  model.set_training(True)