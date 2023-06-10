import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from pista import *
import numpy as np
import cv2
import json
import os

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  # input_mask -= 1
  return input_image, input_mask

def resize(input_image, input_mask, image_size):
  input_image = tf.image.resize(input_image, image_size)
  input_mask = tf.image.resize(input_mask, image_size,
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
  )
  return input_image, input_mask

def load_image(datapoint, image_size = (128, 128)):
  '''
  args ::
  datapoint -- dict | contains info about each datapoint
                    | {
                        "image" : array of the input image
                        "segmentation_mask" : array of the gt segmentation
                                mask of shapre (h, w), same as the image.
                                Contains categorical representation of the mask
                    }
  returns ::
    input_image : resized and normalized input image
    input_mask : resized and normalized input mask
  '''
  input_image = tf.image.resize(datapoint['image'], image_size)
  input_mask = tf.image.resize(
    datapoint['segmentation_mask'],
    image_size,
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
  )

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def oxford_IIIT_dataset(image_size):
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    def process_record(datapoint):
        return load_image(datapoint, image_size)

    train_images = dataset['train'].map(process_record, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = dataset['test'].map(process_record, num_parallel_calls=tf.data.AUTOTUNE)

    return train_images, test_images

class CustomDataset1:
    def __init__(self, train_data_dir, val_data_dir, image_size = (128, 128)) -> None:
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.image_size = image_size
        self.dataset = self.create_tfDataset()
        
    def get_json_path(self, path_tensor):
        file_path = bytes.decode(path_tensor.numpy())
        print("file_path: ", file_path,type(bytes.decode(path_tensor.numpy())))
        return file_path

    def load_from_json(self, json_path_eager_tensor, data_dir):
        data_dir = bytes.decode(data_dir.numpy())
        json_path = bytes.decode(json_path_eager_tensor.numpy())
        with open(json_path) as f:
            data = json.load(f)                

            image_name = os.path.basename(data['images'][0]['file_name']).split('-')[-1]
            image_path = os.path.join(os.path.join(data_dir, 'images'), image_name) ## path containing the image
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            
            borders = data['annotations'][0]['segmentation'][0]
            points = []
            for idx in range(len(borders) // 2):
                points.append([int(borders[2*idx]), int(borders[(2*idx)+1])])

            points = np.array(points)
            mask = np.zeros_like(image)
            
            mask = cv2.fillPoly(mask, pts=[points], color=1)[:, :, 0][:, :, np.newaxis]
            
            image, mask = resize(image, mask, self.image_size)
            image, mask = normalize(image, mask)

            return image.astype(np.float32), mask.astype(np.float32)
    
    def create_tfDataset(self):
      train_anno_dir = os.path.join(self.train_data_dir, 'annotations')
      val_anno_dir = os.path.join(self.val_data_dir, 'annotations')

      print(train_anno_dir, val_anno_dir)

      train_dataset = tf.data.Dataset.list_files(f"{train_anno_dir}/*.json")
      val_dataset = tf.data.Dataset.list_files(f"{val_anno_dir}/*.json")

      return {'train' : train_dataset, 'test' : val_dataset}

    def get_dataset(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        
        def process_record(datapoint):
            image, mask = tf.py_function(self.load_from_json, [datapoint, self.train_data_dir], \
                                        (tf.TensorSpec(shape=(self.image_size[0], self.image_size[1], 3), dtype=tf.float32), \
                                            tf.TensorSpec(shape=(self.image_size[0], self.image_size[1], 1), dtype=tf.uint8))) # (tf.float32, tf.float32))
            return image, mask     
        
        train_images = self.dataset['train'].map( \
                          process_record, num_parallel_calls=AUTOTUNE)
        test_images = self.dataset['test'].map( \
                          process_record, num_parallel_calls=AUTOTUNE)

        return train_images, test_images




class BinarySegLoader:
    def __init__(self, data_dir, batch_size, image_size = (128, 128), \
                    grayscale=False, hflip=False, vflip=False, rotate=False, blur=False, visualize=False) -> None:
        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.anno_dir = os.path.join(self.data_dir, 'annotations')

        self.batch_size = batch_size
        self.image_size = image_size

        self._fetched_batch = 0
        self.grayscale = grayscale
        self.rotate = rotate
        self.blur = blur
        self.hflip = hflip
        self.vflip = vflip
        self.visualize = visualize

        ## get number of records
        self.m = len(os.listdir(self.anno_dir))      
        
        ## prepare the dataset
        self.dataset = self._prepare_data()
        random.shuffle(self.dataset)                    

    def load_from_json(self, json_path):
        with open(json_path) as f:
            data = json.load(f)                

            image_name = os.path.basename(data['images'][0]['file_name'])
            image_path = os.path.join(self.image_dir, image_name) ## path containing the image
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            
            borders = data['annotations'][0]['segmentation'][0]
            points = []
            for idx in range(len(borders) // 2):
                points.append([int(borders[2*idx]), int(borders[(2*idx)+1])])

            points = np.array(points)
            mask = np.zeros_like(image)
            
            mask = cv2.fillPoly(mask, pts=[points], color=1)[:, :, 0][:, :, np.newaxis]
            
            image, mask = resize(image, mask, self.image_size)
            image, mask = normalize(image, mask)

            return image, mask
    
    def _create_minibatches(self):
      return random_mini_batches_from_SSD_no_gt(self.anno_paths, self.batch_size)

    def _prepare_data(self):
      anno_paths = [os.path.join(self.anno_dir, json_name) for json_name in os.listdir(self.anno_dir)]
      self.anno_paths = anno_paths
      return self._create_minibatches()

    def _get_weight(self, mask):
        '''
        this function computes weight to be used in the cost function.
        '''
        total = mask.shape[0] * mask.shape[1] * mask.shape[2]
        pos = np.sum(mask)
        weights = np.ones_like(mask)
        weights = tf.where(mask == 1, (total / pos) * weights, weights)
        return weights.numpy()
    
    def _get_item(self, anno_path):  
      image, anno = self.load_from_json(anno_path)
      weight = self._get_weight(anno)
      
      return image, anno, weight
    
    def _get_items(self, anno_paths):
        images = []
        annos = []
        weights = []
        for i in range(len(anno_paths)):
            image, anno, weight = self._get_item(anno_paths[i])

            images.append(image)
            annos.append(anno)
            weights.append(weight)
            
        images = np.array(images)
        annos = np.array(annos)
        weights = np.array(weights)
        
        return images, annos, weights 

    def get_data(self):
        '''
        fetch one batch at a time
        '''
        if self._fetched_batch >= len(self.dataset):
            self.dataset = self._create_minibatches()
            random.shuffle(self.dataset)
            self._fetched_batch = 0

        ## fetch 1 batch at a time from the iterator
        data_batch_paths = self.dataset[self._fetched_batch]
        self._fetched_batch += 1
        
        anno_paths = data_batch_paths
        images, labels, weights = self._get_items(anno_paths)
        data_batch = {'images': images, 'labels': labels, 'weights' : weights}

        return data_batch
    
    def count_minibatches(self):
        '''
        Returns number of minibatches.
        '''
        return len(self.dataset)



class SegmentationDataset(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size, num_classes, image_size = (128, 128), weights = {}, \
                    grayscale=False, hflip=False, vflip=False, rotate=False, blur=False, noise=False,
                    random_earsing=False,  visualize=False) :
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'annotations')
        self.batch_size = batch_size
        
        self.image_filenames = sorted(os.listdir(self.image_dir))
        self.label_filenames = sorted(os.listdir(self.label_dir))
        self.dataset = list(zip(self.image_filenames, self.label_filenames))
        # random.shuffle(self.dataset)

        self.num_classes = num_classes
        self.image_size = image_size
        self.weights = weights

        self._idx = 0
        self.m = len(self.dataset)

        self.general_aug = ImageDataGenerator(
                        rotation_range=45 if rotate else 0,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.3,
                        horizontal_flip=hflip,
                        vertical_flip=vflip,            
                    ) if hflip or vflip or rotate else None

        self.noise = ImageDataGenerator(
                        preprocessing_function=self.add_noise_to_random_images
                    ) if noise else None

        self.blur = ImageDataGenerator(
                        preprocessing_function=self.apply_random_blur
                    ) if blur else None
        # self.random_erasing
                    
    def add_noise_to_random_images(self, image):
        # Randomly choose whether to add noise or not
        # add_noise = np.random.choice([True, False])
        noise = np.random.normal(loc=0.0, scale=0.1, size=image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        return noisy_image
        
    def apply_random_blur(self, image):
        # Randomly choose whether to apply blur or not
        # apply_blur = np.random.choice([True, False])
        blurred_image = tf.image.random_blur(image, (3, 3))
        return blurred_image
    
    def apply_random_erasing(self, image, mask):
        p = 0.5  # Probability of applying random erasing
        s_l = 0.02  # Minimum erasing area
        s_h = 0.4  # Maximum erasing area
        r_1 = 0.3  # Minimum aspect ratio
        r_2 = 1 / r_1  # Maximum aspect ratio
        
        if np.random.rand() > p:
            return image, mask
        
        img_h, img_w, _ = image.shape
        img_area = img_h * img_w
        
        while True:
            target_area = np.random.uniform(s_l, s_h) * img_area
            aspect_ratio = np.random.uniform(r_1, r_2)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < img_w and h < img_h:
                x = np.random.randint(0, img_w - w)
                y = np.random.randint(0, img_h - h)
                
                # Replace the random area with the background class in the mask
                mask[y:y+h, x:x+w] = 0
                
                # Fill the erased region in the image with random pixel values
                image[y:y+h, x:x+w, :] = np.random.uniform(0, 1, (h, w, image.shape[2]))
                break
        
        return image, mask
        
    def __len__(self):
        return len(self.image_filenames) // self.batch_size
    
    def __getitem__(self, idx):
        if idx == 0:
          # batch_filenames = list(zip(batch_image_filenames, batch_label_filenames))
          # random.shuffle(batch_filenames)
          # batch_image_filenames, batch_label_filenames = zip(*batch_filenames)
          random.shuffle(self.dataset)

        batch_dataset = self.dataset[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_image_filenames = []
        batch_label_filenames = []
        for batch_image_filename, batch_label_filename in batch_dataset:
            batch_image_filenames.append(batch_image_filename)
            batch_label_filenames.append(batch_label_filename)

        # batch_image_filenames  = self.image_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        # batch_label_filenames = self.label_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]        
        
        batch_images = []
        batch_labels = []
        batch_weights = []
                
        for image_filename, label_filename in zip(batch_image_filenames, batch_label_filenames):
            image_path = os.path.join(self.image_dir, image_filename)
            label_path = os.path.join(self.label_dir, label_filename)
            
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=self.image_size)
            label = tf.keras.preprocessing.image.load_img(label_path, color_mode='grayscale', target_size=self.image_size)
            
            image = tf.keras.preprocessing.image.img_to_array(image)
            label = tf.keras.preprocessing.image.img_to_array(label)

            # Apply data augmentation to the image and label
            seed = np.random.randint(0, 2**32 - 1)
            if self.general_aug is not None:
                image = self.general_aug.random_transform(image, seed=seed)
                label = self.general_aug.random_transform(label, seed=seed)            
            if self.noise is not None:
                image = self.noise.random_transform(image, seed=seed)
            if self.blur is not None:
                image = self.blur.random_transform(image, seed=seed)
            
            # Normalize the image and label pixel values if needed
            image = image / 255.0
            label = label # / 255.0
            
            batch_images.append(image)
            batch_labels.append(label)

            # Calculate pixel weights based on class distribution in the label
            label_flat = label.flatten().astype(np.uint8)
            weights = np.zeros_like(label_flat, dtype=np.float32)
            for class_id in range(self.num_classes):
                class_pixels = np.count_nonzero(label_flat == class_id)
                if class_pixels > 0:                    
                    weights[label_flat == class_id] = self.weights.get(class_id, 1) * (1.0 / class_pixels)
            weights = weights.reshape(label.shape)
            batch_weights.append(weights)
        
        return tf.stack(batch_images), tf.stack(batch_labels), tf.stack(batch_weights)
    
    def get_data(self):
        '''
        fetch one batch at a time
        '''
        if self._idx >= len(self):
            self._idx = 0

        images, labels, weights = self[self._idx]
        self._idx += 1
        data_batch = {'images': images, 'labels': labels, 'weights' : weights}

        return data_batch
    
    def count_minibatches(self):
        '''
        Returns number of minibatches.
        '''
        return len(self)
