import sys
import os
import cv2
import numpy as np

input_labels_dir = sys.argv[1]
output_labels_dir = sys.argv[2]

if not os.path.exists(output_labels_dir):
    os.makedirs(output_labels_dir)

input_label_paths = [os.path.join(input_labels_dir, input_lable_name) for input_lable_name in os.listdir(input_labels_dir)]

for input_label_path in input_label_paths:
    print('########', os.path.basename(input_label_path))
    output_label_path = os.path.join(output_labels_dir, os.path.basename(input_label_path))

    label = cv2.imread(input_label_path, cv2.IMREAD_UNCHANGED)
    print(np.unique(label))
    
    label *= 255

    cv2.imwrite(output_label_path, label) 