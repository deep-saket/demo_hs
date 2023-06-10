# Hand Segmentaion
This codebase is developed and maintained by Saket Mohanty, This code base uses mlcv framework, which is build by Saket and will be make available through pypi.

### Requiremets
1. Use the environment.yml to create conda environment

### Running this project
3. First Train the model.
    ```
    source run.sh
    ```
2. Now to test the model you just need to 
    ```
    python infer_ckpt.py --image_dir <PATH_TO_THE_DIR_CONTAINING_VIDEO>  --output_dir <PATH_TO_THE_DIR_WHERE_OUTPUT_VIDEO_WILL_BE_STORED> --restore_from <PATH_TO_CKPT>
    ```
### Trained Model
The trained model for this model can be found <a href="https://drive.google.com/drive/folders/15If0uH04UEdvV94S_k76JSrLAXD5IFdO?usp=drive_link"> here. </a>

### IOU
IOU on single image for the model provided is 71%.



