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
    Adjust the hsv values from the track bar to segment your hand color.

    ```

### Trained Model
The trained model for this model can be found <a href=""> here. </a>

### IOU
IOU on single image for the model provided is 71%.


## Sample video
Like - https://drive.google.com/file/d/1et-BiF64NgO8yVH4cSOeABJTEbULVOtU/view?usp=drive_link




