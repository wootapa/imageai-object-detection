# Folders:
Folders here should have same naming as its class/category. Ex, bird, dog, cat, person etc.
Each of these folders should have a train and validation folder
Each of those folders should have a annotation and images folder.

Example:
- bird
    - train
        - annotations
        - images
    - validation
        - annotations
        - images

Read here: https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/Custom/CUSTOMDETECTIONTRAINING.md

After training, bird folder will contain json/models folders (and other temp folders). These contain the files you need for object detection and should be copied into the inference app. 

However, for inference app to use it, it needs codechanges. See more:
https://github.com/OlafenwaMoses/ImageAI/blob/master/imageai/Detection/Custom/CUSTOMDETECTION.md

