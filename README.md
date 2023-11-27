# Repository for WBK Hackathon

run ´pip install -r requirements.txt´ to install all necessary requirements

If you get the following Error "module 'keras.utils.generic_utils' has no attribute 'get_custom_objects' when importing segmentation_models" add 

´´´
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
´´´
at the top of your code
