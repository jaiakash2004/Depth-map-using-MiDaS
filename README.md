# Depth-map-using-MiDaS
Run inference on TensorFlow-model by using TensorFlow
Download the model weights model-f6b98070.pb and model-small.pb and place the file in the /tf/ folder.

Set up dependencies:

# install OpenCV
pip install --upgrade pip
pip install opencv-python

# install TensorFlow
pip install -I grpcio tensorflow==2.3.0 tensorflow-addons==0.11.2 numpy==1.18.0

Usage
Place one or more input images in the folder tf/input.

Run the model:

python tf/run_pb.py
Or run the small model:

python tf/run_pb.py --model_weights model-small.pb --model_type small
The resulting inverse depth maps are written to the tf/output folder.
