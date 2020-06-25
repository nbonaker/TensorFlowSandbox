# TensorFlowSandbox

Some useful docs

Essential docs
https://www.tensorflow.org/guide

# Setup
https://www.tensorflow.org/install/pip
python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate
pip install tensorflow

# Check
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

# Make a jupyter kernel
python -m ipykernel install --user --name=tf_sandbox
