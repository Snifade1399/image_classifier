import tensorflow as tf

# Check if GPU is detected
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the following GPUs:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPUs found. TensorFlow will use the CPU.")

# Create a tensor and perform an operation
with tf.device('/GPU:0'):  # Specify GPU device (adjust if using multiple GPUs)
    x = tf.random.normal([1000, 1000])
    y = tf.matmul(x, x)

print("Operation performed on:", y.device)  # It should show /GPU:0 if it's using GPU

