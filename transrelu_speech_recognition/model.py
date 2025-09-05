import tensorflow as tf
from tensorflow.keras import layers, models

class TransReLU(layers.Layer):
    def __init__(self, **kwargs):
        super(TransReLU, self).__init__(**kwargs)

    def call(self, inputs):
        shape = tf.shape(inputs)
        
        num_freq_bins = shape[1]
        low_freq_end_idx = tf.cast(tf.cast(num_freq_bins, tf.float32) * 0.62, tf.int32)
        s = tf.reduce_mean(inputs[:, :low_freq_end_idx, :, :], axis=1, keepdims=True)

        tau = tf.reduce_mean(inputs, axis=1, keepdims=True)
        
        condition = tf.cast(s > tau, dtype=tf.float32)
        alpha = 0.3 + 0.7 * condition
        
        return tf.nn.relu(inputs)

    def get_config(self):
        base_config = super(TransReLU, self).get_config()
        return base_config

def build_model(input_shape, num_classes=10):
    """
    TransReLU를 사용하는 CNN 모델을 생성합니다.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="input"),
        
        # Block 1
        layers.Conv2D(32, 3, padding='same', name="conv1"),
        layers.BatchNormalization(name="bn1"),
        TransReLU(),
        layers.MaxPooling2D(name="pool1"),
        
        # Block 2
        layers.Conv2D(64, 3, padding='same', name="conv2"),
        layers.BatchNormalization(name="bn2"),
        TransReLU(),
        layers.MaxPooling2D(name="pool2"),
        
        # Block 3
        layers.Conv2D(128, 3, padding='same', name="conv3"),
        layers.BatchNormalization(name="bn3"),
        TransReLU(),
        layers.MaxPooling2D(name="pool3"),
        
        # Classifier
        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dense(128, activation='relu', name="dense1"),
        layers.Dropout(0.5, name="dropout"),
        layers.Dense(num_classes, name="output")
    ])
    return model