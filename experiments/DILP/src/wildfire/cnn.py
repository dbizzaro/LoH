"""TensorFlow CNN for wildfire visual predicates."""

import tensorflow as tf


class SimpleCNN(tf.keras.Model):
    def __init__(self, n_output=2, dropout=0.2):
        super(SimpleCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(16, 5, padding='same', activation='relu')
        self.pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(16, 5, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(32, activation='relu')
        self.fc2 = tf.keras.layers.Dense(n_output, activation='sigmoid')

    def call(self, x, training=False):
        x = self.pool(self.conv1(x))
        x = self.dropout(x, training=training)
        x = self.pool(self.conv2(x))
        x = self.dropout(x, training=training)
        x = self.pool(self.conv3(x))
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
