#!/usr/bin/env python3
"""It is my first ML model with transfer learning!"""
import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(X, Y):
    """Preprocess Data"""
    X = X.astype('float32')
    X_p = preprocess_input(X)
    Y_p = tf.keras.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p

if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Lambda(lambda img: tf.image.resize(img, (224, 224)))(inputs)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)
    ]

    model.fit(datagen.flow(X_train, Y_train, batch_size=64),
              validation_data=(X_test, Y_test), epochs=20, callbacks=callbacks)

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])

    model.fit(datagen.flow(X_train, Y_train, batch_size=64),
              validation_data=(X_test, Y_test), epochs=10, callbacks=callbacks)

    model.save('cifar10.h5')
