"""
Feature extraction transfer learning 
on 1% of the data with data augmentation
"""

# Modules
from helper_functions import create_tensorboard_callback, plot_loss_curves, walk_through_dir
import tensorflow as tf

# Walk through 10 percent data directory and list number of files
walk_through_dir("C:/VS CODE/Deep Learning/Les 02/10_food_classes_10_percent") # you can print if you want :)

# Create training and test dirs
train_dir_1_percent = "C:/VS CODE/Deep Learning/Les 02/10_food_classes_1_percent/train"
test_dir = "C:/VS CODE/Deep Learning/Les 02/10_food_classes_1_percent/test"

IMG_SIZE =(224, 224)
train_data_1_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir_1_percent,
                                                                           label_mode='categorical',
                                                                           batch_size=32,
                                                                           image_size=IMG_SIZE)
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode='categorical',
                                                                image_size=IMG_SIZE)

from tensorflow import keras
from tensorflow.keras import layers # type: ignore

# NEW: Newer versions of TensorFlow (2.10+) can use the tensorflow.keras.layers API directly for data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomHeight(0.2),
    layers.RandomWidth(0.2)
], name='data_augmentation')

# Creating Model
input_shape = (224, 224, 3)
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
base_model.trainable = False
inputs = layers.Input(shape=input_shape, name="input_layer")
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)
model_1 = keras.Model(inputs, outputs)
model_1.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
history_1_percent = model_1.fit(train_data_1_percent,
                    epochs=5,
                    steps_per_epoch=len(train_data_1_percent),
                    validation_data=test_data,
                    validation_steps=int(0.25* len(test_data)), # validate for less steps
                    # Track model training logs
                    callbacks=[create_tensorboard_callback("transfer_learning", "1_percent_data_aug")])

# Check out model summary
sum_model = model_1.summary()
print(f"Summary: {sum_model}")

# Evaluate on the test data
results_1_percent_data_aug = model_1.evaluate(test_data)
print(f"Results: {results_1_percent_data_aug}")

# How does the model go with a data augmentation layer with 1% of data
plot_loss_curves(history_1_percent)