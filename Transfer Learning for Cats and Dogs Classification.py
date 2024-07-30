# Step 1: Import libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

training_dir = 'dataset/training_set'
test_dir = 'dataset/test_set'

# Step 3: Building the model
img_shape = (128,128,3)

# Loading the Pre-Trained Model (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet')

# Print the summary of the base model
base_model.summary()

# Freeze the base model to prevent its weights from being updated during training
base_model.trainable = False

# Add a Global Average Pooling layer on top of the base model
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

# Add a Dense layer with a single unit and sigmoid activation for binary classification
prediction = tf.keras.layers.Dense(units=1, activation='sigmoid')(global_average_layer)

# Define the transfer learning model
# Create the final model by specifying the input and output tensors
model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction)

# Print the summary of the final model
model.summary()

# compile the model
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Create Data Generators
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_gen_train= ImageDataGenerator(rescale=1/255.0)
data_gen_test = ImageDataGenerator(rescale=1/255.0) 

train_generator= data_gen_train.flow_from_directory(directory=training_dir,
                                                        target_size=(128,128), batch_size=128,class_mode='binary')
test_generator= data_gen_test.flow_from_directory(directory=test_dir,  
                                                  target_size=(128,128), batch_size=128, class_mode='binary')


# Step 4: Training the model
model.fit(train_generator, epochs=5, validation_data=test_generator)

#Fine Tuning
# Set the base model layers to trainable
base_model.trainable = True
# Print the number of layers in the base model
print(f"Number of layers in the base model: {len(base_model.layers)}")

# Step 5: Fine Tuning

# Freeze all the layers before the fine_tune_at layer
fine_tune_at = 100

# Freeze all the layers before the fine_tune_at layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Create the optimizer instance (must be done before compiling the model)
opt = tf.keras.optimizers.Adam()

# Compile the model with the new optimizer
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model (use model.fit instead of the deprecated model.fit_generator)
model.fit(train_generator, epochs=5, validation_data=test_generator)
