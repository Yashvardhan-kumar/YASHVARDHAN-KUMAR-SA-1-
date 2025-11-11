# src/train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# Paths
train_dir = "../data/train"
test_dir = "../data/test"

IMG_SIZE = (224, 224)
BATCH = 16
EPOCHS = 10

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
).flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH, class_mode='categorical')

base = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
out = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base.input, outputs=out)

for layer in base.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Fine-tune
for layer in base.layers[-20:]:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS//2)

os.makedirs("../models", exist_ok=True)
model.save("../models/drowsiness_model.h5")
print("✅ Model saved to models/drowsiness_model.h5")
