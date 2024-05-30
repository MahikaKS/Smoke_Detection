from sklearn.metrics import confusion_matrix as conf_matrix
import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications import ResNet50, ResNet101V2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

input_shape = (224, 224, 3)
num_classes = 3
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

train_generator = datagen.flow_from_directory(
    './data/train',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    './data/train',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


#base_model = ResNet50(weights='imagenet', include_top=False,
 #                   input_shape=input_shape)
base_model = ResNet101V2(
    weights='imagenet', include_top=False, input_shape=input_shape)


for layer in base_model.layers:
    layer.trainable = False

model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


print("Train class indices:", train_generator.class_indices)
print("Train class indices:", validation_generator.class_indices)

model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples / batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples
)

model.save('./model.h5')

# Testing
test_generator = datagen.flow_from_directory(
    './data/test',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print("Train class indices:", test_generator.class_indices)
evaluation_result = model.evaluate(
    test_generator,
    steps=test_generator.samples // batch_size
)

# Print the evaluation result (e.g., accuracy and loss)
print("Test Accuracy:", evaluation_result[1])
print("Test Loss:", evaluation_result[0])

# Predict the test data
predictions = model.predict(test_generator, steps=test_generator.samples)

# Confusion matrix, Hit rate, accuract, f1 score, precision, mcc
y_true = test_generator.classes
y_pred = np.argmax(predictions, axis=1)


def calculate_mcc(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp)
                                        * (tp + fn) * (tn + fp) * (tn + fn))
    return mcc


def calculate_fpr(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    fpr = fp / (tn + fp)
    return fpr


def calculate_fnr(confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    fnr = fn / (tp + fn)
    return fnr


confusion_matrix = conf_matrix(y_true, y_pred)
combined_conf_matrix = np.zeros((3, 2), dtype=int)
combined_conf_matrix[:, 0] = np.sum(confusion_matrix[:, :2], axis=1)  # Sum of the first two columns
combined_conf_matrix[:, 1] = np.sum(confusion_matrix[:, 1:], axis=1)  # Sum of the last two columns

# Combine the last two columns to make a 2x2 matrix
confusion_matrix = np.zeros((2, 2), dtype=int)
confusion_matrix[0, 0] = np.sum(combined_conf_matrix[:2, 0]) # Sum of the first column
confusion_matrix[0, 1] = combined_conf_matrix[0, 1]          # Second column
confusion_matrix[1, 0] = np.sum(combined_conf_matrix[1:, 0]) # Sum of the third column
confusion_matrix[1, 1] = combined_conf_matrix[1, 1]          # Fourth column

hit_rate = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1) 
accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
f1_score = 2 * (hit_rate * (1 / (1 - hit_rate)))
precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
mcc = calculate_mcc(confusion_matrix)
falsePositiveRate = calculate_fpr(confusion_matrix)
falseNegativeRate = calculate_fnr(confusion_matrix)

print("Confusion Matrix: \n", confusion_matrix)
print(f"Hit Rate: {hit_rate[0]}")
print(f"Accuracy: {accuracy}")
#print(f"F1 Score: {f1_score[0]}")
print(f"Precision: {precision[0]}")
#print(f"MCC: {mcc}")
print(f"False Positive Rate: {falsePositiveRate}")
print(f"False Negative Rate: {falseNegativeRate}")
