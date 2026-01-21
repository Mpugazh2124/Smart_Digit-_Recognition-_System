import numpy as np
import idx2numpy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

X_train = idx2numpy.convert_from_file('emnist-balanced-train-images-idx3-ubyte')
y_train = idx2numpy.convert_from_file('emnist-balanced-train-labels-idx1-ubyte')
X_test = idx2numpy.convert_from_file('emnist-balanced-test-images-idx3-ubyte')
y_test = idx2numpy.convert_from_file('emnist-balanced-test-labels-idx1-ubyte')

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = np.transpose(X_train, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

num_classes = len(np.unique(y_train))

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)
model.save("emnist_balanced_model.h5")
print("✅ Model trained and saved as emnist_balanced_model.h5")
