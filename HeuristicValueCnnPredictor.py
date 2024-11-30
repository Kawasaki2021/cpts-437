import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.utils import to_categorical

##### Create CNN model for face recognition #####
def Create_CNN_Model(input_shape):
    face_input = Input(shape=input_shape)

    # First convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(face_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second convolutional layer
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Third convolutional layer
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    face_output = Dense(4, activation='softmax')(x)

    # Create and compile the model
    model = Model(inputs=face_input, outputs=face_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

##### Train model #####
def train_model(model, X_train, y_train):
    # Train the model
    trained_model = model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    return trained_model
