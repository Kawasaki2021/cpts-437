import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.utils import to_categorical

##### Create CNN model for face recognition #####
def create_cnn_model(input_shape):
    # Input layer for the face image with the given input shape
    face_input = Input(shape=input_shape)

    # First convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(face_input)

    # Max-pooling layer
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flatten the output of the pooling layer
    x = Flatten()(x)

    # Fully connected layer
    x = Dense(128, activation='relu')(x)

    # Output layer
    # Output: Kamala, Trump, Mediator, Unknown
    face_output = Dense(4, activation='softmax')(x)

    # Create and return model with input layer and output layer
    return Model(inputs=face_input, outputs=face_output)
