import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

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

    # Return CNN model
    return model

##### Train model #####
def train_model(model, X_train, y_train):
    # Train the model
    trained_model = model.fit(X_train, y_train, epochs=10, batch_size=32)

    return trained_model

##### Heuristic Value CNN Predictor #####
def heuristic_predictor(model, face_image):
    # Intialize speaker_map
    speaker_map = {0: 'Kamala Harris', 1: 'Donald Trump', 2: 'Mediator', 3: 'Unknown'}

    # Process the input face image
    # Add batch dimension
    face_input = np.expand_dims(face_image, axis=0)
    
    # Predict with the CNN model
    prediction = model.predict(face_input)
    
    # Get the predicted class with the highest probability
    predicted_class = np.argmax(prediction)
    
    # Map the predicted class to the corresponding speaker
    predicted_speaker = speaker_map.get(predicted_class, 'Unknown')

    # Return predicted speaker
    return predicted_speaker
