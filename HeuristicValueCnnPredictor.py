import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
import FrontilizationAlgorithm

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

def load_train_data():
    x_data = []
    y_data = []

    for image in os.listdir("./moderator"):
        face_img = FrontilizationAlgorithm.find_and_frontalize(os.getcwd() + "\moderator\\" + image)
        if face_img != []:
            x_data.append(face_img[0])
            y_data.append(2)

    for image in os.listdir("./harris"):
        face_img = FrontilizationAlgorithm.find_and_frontalize(os.getcwd() + "\harris\\" + image)
        if face_img != []:
            x_data.append(face_img[0])
            y_data.append(0)

    for image in os.listdir("./trump"):
        face_img = FrontilizationAlgorithm.find_and_frontalize(os.getcwd() + "\\trump\\" + image)
        if face_img != []:
            x_data.append(face_img[0])
            y_data.append(1)

    return x_data, y_data

x_data, y_data = load_train_data()

x_data = np.array(x_data)
y_data = np.array(y_data)

model = Create_CNN_Model((128, 128, 3))

trained_model = train_model(model, x_data, y_data)

front_img = FrontilizationAlgorithm.find_and_frontalize("frame_0050.jpg")

print(heuristic_predictor(trained_model, front_img))