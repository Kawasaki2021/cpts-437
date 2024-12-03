import numpy as np
import os
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import FrontilizationAlgorithm
import pickle
from collections import Counter

##### Create CNN model for face recognition #####
def Create_CNN_Model(input_shape):
    face_input = Input(shape=input_shape)

    # First convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(face_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # Second convolutional layer
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # Third convolutional layer
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    face_output = Dense(4, activation='softmax')(x)

    # Create and compile the model
    model = Model(inputs=face_input, outputs=face_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Return CNN model
    return model

##### Train model #####
def train_model(model, X_train, y_train):
    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    return model

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

def save_model_with_pickle(model, filepath):
    # Save the architecture and weights of the model
    model_json = model.to_json()  # Get the architecture as a JSON string
    model_weights = model.get_weights()  # Get the weights of the model
    
    # Save the model structure and weights to a file using pickle
    with open(filepath, 'wb') as f:
        pickle.dump({'model_json': model_json, 'model_weights': model_weights}, f)

    print("Model saved successfully with Pickle!")

def load_model_with_pickle(filepath):
    # Load the model structure and weights from the pickle file
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    # Reconstruct the model from the JSON structure
    model_json = model_data['model_json']
    model = model_from_json(model_json)  # Create model from JSON
    
    # Set the weights of the model
    model.set_weights(model_data['model_weights'])
    
    print("Model loaded successfully with Pickle!")
    
    return model

"""x_data, y_data = load_train_data()

x_data = np.array(x_data)
y_data = np.array(y_data)

model = Create_CNN_Model((128, 128, 3))

trained_model = train_model(model, x_data, y_data)

save_model_with_pickle(trained_model, 'model_pickle.pkl')"""

model = load_model_with_pickle('model_pickle.pkl')

results = []

for image in os.listdir("./frames"):
        face_img = FrontilizationAlgorithm.find_and_frontalize(os.getcwd() + "\\frames\\" + image)
        for img in face_img:
            results.append(heuristic_predictor(model, img))

count = Counter(results)

file = open("Output.txt", "w")

for item in count.items():
    string = "Item: " + str(item[0]) + " Frequency: " + str(item[1]) + "\n"
    file.write(string)

file.close()