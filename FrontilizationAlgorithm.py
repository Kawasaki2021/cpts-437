import face_recognition
from PIL import Image

import numpy as np 

import torch
from torchvision import transforms
from torch.autograd import Variable
import network

saved_model = torch.load("./generator_v0.pt", map_location=torch.device('cpu'))

def find_faces(image_path):
    faces = []

    # open the images and find the faces
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    # subsect the images fromt the locations and return the array of Image objects
    for location in face_locations:
        top, right, bottom, left = location

        face_image = image[top:bottom, left:right]

        pil_image = Image.fromarray(face_image)

        faces.append(pil_image)

    return faces

# frontalize a PIL image
# sourced from https://huggingface.co/opetrova/face-frontalization
def frontalize(image):
    # convert to [1, 3, 128, 128] tensor
    preprocess = transforms.Compose((transforms.Resize(size = (128, 128)), transforms.ToTensor()))

    input_tensor = torch.unsqueeze(preprocess(image), 0)

    generated_image = saved_model(Variable(input_tensor.type('torch.FloatTensor')))
    generated_image = generated_image.detach().squeeze().permute(1, 2, 0).numpy()
    generated_image = (generated_image + 1.0) / 2.0
   
    return generated_image

# combine both into one function
def find_and_frontalize(image_path):
    frontalized_faces = []

    faces = find_faces(image_path)

    if faces != []:
        for face in faces:
            front_face = frontalize(face)
            frontalized_faces.append(front_face)

    return frontalized_faces