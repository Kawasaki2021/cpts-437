# from tkinter import *
# from tkinter import filedialog
import os
import model_utils
import im_utils
import torch
import numpy as np
import pandas as pd
from PIL.ExifTags import TAGS
from PIL import Image
# from osgeo import gdal, gdalconst 


# """
#         Global GUI Variables
# """

# global  window
# window = Tk()

# """
#         Global Path Variables
# """

# global modelPath
# global imgPath
# global outpath
# modelPath = ""
# imgPath = ""
# outpath = ""

"""
        Segmentation Code
"""

"""
    Author: Elliott Hendrickson
    Inpue: N/A
    Output: Batch Size As Rootpainter Would Accept
"""
def findBS():

    mem_per_item = 3800000000
    total_mem = 0

    if torch.cuda.is_available():

        for i in range(torch.cuda.device_count()):

            total_mem += torch.cuda.get_device_properties(i).total_memory

        bs = total_mem // mem_per_item
        bs = min(12, bs)

    else:
        bs = 1 # cpu is batch size of 1

    return bs

"""
    Author: Elliott Hendrickosn
    Input: Numpy black and white segmented image of some size > 0, the print path
    Output: Prints that array as a tiff file (no metadata yet)
"""
def printBWAsTiff(bwImg, printpath):

    image = Image.fromarray(np.uint8(bwImg)).convert('RGB')

    image.save(printpath)

# """
#     Author: Elliott Hendrickson
#     Input: Image Object, Base Image Path, Outpath
#     Output: Printed Image To Outpath With The Input Image Path's Metadata
# """
# def transferTIFFMetaData(img, ImagePath, OutPath):

#     outputPath = os.path.join(OutPath, str(os.path.split(ImagePath)[-1]))

#     printBWAsTiff(img, outputPath)

#     originalDataset = gdal.Open(ImagePath, gdal.GA_Update)
#     segmentedDatset = gdal.Open(outputPath, gdal.GA_Update)

#     segmentedDatset.SetMetadata(originalDataset.GetMetadata())
#     segmentedDatset.SetProjection(originalDataset.GetProjection())
#     segmentedDatset.SetGeoTransform(originalDataset.GetGeoTransform())

#     segmentedDatset.FlushCache()

#     segmentedDatset = None
#     originalDataset = None

#     print("Finished Printing File To: " + outputPath)

"""
    Author: Elliott Hendrickson
    Input: Image Object, Base Image Path, Outpath
    Output: Printed Image To Outpath With The Input Image Path's Metadata
"""
def transferMetaData(img, ImagePath, OutPath):

    originalImage = Image.open(ImagePath)
    exif_data = originalImage.getexif()

    if exif_data:

        exif_bytes = originalImage.info.get('exif')

        outputImage = Image.fromarray(img)

        if outputImage.mode != 'RGB':
            outputImage = outputImage.convert('RGB')

        outputImage.save(OutPath, exif=exif_bytes)
        
    else:
        outputImage = Image.fromarray(img)

        if outputImage.mode != 'RGB':
            outputImage = outputImage.convert('RGB')

        outputImage.save(OutPath)
    
    # print("Finished Printing File To: " + OutPath)

"""
    Author: Elliott Hendrickson
    Input: uNetGres Model, input Image path, Outpath(directory)
    Output: N/A (prints to)
"""
def segmentAndPrint(model, inputPath, outputPath):

    image = im_utils.load_image(inputPath)

    # print("shello")

    in_w = 572
    out_w = in_w - 72
    bs = findBS()

    outputImage = model_utils.unet_segment(model, image, bs, in_w, out_w, 0.5)

    correctlyColoredImage = outputImage * 255

    fileType = os.path.splitext(inputPath)[-1].lower()

    if(fileType == ".tif" or fileType == ".tiff"):

        print("Not this project")
        # transferTIFFMetaData(correctlyColoredImage, inputPath, outputPath)
    
    else:

        transferMetaData(correctlyColoredImage, inputPath, outputPath)

# """
#         GUI Code
# """
# def browseForModel():

#     global modelPath
#     modelPath = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select a File", filetypes = [("Pickle Files", "*.pkl")])
#     label_file_explorer.configure(text="Model Selected: "+ modelPath)
#     button_model = Button(window, text = "Selected Model: " + modelPath, command = browseForModel) 
#     button_model.grid(column = 1, row = 2)

# """
#     Author: Elliott Hendrickson
#     Input: User Input selects folder or image to segment
#     Output: updating the path to the input path
# """
# def browseForImg():

#     global imgPath

#     if (inputTypeString.get() == "Singular Image"):

#         imgPath = filedialog.askopenfilename()

#         if (isImage(imgPath)):

#             label_file_explorer.configure(text = "Image selected: " + os.path.split(imgPath)[-1])

#         else:

#             label_file_explorer.configure(text = "Unsupported Image Type Selected")

#     else:

#         imgPath = filedialog.askdirectory()

#         if imgPath:

#             label_file_explorer.configure(text = "Directory selected: " + os.path.split(imgPath)[-1])

"""
    Author: Elliott Hendrickson
    Input: Image path
    Output: boolean value true if it is a valid image path, false if otherwise
"""
def isImage(imgPath):

    try:
        with Image.open(imgPath) as img:
            img.verify()
            return True
        
    except (IOError, SyntaxError):

        return False

# """
#     Author: Elliott Hendrickson
#     Input: User input 
#     Output: Updates global variables for what the output folder is
# """
# def browseForOutputFolder():

#     global outpath
#     outpath = filedialog.askdirectory(initialdir = os.getcwd(), title = "Select a Ouput Folder")
#     label_file_explorer.configure(text="Output Folder Selected: "+ outpath)

# """
#     Author: Elliott Hendrickson
#     Input: Global variables being filled out
#     Output: Segments based on input global variables
# """
# def imageSegmentation():

#     if(len(modelPath) != 0 and len(imgPath) != 0 and len(outpath) != 0):

#         label_file_explorer.configure(text = "Segmentation Finished")

#         loadedModel = model_utils.load_model(modelPath)

#         if(os.path.isdir(imgPath)):

#             for image in os.listdir(imgPath):

#                 image = os.path.join(imgPath, image)

#                 if(isImage(image)):

#                     print("Started Segmenting Image: " + str(image))
#                     segmentAndPrint(loadedModel, image, outpath)

#         else:

#             label_file_explorer.configure(text = "Segmentation Finished")

#             if(isImage(imgPath)):
                
#                 segmentAndPrint(loadedModel, imgPath, outpath)

#     else:

#         outputString = ""

#         if(len(modelPath) == 0):
#             outputString += "Please Specify A Rootpainter Model"

#         if(len(imgPath) == 0):

#             if(len(outputString) != 0):
#                 outputString += "; "

#             outputString += "Please Specify A Image Path Or Image Directory"

#         if(len(outpath) == 0):

#             if(len(outputString) != 0):
#                 outputString += "; "

#             outputString += "Please Specify An Output Directory"

#         label_file_explorer.configure(text=outputString)

# """
#     Author: Elliott Hendrickson
#     Purpose: allows the image selector button to be pressed after called
# """
# def enableFileSelection(*args):

#     button_image.config(state = NORMAL)

# """
#     Author: Elliott Hendrickson
#     Purpose: User selects whether to segment a folder or singular image
# """
# def dropDownImageType(window, locationTuple):

#     dropDownOptions = ["Singular Image", "Image Directory"]

#     global inputTypeString
#     inputTypeString = StringVar(window) 
#     inputTypeString.set("Select Input Image Type")
#     dropDownButton = OptionMenu(window, inputTypeString, *dropDownOptions)
#     dropDownButton.pack()
#     dropDownButton.grid(column = locationTuple[0], row = locationTuple[1])
#     inputTypeString.trace_add("write", enableFileSelection)

# """
#     Author: Elliott Hendrickson
#     Purpose: This is the GUI
# """
# def localSegmenterGUI(): 

#     window.title('Local Rootpainter Segmenter GUI')
#     window.maxsize()
#     window.config(background = "white")

#     global label_file_explorer
#     label_file_explorer = Label(window, text = "Local Rootpainter Segmenter", width = 200, height = 4, fg = "blue")
#     button_model = Button(window, text = "Select Model", command = browseForModel)
#     global button_image
#     button_image = Button(window, text = "Select Image Or Image Directory", command = browseForImg)
#     button_image["state"] = "disabled"
#     dropDownImageType(window, (1, 3)) #This line must be behind button_image defintion
#     button_outpath = Button(window, text = "Select Destination Directory", command = browseForOutputFolder)
#     button_segmentation = Button(window, text = "Segment", command = imageSegmentation)
#     button_exit = Button(window, text = "End Software", command = exit) 
    
#     label_file_explorer.grid(column = 1, row = 1)
#     button_model.grid(column = 1, row = 2)
#     button_image.grid(column = 1, row = 4)
#     button_outpath.grid(column = 1, row = 5)
#     button_segmentation.grid(column = 1, row = 6)
#     button_exit.grid(column = 1,row = 7)
    
#     window.mainloop()

# def iterateThroughAllFramesMakingMasks(modelPath, inputFolder, outputFolder):

#     loadedModel = model_utils.load_model(modelPath)

#     for image in os.listdir(inputFolder):

#         image = os.path.join(inputFolder, image)
#         outputPath = os.path.join(outputFolder, os.path.split(image)[-1])
#         segmentAndPrint(loadedModel, image, outputPath)

iterateThroughAllFramesMakingMasks(os.path.join("ModelFolder", "000016_1733109385.pkl"),"../frames/", "segmentationFolder/")