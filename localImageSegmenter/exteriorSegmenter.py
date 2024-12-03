#Trainer.py to load model
#model_utils.load_model(model_paths[0])

import model_utils
import im_utils
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import pandas as pd
from PIL.ExifTags import TAGS
from PIL import Image
import tifffile
from osgeo import gdal, gdalconst 


"""
    Author: Elliott Hendrickson
    Inpue: N/A
    Output: Batch Size As Rootpainter Would Accept
"""
def findBS():

    mem_per_item = 3800000000

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

"""
    Author: Elliott Hendrickson
    Input: Image Object, Base Image Path, Outpath
    Output: Printed Image To Outpath With The Input Image Path's Metadata
"""
def transferTIFFMetaData(img, ImagePath, OutPath):

    outputPath = os.path.join(OutPath, str(os.path.split(ImagePath)[-1]))

    printBWAsTiff(img, outputPath)

    originalDataset = gdal.Open(ImagePath, gdal.GA_Update)
    segmentedDatset = gdal.Open(outputPath, gdal.GA_Update)

    segmentedDatset.SetMetadata(originalDataset.GetMetadata())
    segmentedDatset.SetProjection(originalDataset.GetProjection())
    segmentedDatset.SetGeoTransform(originalDataset.GetGeoTransform())

    segmentedDatset.FlushCache()

    segmentedDatset = None
    originalDataset = None

"""
    Author: Elliott Hendrickson
    Input: Image Object, Base Image Path, Outpath
    Output: Printed Image To Outpath With The Input Image Path's Metadata
"""
def transferMetaData(img, ImagePath, OutPath):

    OutPath += str(os.path.split(ImagePath)[-1])

    originalImage = Image.open(ImagePath)
    exif_data = originalImage.getexif()
    if exif_data:
        exif_bytes = originalImage.info.get('exif')
    else:
        exif_bytes = None

    img.save(OutPath, exif=exif_bytes)

"""
    Author: Elliott Hendrickson
    Input: uNetGres Model, input Image path, Outpath(directory)
    Output: N/A (prints to)
"""
def segmentAndPrint(model, inputPath, outputPath):

    image = im_utils.load_image(inputPath)

    in_w = 572
    out_w = in_w - 72
    bs = findBS()

    outputImage = model_utils.unet_segment(model, image, bs, in_w, out_w, 0.5)

    correctlyColoredImage = outputImage * 255

    fileType = os.path.splitext(inputPath)[-1]

    if(fileType == "tif" or fileType = "tiff"):

        transferTIFFMetaData(correctlyColoredImage, inputPath, outputPath)
    
    else:

        transferMetaData(correctlyColoredImage, inputPath, outputPath)


loadedModel = model_utils.load_model("modelToLoad/000004_1725063471.pkl")
segmentAndPrint(loadedModel, "imageToSegmentBasic/NJ_Essex County_50700_NUTL.tif", "segmentedImageWithMetaData")
