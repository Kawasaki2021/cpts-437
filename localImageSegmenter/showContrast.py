import cv2
import os
import numpy as np
import copy
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes
from torchvision.io import read_image
from torchvision import transforms
import torch
import torchvision.transforms.functional as F

"""
    Author: Elliott Hendrickson
    Inpue: The base image folder, the folder of segmented files, and the output folder path
    Output: Images in the output folder, which show the contrast of the segmentation
"""
def showContrast(imageFolder, segmentationFolder, outpath):
    listOfImages = os.listdir(imageFolder)
    segmentationImages = os.listdir(segmentationFolder)
    listOfImagesNamaes = [os.path.split(filepath)[-1] for filepath in listOfImages]
    listOfSegmentationNames = [os.path.split(filepath)[-1] for filepath in segmentationImages]
    listOfImageToMask = [name for name in listOfImagesNamaes if name in listOfSegmentationNames]

    for imageName in listOfImageToMask:

        baseImage = cv2.imread(os.path.join(imageFolder, imageName))
        booleanMask = cv2.imread(os.path.join(segmentationFolder, imageName), cv2.IMREAD_GRAYSCALE)

        originalImage = copy.deepcopy(baseImage)
        greyScaleImage = copy.deepcopy(booleanMask)

        booleanMask = booleanMask > 0
        baseImage[booleanMask] = [0,0,255]

        print(str(os.path.join(outpath, imageName)))

        greyScaleImage = cv2.cvtColor(greyScaleImage, cv2.COLOR_GRAY2BGR)

        holder = np.concatenate((originalImage,baseImage),axis=0)
        holder = np.concatenate((holder,greyScaleImage),axis=0)

        cv2.imwrite(str(os.path.join(outpath, str(os.path.splitext(imageName)[0])+".png")), holder)

"""
    Author: Elliott Hendrickson
    Inpue: A segmentation mask of unkown size
    Output: A list of matrices that contains every segmentation object found in the mask
    Algorithim: Uses depth first search to find all non zero location
"""

def findObjects(boolean_mask):
    # Ensure the mask is binary
    proccessingFilter = np.ones((5,5),np.uint8)
    bwImage = cv2.dilate(cv2.erode(boolean_mask,proccessingFilter,iterations = 1), proccessingFilter,iterations = 1)
    bwImage = (bwImage > 0).astype(np.uint8)

    # Label connected components
    labeled_array, num_features = label(bwImage)

    # Create a list to store the coordinates of each object
    objects = []
    
    for feature in range(1, num_features + 1):
        # Find the coordinates of each object
        object_coords = np.argwhere(labeled_array == feature)
        objects.append(object_coords)

    numberOfPositivePixels = sum([len(item) for item in objects])
    h, w = bwImage.shape
    numberOfTotalPixels = h * w
    numberOfNegatiePixels = numberOfTotalPixels - numberOfPositivePixels

    print("Number Of Positive Pixels: "+ str(numberOfPositivePixels))
    print("Number Of Negative Pixels: "+ str(numberOfNegatiePixels))
    print("Total Number Of Objects Found: " + str(len(objects)))
    
    return objects

# def findObjects(booleanMask):

#     #Preprocessing
#     proccessingFilter = np.ones((5,5),np.uint8)
#     preProcessedBooleanMask = cv2.dilate(cv2.erode(booleanMask,proccessingFilter,iterations = 1), proccessingFilter,iterations = 1)

#     preProcessedBooleanMask[preProcessedBooleanMask != 0] = 255

#     searchedPositivePixels = set()
#     searchedNegativePixels = set()
#     unsearchedPixels = set()
#     localSearchedPixels = set()

#     h, w = preProcessedBooleanMask.shape
#     totalNumberOfPixels = w * h

#     for height in range(h):

#         for width in range(w):

#             unsearchedPixels.add((width,height))

#     h = h - 1
#     w = w - 1

#     print("Entering Searching Function")

#     listOfSegmented = []

#     while len(unsearchedPixels) > 0:

#         currentPixel = unsearchedPixels.pop()

#         if(preProcessedBooleanMask[currentPixel[1],currentPixel[0]] != 0):

#             localObject = []

#             #Out Of Unsearched potentially positive sections, but just found one
#             searchedPositivePixels.add(currentPixel)
#             unsearchedPixels.discard(currentPixel)
#             setOfNeightboringPixels = returningNeigbors(currentPixel, w, h)
#             #print("The Number Of Neighbors Found at pixel: " + str(currentPixel) + " " + str(len(setOfNeightboringPixels)) + " Value: " +str(preProcessedBooleanMask[currentPixel[1],currentPixel[0]]))
#             setOfNeightboringPixels = ensureNeighborsArentAlreadySearched(unsearchedPixels, setOfNeightboringPixels)

#             while(len(setOfNeightboringPixels) != 0):

#                 currentPixel = setOfNeightboringPixels.pop()

#                 if(preProcessedBooleanMask[currentPixel[1],currentPixel[0]] != 0):

#                     searchedPositivePixels.add(currentPixel)
#                     unsearchedPixels.discard(currentPixel)

#                     localObject.append(currentPixel)

#                     setOfNeightboringPixels.update(returningNeigbors(currentPixel, w, h))
#                     setOfNeightboringPixels = ensureNeighborsArentAlreadySearched(unsearchedPixels, setOfNeightboringPixels)
#                     print("Set of neighbors after intersection: " + str(len(setOfNeightboringPixels)))

#                 else:

#                     searchedNegativePixels.add(currentPixel)
#                     unsearchedPixels.discard(currentPixel)

#             listOfSegmented.append(localObject)

#         else:

#             searchedNegativePixels.add(currentPixel)
#             unsearchedPixels.discard(currentPixel)

#     print("Number Of Positive Pixels: "+ str(len(searchedPositivePixels)))
#     print("Number Of Negative Pixels: "+ str(len(searchedNegativePixels)))
#     print("Total Unsearched Pixels: " + str(len(unsearchedPixels)))
#     print("Difference between total pixels and found pixels: " + str(totalNumberOfPixels-(len(searchedPositivePixels)+len(searchedNegativePixels))))
#     print("Total Number Of Objects Found: " + str(len(listOfSegmented)))

#     return listOfSegmented
"""
    Author: Elliott Hendrickson
    Input: The set of all unsearched pixels in the image, the current pixels about to be searched
    Output: The terms in the about to be searched pixels that havent been
"""
def ensureNeighborsArentAlreadySearched(unsearchedPixels, setOfNeightboringPixels):

    return unsearchedPixels.intersection(setOfNeightboringPixels)

"""
    Author: Elliott Hendrickson
    Input: The current pixel (tuple), width of input image, height of input image
    Output: All eight neighbors of the current pixel
"""
def returningNeigbors(currentPixel, w, h):

    allEightNeigbors = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    outputSet = set()

    for neighbor in allEightNeigbors:

        neighborPixel = (currentPixel[1]+neighbor[1],currentPixel[0]+neighbor[0])

        if checkWithinBounds(neighborPixel, w, h):

            outputSet.add((currentPixel[1]+neighbor[1],currentPixel[0]+neighbor[0]))

    return outputSet

"""
    Author: Elliott Hendrickson
    Input: The current pixel (tuple), width and height of input boolean image
    Output: Boolean value if input pixel tuple is within bounds
"""
def checkWithinBounds(currentPixel, w, h):

    if(currentPixel[0] >= w or currentPixel[1]>= h):
    
        return False

    else:

        return True

"""
    Author: Elliott Hendrickson
    Input: The path to the image that has been segmented
    Output: Returns the boolean mask
"""
def readBooleanMask(pathName):
    
    return cv2.imread(pathName, cv2.IMREAD_GRAYSCALE)

"""
    Author: Elliott Hendrickson
    Input: Boolean Mask
    Output: Shape objects in blackwhite with all segmented objects
"""
def blobDetectorMethod(greyImage):

    proccessingFilter = np.ones((5,5),np.uint8)
    greyImage = cv2.dilate(cv2.erode(greyImage,proccessingFilter,iterations = 3), proccessingFilter,iterations = 4)

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 15
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(greyImage)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    img_with_keypoints = cv2.drawKeypoints(greyImage, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Show keypoints
    cv2.imwrite("imageWithDection.png", img_with_keypoints)

"""
    Author: Elliott Hendrickson
    Input: Folder Path
    Output: List of all files inside of the folder
"""
def readFiles(folderPath):

    files_list = []

    for root, dirs, files in os.walk(folderPath):

        for file in files:

            # Create the full file path
            full_path = os.path.join(root, file)
            if "json" not in full_path:
                files_list.append(full_path)

    return files_list

"""
    Author: Elliott Hendrickson
    Input: List of input object masks, as well as the outputpath
    Output: Printing the histogram to a file
"""
def createHistogram(listOfObjects, outputPath):

    x = [len(item) for item in listOfObjects]
    plt.hist(x, bins=30, color='skyblue', edgecolor='black')
    
    # Adding labels and title
    plt.xlabel('Size Of Objects Found')
    plt.ylabel('Number Of Objects Found At Size')
    plt.title("Total Number Of Objects Detected: " + str(len(x)))

    plt.savefig(outputPath) 
    plt.close()

"""
    Author: Elliott Hendrickson
    Input: Input folder filed with images outputFolder empty but exists
    Output: Histogram for all files inside of the input folder
"""
def imageClassifierPreparation(inputFolder,outputFolder):

    for file in readFiles(inputFolder):

        createHistogram(findObjects(readBooleanMask(file)), os.path.join(outputFolder, os.path.split(file)[-1]))

"""
    Author: Elliott Hendrickson
    Input: List Of List where each nested list contains a list of size of all found objects
    Ouput: The size of the lower quartile as well as the upper quartile in order to classify as small, medium, large
"""
def findQuartiles(listOfListObjects):

    listOfAllObjects = []

    for listOfObjects in listOfListObjects:

        listOfAllObjects = listOfAllObjects + listOfObjects

    with open("quartiles.txt", "w") as f:

        f.write(str(listOfAllObjects))


"""
    Author: Elliott Hendrickson
    Input: Pathname to segmented images
    Output: Non Memory Intense list of list of object sizes found in each segmentation
"""
def makeScrubbedFileObjectsLists(pathName):

    outputList = []

    for file in readFiles(pathName):

        outputList.append([len(item) for item in findObjects(readBooleanMask(file))])

    return outputList

def split_into_bins(data, bin_size=1000):
    return [data[i:i + bin_size] for i in range(0, len(data), bin_size)]

"""
    Author: Elliott Hendrickson
    Input: List of ints where each int object is a segmented object
    Output: Bundled Int numbers into 10 bundles
"""
def counterFunction(listOfObjectSizes):
    
    maxSize = max(listOfObjectSizes)
    sortedBinsInput = split_into_bins(sorted(listOfObjectSizes))
    
    x = []
    y = []

    for i, bin in enumerate(sortedBinsInput):

        x.append(i)
        y.append(math.log2(np.mean(bin)))
        print("Mean For Bin: " + str(i) + " Mean: " + str(np.mean(bin)) + " Number Of Terms In Bin: " + str(len(bin)))

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color='blue', s=100, marker='o')  # 's' is the marker size

    # Add labels and title
    plt.xlabel('X Values')
    plt.ylabel('Y Values')
    plt.title('Sample Scatter Plot')

    # Display the plot
    plt.show()

def createBigHistogram():

    with open("quartiles.txt", "r") as f:

        line = f.read().splitlines()[0]

    line = line.replace("[", "")
    line = line.replace("]", "")
    bigList = line.split(", ")

    bigList = [int(item) for item in bigList]

    # print(len(bigList))

    counterFunction(bigList)
    # fig, ax = plt.subplots()
    # plt.hist(bigList, bins=15, color='skyblue', edgecolor='black')
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
    # # Adding labels and title
    # plt.xlabel('Size Of Objects Found')
    # plt.ylabel('Number Of Objects Found At Size')
    # plt.title("Total Number Of Objects Detected: " + str(len(bigList)))

    # plt.savefig("bigHistogram.png") 
    # plt.close()


# def imageClassifier(listOfObjects):

#     upperQuartileOfBurrowArea = 0



# showContrast("GSI-PV-Burrows","segmentedImageWithMetaData/","pngContrastFolder/")
# imageClassifierPreparation("segmentedImageWithMetaData/", "HistogramsOfDeepSeaSegmentations/")
# findQuartiles(makeScrubbedFileObjectsLists("segmentedImageWithMetaData/"))
# createBigHistogram()

def is_inside(bbox1, bbox2):

    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return (x2 <= x1 <= x2 + w2) and (y2 <= y1 <= y2 + h2)

def readAllImagesThenMakeBoundingBoxes(directoryPathOfSegmentations, directoryPathOfImages, outputDirectoryPath):

    listOfSegmentationFiles = readFiles(directoryPathOfSegmentations)

    proccessingFilter = np.ones((5,5),np.uint8)
    listOfImages = []

    print(len(listOfSegmentationFiles))

    for i in range(len(listOfSegmentationFiles)):

        print(str(os.path.split(listOfSegmentationFiles[i])[-1]))
        baseImg = cv2.imread(os.path.join(directoryPathOfImages, str(os.path.split(listOfSegmentationFiles[i])[-1])))
        baseBooleanMask = cv2.imread(os.path.join(directoryPathOfSegmentations, os.path.split(listOfSegmentationFiles[i])[-1]), cv2.IMREAD_GRAYSCALE)

        preProcessBooleanMask = cv2.dilate(baseBooleanMask,proccessingFilter,iterations = 5)
        # preProcessBooleanMask = cv2.dilate(cv2.erode(baseBooleanMask,proccessingFilter,iterations = 5), proccessingFilter,iterations = 5)
        # preProcessBooleanMask[preProcessBooleanMask != 0] = 255
        contours, _ = cv2.findContours(preProcessBooleanMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boundingBoxes = [cv2.boundingRect(contour) for contour in contours]

        validBoxes = []

        for bbox1 in  boundingBoxes:

            isNested = False

            for  bbox2 in boundingBoxes:

                if bbox1 != bbox2 and is_inside(bbox1, bbox2):

                    isNested = True
                    break

            if not isNested:

                validBoxes.append(bbox1)
        
        for bbox in validBoxes:
            x, y, w, h = bbox
            cv2.rectangle(baseImg, (x, y), (x + w, y + h), (0, 0, 255), 2)

        outputPath = os.path.join(outputDirectoryPath, os.path.split(listOfSegmentationFiles[i])[-1])
        cv2.imwrite(outputPath, baseImg)


readAllImagesThenMakeBoundingBoxes("segmentedImageWithMetaData", "GSI-PV-Burrows", "boundingImagesDilation")