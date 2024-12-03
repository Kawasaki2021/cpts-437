from tkinter import *
from tkinter import filedialog
import  os
from PIL import Image

"""
        Global GUI Variables
"""

global  window
window = Tk()

"""
        Global Path Variables
"""

global modelPath
global imgPath
global outpath
modelPath = ""
imgPath = ""
outpath = ""

def browseForModel():

    global modelPath
    modelPath = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select a File", filetypes = [("Pickle Files", "*.pkl")])
    label_file_explorer.configure(text="Model Selected: "+ modelPath)
    button_model = Button(window, text = "Selected Model: " + modelPath, command = browseForModel) 
    button_model.grid(column = 1, row = 2)

def browseForImg():

    global imgPath

    if (inputTypeString.get() == "Singular Image"):

        imgPath = filedialog.askopenfilename()

        if (isImage(imgPath)):

            label_file_explorer.configure(text = "Image selected: " + os.path.split(imgPath)[-1])

        else:

            label_file_explorer.configure(text = "Unsupported Image Type Selected")

    else:

        imgPath = filedialog.askdirectory()

        if imgPath:

            label_file_explorer.configure(text = "Directory selected: " + os.path.split(imgPath)[-1])

def isImage(imgPath):

    try:
        with Image.open(imgPath) as img:

            img.verify()
            return True
        
    except (IOError, SyntaxError):

        return False

def browseForOutputFolder():

    outpath = filedialog.askdirectory(initialdir = os.getcwd(), title = "Select a Ouput Folder")
    label_file_explorer.configure(text="Output Folder Selected: "+ outpath)

def imageSegmentation():

    if(len(modelPath) != 0 and len(imgPath) != 0 and len(outpath) != 0):

        label_file_explorer.configure(text= "Segmentation Starting")

        loadedModel = model_utils.load_model(modelPath)

        if(os.path.isdir(imgPath)):

            for image in os.listdir(imgPath):
 
                if(isImage(image)):

                    segmentAndPrint(loadedModel, image, outpath)

        else:

            if(isImage(imgPath)):
                
                segmentAndPrint(loadedModel, imgPath, outpath)

    else:

        outputString = ""

        if(len(modelPath) == 0):
            print(modelPath)
            outputString += "Please Specify A Rootpainter Model"

        if(len(imgPath) == 0):

            if(len(outputString) != 0):
                outputString += "; "

            outputString += "Please Specify A Image Path Or Image Directory"

        if(len(outpath) == 0):

            if(len(outputString) != 0):
                outputString += "; "

            outputString += "Please Specify An Output Directory"

        label_file_explorer.configure(text=outputString)

def enableFileSelection(*args):

    button_image.config(state = NORMAL)

def dropDownImageType(window, locationTuple):

    dropDownOptions = ["Singular Image", "Image Directory"]

    global inputTypeString
    inputTypeString = StringVar(window) 
    inputTypeString.set("Select Input Image Type")
    dropDownButton = OptionMenu(window, inputTypeString, *dropDownOptions)
    dropDownButton.pack()
    dropDownButton.grid(column = locationTuple[0], row = locationTuple[1])
    inputTypeString.trace_add("write", enableFileSelection)

def localSegmenterGUI(): 

    window.title('Local Rootpainter Segmenter GUI')
    window.maxsize()
    window.config(background = "white")

    global label_file_explorer
    label_file_explorer = Label(window, text = "Local Rootpainter Segmenter", width = 200, height = 4, fg = "blue")
    button_model = Button(window, text = "Select Model", command = browseForModel)
    global button_image
    button_image = Button(window, text = "Select Image Or Image Directory", command = browseForImg)
    button_image["state"] = "disabled"
    dropDownImageType(window, (1, 3)) #This line must be behind button_image defintion
    button_outpath = Button(window, text = "Select Destination Directory", command = browseForOutputFolder)
    button_segmentation = Button(window, text = "Segment", command = imageSegmentation)
    button_exit = Button(window, text = "End Software", command = exit) 
    
    label_file_explorer.grid(column = 1, row = 1)
    button_model.grid(column = 1, row = 2)
    button_image.grid(column = 1, row = 4)
    button_outpath.grid(column = 1, row = 5)
    button_segmentation.grid(column = 1, row = 6)
    button_exit.grid(column = 1,row = 7)
    
    window.mainloop()

localSegmenterGUI()