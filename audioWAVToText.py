import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from pyAudioAnalysis.audioSegmentation import speaker_diarization
import pickle
import torch.nn as nn
import torch.optim as optim
import torch

def loadData(filePath):

    wavFile, Frequency = librosa.load(filePath)
    return (wavFile, Frequency)

def showAudiofFreqOverTime(audio):

    # split_indices = np.where(array < threshold)[0] + 1
    plt.plot(audio[0:102400])
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.show()
    # print(wavFile.shape)

def trainingProtocol(fileName):

    wavFile, Frequency = loadData(fileName)

    print("The Frequency is: " + str(Frequency))
    # sd.play(wavFile, Frequency, blocking=True)

def readAndsSegmentWavBySpeaker(wavFileName):

    """
        ARGUMENTS:
        - filename:        the name of the WAV file to be analyzed
        - n_speakers       the number of speakers (clusters) in
                           the recording (<=0 for unknown)
        - mid_window (opt)    mid-term window size
        - mid_step (opt)    mid-term window step
        - short_window  (opt)    short-term window size
        - lda_dim (opt     LDA dimension (0 for no LDA)
        - plot_res         (opt)   0 for not plotting the results 1 for plotting
    """

    array = speaker_diarization(wavFileName, n_speakers = 0, mid_step=0.2)

    print(array[0])

    with open("unkownSpeakersRecognition.txt", "w") as file:
        file.write(str(array[0].tolist()))

    print(array[0])

def runModelOverAudio(audio):

    speechDetected = []

    chunksOfAudioForModel = prepAudio(audio)
    model = modelLoader("PathToModel")

    for chunk in chunksOfAudioForModel:
        
        speechDetected.append(" " + model.forward(chunk))

def prepAudio(audio):

    audioChunksForModel = []    

    return audioChunksForModel

def modelLoader(filePathToModel):
     
    model = pickle.load(open(filePathToModel, 'rb')) 

    return model

def convertFromPhenomeToEnglishSounds(listOfSounds):

    outputList = []

    phon61_map39 = {
    'iy':'iy',  'ih':'ih',   'eh':'eh',  'ae':'ae',    'ix':'ih',  'ax':'ah',   'ah':'ah',  'uw':'uw',
    'ux':'uw',  'uh':'uh',   'ao':'aa',  'aa':'aa',    'ey':'ey',  'ay':'ay',   'oy':'oy',  'aw':'aw',
    'ow':'ow',  'l':'l',     'el':'l',  'r':'r',      'y':'y',    'w':'w',     'er':'er',  'axr':'er',
    'm':'m',    'em':'m',     'n':'n',    'nx':'n',     'en':'n',  'ng':'ng',   'eng':'ng', 'ch':'ch',
    'jh':'jh',  'dh':'dh',   'b':'b',    'd':'d',      'dx':'dx',  'g':'g',     'p':'p',    't':'t',
    'k':'k',    'z':'z',     'zh':'sh',  'v':'v',      'f':'f',    'th':'th',   's':'s',    'sh':'sh',
    'hh':'hh',  'hv':'hh',   'pcl':'h#', 'tcl':'h#', 'kcl':'h#', 'qcl':'h#','bcl':'h#','dcl':'h#',
    'gcl':'h#','h#':'h#',  '#h':'h#',  'pau':'h#', 'epi': 'h#','nx':'n',   'ax-h':'ah','q':'h#' 
    }

    for sound in listOfSounds:

        outputList.append(phon61_map39[sound])


def trainingFunction():

    model = soundToEnglishModel().cuda()

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loss_criteria = nn.CrossEntropyLoss()

    epochs = 100
    patience = 5
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)

        if test_loss < best_valid_loss:

            best_valid_loss = test_loss
            patience_counter = 0

            with open("speechRecognitionModel", 'wb') as file:
                pickle.dump(model, file)

        else:

            patience_counter += 1

        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)

        if patience_counter >= patience:
            print(f'Early stopping after {epoch} epochs')
            break


    optimizer.zero_grad()
    
def train(model, device, train_loader, optimizer, epochNumber):
    
    model.train()
    train_loss = 0
    print("------------------------------- Epoch:", epochNumber,"-------------------------------")

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data.to(device))

        loss = nn.CrossEntropyLoss(output, target)

        train_loss += loss.item()

        loss.backward(retain_graph=True)
        optimizer.step()
            
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss

def test(model, device, test_loader):

    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            test_loss += nn.CrossEntropyLoss(output, target).item()
            
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return avg_loss

class soundToEnglishModel:

    def __init__(self):
        super().__init__()
        # 5 conv blocks / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(6,6), stride=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(6,6), stride=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(5,5), stride=(2,2)),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=(5,5), stride=(2,2)),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(4,4), stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,4), stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.connected_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*43, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 50),
            nn.Softmax(),
        )

    def forward(self, input_data):
        input_data = input_data.to('cuda')
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x= self.connected_layer(x)
        return x

# readAndsSegmentWavBySpeaker("presidentialData.wav")

dictionary = {
    'iy':'iy',  'ih':'ih',   'eh':'eh',  'ae':'ae',    'ix':'ih',  'ax':'ah',   'ah':'ah',  'uw':'uw',
    'ux':'uw',  'uh':'uh',   'ao':'aa',  'aa':'aa',    'ey':'ey',  'ay':'ay',   'oy':'oy',  'aw':'aw',
    'ow':'ow',  'l':'l',     'el':'l',  'r':'r',      'y':'y',    'w':'w',     'er':'er',  'axr':'er',
    'm':'m',    'em':'m',     'n':'n',    'nx':'n',     'en':'n',  'ng':'ng',   'eng':'ng', 'ch':'ch',
    'jh':'jh',  'dh':'dh',   'b':'b',    'd':'d',      'dx':'dx',  'g':'g',     'p':'p',    't':'t',
    'k':'k',    'z':'z',     'zh':'sh',  'v':'v',      'f':'f',    'th':'th',   's':'s',    'sh':'sh',
    'hh':'hh',  'hv':'hh',   'pcl':'h#', 'tcl':'h#', 'kcl':'h#', 'qcl':'h#','bcl':'h#','dcl':'h#',
    'gcl':'h#','h#':'h#',  '#h':'h#',  'pau':'h#', 'epi': 'h#','nx':'n',   'ax-h':'ah','q':'h#' 
    }

outputDictionary = {}

for key in dictionary.keys():

    answer = dictionary[key]
    outputDictionary[answer] = answer

print(len(outputDictionary.keys()))