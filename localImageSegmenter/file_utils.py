import os

def ls(dir_path):
    # Don't show hidden files
    # These can happen due to issues like file system 
    # synchonisation technology. RootPainter doesn't use them anywhere
    fnames = os.listdir(dir_path)
    fnames = [f for f in fnames if f[0] != '.']
    return fnames