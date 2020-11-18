# file access simplified

import os

# sliding window with channel last
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    #print(image.shape)
    #print(stepSize)
    for y in range(0, image.shape[1], stepSize):
        #print('y',y)
        for x in range(0, image.shape[0], stepSize):
            #print('x',x,x + windowSize[0], y,y + windowSize[1] )
            # yield the current window
            yield x, y, image[x:x + windowSize[0], y:y + windowSize[1],:]

def sliding_windowBW(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[1], stepSize):
        for x in range(0, image.shape[0], stepSize):
            # yield the current window
            yield x, y, image[x:x + windowSize[0], y:y + windowSize[1]]

def ListFiles(PATH, contains, avoid, file_types):
    list_Files = []
    imagePath = []
    for (rootDir, dirNames, filenames) in os.walk(PATH):
        # loop over the filenames in the current directory

        for filename in filenames:
            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if avoid is None:
                if filename.find(contains) >= 1 and ext.endswith(file_types):
                    list_Files.append(filename)
                    imagePath.append(os.path.join(rootDir, filename))

            elif avoid is not None:
            # check to see if the file is an image and should be processed
                if filename.find(avoid) == -1 and ext.endswith(file_types):
                # construct the path to the image and yield it
                    list_Files.append(filename)
                    imagePath.append(os.path.join(rootDir, filename))

    # print('[DEBUG], path DSM',imagePath)
    return imagePath, list_Files

def ListFilesServer(PATH, contains, avoid, file_types):
    #works with SMB mounted directory (not AFP)
    list_Files = []
    imagePath = []
    for (rootDir, dirNames, filenames) in os.walk(PATH):
        # loop over the filenames in the current directory
        for filename in filenames:
            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            if ext.endswith(file_types):
                list_Files.append(filename)
                imagePath.append(os.path.join(rootDir, filename))

    # print('[DEBUG], path DSM',imagePath)
    return imagePath, list_Files