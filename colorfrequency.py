'''Extracts frames from a recording with the .bk2 extension.
Croops and reduces the size of each frame
Output: database in a HDF5 file
'''
import h5py
import numpy as np

database_names = ['concatTrain.hdf5', 'concatValid.hdf5', 'concatTest.hdf5']
output_database_names = ['colorTrain.hdf5', 'colorValid.hdf5', 'colorTest.hdf5']

# database_names = ['concatNeighbourTrain.hdf5', 'concatNeighbourValid.hdf5', 'concatNeighbourTest.hdf5']
# output_database_names = ['colorNeighbourTrain.hdf5', 'colorNeighbourValid.hdf5', 'colorNeighbourTest.hdf5']
repertory = './data/'
#data = h5py.File('./data/data_All_Stars.hdf5', 'r')
#list_nbMovies = [0, 0, 4, 1, 4, 4]

def getIndexPix(pixel):
    return (pixel[0], pixel[1], pixel[2])

def buildPixDictionary(database):
    pixDictionary = {}
    batch, height, width, channel = database.shape
    print(batch)
    for img in range(100):
        for i in range(height):
            for j in range(width):
                key = getIndexPix(database[img,i,j])
                if key in pixDictionary:
                    pixDictionary[key] += 1
                else:
                    pixDictionary[key] = 1
        print("Img number ",img," len Dictionary",len(pixDictionary))
    return pixDictionary

def convertToFreq(database, pixDictionary, output_database_name):
    sizePixDict = len(pixDictionary)
    batch, height, width, channel = database.shape
    const_coef = height*width*batch / sizePixDict
    freqDatabase = []
    for img in range(batch):
        freqImg = []
        for i in range(height):
            freqRow = []
            for j in range(width):
                key = getIndexPix(database[img,i,j])
                if key in pixDictionary:
                    apparitions = pixDictionary[key]
                else:
                    apparitions = 1
                #color_coef = height*width*batch / (sizePixDict*apparitions)
                freq_coef = const_coef / apparitions
                freqRow.append(freq_coef)
            freqImg.append(freqRow)
        freqDatabase.append(freqImg)
    print('Freq Database shape: ',freqDatabase.shape)
    my_file = h5py.File(output_database_name, 'a')
    try:
        dataset = my_file.create_dataset(name='freqColor', data=freqDatabase, dtype="f8")
    except RuntimeError:
        pass
    my_file.close()


def main(database_names, output_database_names, repertory):
    trainData_name = repertory + database_names[0]
    trainData = h5py.File(trainData_name, 'r')
    pixDictionary = buildPixDictionary(trainData['/']['runs'])
    print("DONE Building Dictionary of length", len(pixDictionary))
    for nBase in range(len(database_names)):
        database_name = repertory + database_names[nBase]
        output_database_name = repertory + output_database_names[nBase]
        database = h5py.File(database_name, 'r')
        convertToFreq(database['/']['runs'], pixDictionary,
                      output_database_name)

if __name__ == '__main__':
    main(database_names, output_database_names, repertory)
