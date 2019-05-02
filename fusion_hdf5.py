'''Extracts frames from a recording with the .bk2 extension.
Croops and reduces the size of each frame
Output: database in a HDF5 file
'''
import h5py
import numpy as np

database_name = './data/data_All_Stars.hdf5'
newNames = ['concatAllTrain.hdf5', 'concatAllValid.hdf5', 'concatAllTest.hdf5']
# database_name = './data/dataNeigbour.hdf5'
# newNames = ['concatNeighbourTrain.hdf5', 'concatNeighbourValid.hdf5', 'concatNeighbourTest.hdf5']
repertory = './data/'

#data = h5py.File('./data/data_All_Stars.hdf5', 'r')

listData = []
list_players = [ 'Flash', 'CaptainAmerica', 'NewGandhi', 'NewKickAss' ]
#list_nbMovies = [0, 0, 4, 1, 4, 4]

indexMovies = [ (0,45,50,55), (0,15,30,34), (0,20,25,30), (0,65,70,75)]
# indexMovies = [ (0,4,5,6), (0,1,2,3), (0,3,4,5), (0,4,5,6)]



def buildListData(list_players, data, nBase):
    listData = []
    for id in range(len(list_players)):
        joueur = list_players[id]
        list_movies = list(data['/'][joueur].keys())
        print("len movie", joueur, len(list_movies))
        for movie in list_movies[indexMovies[id][nBase]: indexMovies[id][nBase+1]]:
            listData.append(data['/'][joueur][movie])
    return listData

def saveToHDF5(newdatabase_name, data):
    my_file = h5py.File(repertory + newdatabase_name, 'a')
    try:
        dataset = my_file.create_dataset(name='runs', data=data, dtype="i8")
    except RuntimeError:
        print(database_name + " Not Save")
        pass
    my_file.close()

def concatAll(listData):
    all_runs = np.array(listData[0])
    for i in range(1, len(listData)):
        all_runs = np.concatenate((all_runs, listData[i]))
    return all_runs

def main():
    data = h5py.File(database_name, 'r')
    for nBase in range(3):
        listData = buildListData(list_players, data, nBase)
        print('Nombre de movies for ',newNames[nBase],' : ',len(listData))
        all_runs = concatAll(listData)
        #all_runs = all_runs/255. dtype = "f8"
        print('New Database shape: ',all_runs.shape)
        saveToHDF5(newNames[nBase], all_runs)
    data.close()
    print("concatenation done")

if __name__ == '__main__':
    main()
