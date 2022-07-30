import pandas as pd
from typing import List, Dict
from os import path, listdir

from sortedcontainers import SortedList

class Loader:

    def __init__(self, directory, name):
        """
        Load data from a directory. Use name for dataset specific processing.
        Each directory has a set of location, and each location has a set of recordings. All data for a location has the same origin at (0,0).
        """
        self.directory = directory
        self.name = name
        pass

    def getAllLocationsMeta(self):
        """
        returns a list of maps. Each map has a set of recording ids, one background image
        """
        files = [path.join(self.directory, f) for f in listdir(self.directory) if 'recordingMeta' in f]
        print(files)
        
        meta = list( map(lambda f: pd.read_csv(f).to_dict(orient="records")[0], files))

        return meta

    def getRecordingData(self, recordingId):
        """_summary_

        Args:
            recordingId (_type_): index of the recording
        Returns:
            tracksDf, tracksMetaDf, recordingMetaDf
        """
        rMetaFile = path.join(self.directory, f'{recordingId}_recordingMeta.csv')
        tMetaFile = path.join(self.directory, f'{recordingId}_tracksMeta.csv')
        tracksFile = path.join(self.directory, f'{recordingId}_tracks.csv')
        recordingMetaDf = pd.read_csv(rMetaFile)
        tracksMetaDf = pd.read_csv(tMetaFile)
        tracksDf = pd.read_csv(tracksFile)

        return tracksDf, tracksMetaDf, recordingMetaDf


    
    def extractPedFrames(self, tracksMetaDf, tracksDf):
        # return frames with pedestrians only
        pedIds = self.getSortedPedIds(tracksMetaDf)
        criterion = tracksDf['trackId'].map(lambda trackId: trackId in pedIds)
        return tracksDf[criterion]
        

    def getSortedPedIds(self, tracksMetaDf) -> pd.Series:
        return SortedList(tracksMetaDf[tracksMetaDf['class'] == 'pedestrian']['trackId'].tolist())

    def mergePedFrames(self, rDfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        returns a single DataFrame with all the pedestrian records
        """
        return pd.concat(rDf)

    
    def getAllPedData(self):
        meta = self.getAllLocationsMeta()
        pedDfs = []
        for tMeta in meta:
            recordingId = tMeta['recordingId']
            tracksDf, tracksMetaDf, recordingMetaDf = self.getRecordingData(recordingId)
            pedDfs.append(self.extractPedFrames(tracksMetaDf, tracksDf))
        
        return self.mergePedFrames(pedDfs)


        
