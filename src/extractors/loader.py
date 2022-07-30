import pandas as pd
from typing import List, Dict
from os import path

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

    def getLocationMeta(self):
        """
        returns a list of maps. Each map has a set of recording ids, one background image
        """
        return None

    def getRecordingData(self, recordId):
        """_summary_

        Args:
            recordId (_type_): index of the recording
        Returns:
            tracksDf, tracksMetaDf, recordingMetaDf
        """
        rMetaFile = path.join(self.directory, f'{recordId}_recordingMeta.csv')
        tMetaFile = path.join(self.directory, f'{recordId}_tracksMeta.csv')
        tracksFile = path.join(self.directory, f'{recordId}_tracks.csv')
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
        
