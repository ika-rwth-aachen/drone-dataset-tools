import pandas as pd
from typing import List, Dict
from os import path, listdir
import logging

from sortedcontainers import SortedList

from .config import FPS
from .RecordingData import RecordingData
from .LocationData import LocationData

class Loader:

    def __init__(self, directory, name):
        """
        Load data from a directory. Use name for dataset specific processing.
        Each directory has a set of location, and each location has a set of recordings. All data for a location has the same origin at (0,0).
        """
        self.directory = directory
        self.name = name
        self.recordingMeta = None
        self.locationIds = None
        self.recordingToLocationId = None
        self.locationToRecordingIds = None
        self.getAllRecordingMeta()
        pass

    def getAllRecordingMeta(self):
        """
        returns a list of maps. Each map has a set of recording ids, one background image
        """
        if self.recordingMeta is None:
            files = [path.join(self.directory, f) for f in listdir(self.directory) if 'recordingMeta' in f]
            self.recordingMeta = list( map(lambda f: pd.read_csv(f).to_dict(orient="records")[0], files))
            self.recordingToLocationId = {meta["recordingId"]: meta["locationId"] for meta in self.recordingMeta}

            self.locationToRecordingIds = {}
            for meta in self.recordingMeta:
                if meta['locationId'] not in self.locationToRecordingIds:
                    self.locationToRecordingIds[meta['locationId']] = []
                self.locationToRecordingIds[meta['locationId']].append(meta['recordingId'])

        return self.recordingMeta

    def getLocationIds(self):

        if self.locationIds is None:
            self.locationIds = set([])
            for rMeta in self.recordingMeta:
                self.locationIds.add(rMeta["locationId"])

        return self.locationIds
    
    def getRecordingIdsOfALocation(self, locationId):
        # ids = []
        # for rMeta in self.recordingMeta:
        #     if rMeta["locationId"] == locationId:
        #         ids.append(rMeta["recordingId"])
        # return ids
        recordingIds = self.locationToRecordingIds[locationId]
        # recordingIds = recordingIds[:1] # TODO for testing
        # recordingIds = [25]
        return recordingIds
    

    def getBackgroundImagePath(self, recordingId):
        # append a 0 because recordingId is fixed to two digits
        recordingId = int(recordingId)
        recordingId = recordingId if recordingId > 9 else f'0{recordingId}'
        return path.join(self.directory, f'{recordingId}_background.png') 
    

    def getRecordingData(self, recordingId, downSampleFps=FPS):
        """_summary_

        Args:
            recordingId (_type_): index of the recording
        Returns:
            tracksDf, tracksMetaDf, recordingMetaDf
        """

        if isinstance(recordingId, int):
            recordingId = str(recordingId).zfill(2)

        rMetaFile = path.join(self.directory, f'{recordingId}_recordingMeta.csv')
        tMetaFile = path.join(self.directory, f'{recordingId}_tracksMeta.csv')
        tracksFile = path.join(self.directory, f'{recordingId}_tracks.csv')
        recordingMeta = pd.read_csv(rMetaFile).to_dict(orient="records")[0]
        tracksMetaDf = pd.read_csv(tMetaFile)
        tracksDf = pd.read_csv(tracksFile)

        self.addUniqueTrackIds(tracksDf)    
        backgroundImagePath = self.getBackgroundImagePath(recordingId)


        return RecordingData(recordingId, recordingMeta, tracksMetaDf, tracksDf, backgroundImagePath, downSampleFps=downSampleFps)

        # return tracksDf, tracksMetaDf, recordingMeta

    def addUniqueTrackIds(self, tracksDf):
        tracksDf["uniqueTrackId"] = tracksDf["recordingId"] * 1000 + tracksDf["trackId"]
    
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
        return pd.concat(rDf, ignore_index=True)

    
    def getAllPedData(self, frameRate=25):
        """
        Returns all pedestrian data from all the locations.
        @param frameRate: The frameRate of the recording. We filter out the recordings which does not have a frameRate equal to this param.
        """
        meta = self.getAllRecordingMeta()
        pedDfs = []
        for tMeta in meta:
            if tMeta.frameRate != frameRate:
                print(f"Recording {tMeta.recordingId} has a different frameRate {tMeta.frameRate}")
                continue

            recordingId = tMeta['recordingId']
            tracksDf, tracksMetaDf, recordingMetaDf = self.getRecordingData(recordingId)
            pedDfs.append(self.extractPedFrames(tracksMetaDf, tracksDf))
        
        ## TODO, trackIds needs to be converted to unique Ids
        
        return self.mergePedFrames(pedDfs)

    
    def convertToSingleSequenceEpisodes(df):
        """
        We will have all the actor data as columns. Each row will have all the information of all the actors. Each Episode starts with a pedestrian and ends with the same pedestrian
        """
        pass


    def getLocationData(self, locationId, useSceneConfigToExtract=False, precomputeSceneData=True, recordingIds=None, downSampleFps=FPS):
        if recordingIds is None:
            recordingIds = self.getRecordingIdsOfALocation(locationId)
        logging.info(f"recordingIds: {recordingIds}")
        recordingDataList = [self.getRecordingData(rId, downSampleFps=downSampleFps) for rId in recordingIds]
        self.validateLocationRecordingMeta()

        return LocationData(
            locationId = locationId, 
            recordingIds = recordingIds, 
            recordingDataList = recordingDataList, 
            fps = downSampleFps,
            useSceneConfigToExtract = useSceneConfigToExtract,
            backgroundImagePath = self.getBackgroundImagePath(recordingIds[0]),
            precomputeSceneData=precomputeSceneData
        )

    

    def validateLocationRecordingMeta(self):
        pass

    