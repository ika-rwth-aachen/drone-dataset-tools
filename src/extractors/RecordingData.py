import pandas as pd
from sortedcontainers import SortedList

class RecordingData:

  def __init__(self, recordingId, recordingMeta, tracksMetaDf, tracksDf):

    self.recordingId = recordingId
    self.recordingMeta = recordingMeta
    self.tracksMetaDf = tracksMetaDf
    self.tracksDf = tracksDf

    self.__crossingDf = None


  def getPedIds(self) -> SortedList:
    return SortedList(this.tracksMetaDf[tracksMetaDf['class'] == 'pedestrian']['trackId'].tolist())
  

  def getCrossingPedIds(self) -> SortedList:
    if 'crossing' not in self.tracksMetaDf:
      raise Error("crossing annotation not in tracksMetaDf")

    return SortedList(this.tracksMetaDf[(tracksMetaDf['class'] == 'pedestrian') & (tracksMetaDf['crossing'] == 'yes') ]['trackId'].tolist())


  def getDfByTrackIds(self, trackIds):
      criterion = self.tracksDf['trackId'].map(lambda trackId: trackId in pedIds)
      return self.tracksDf[criterion]

  
  def getCrossingDf(self):
    if self.__crossingDf is not None:
      crossingIds = self.getCrossingPedIds()
      self.__crossingDf = self.getDfByTrackIds(crossingIds)

    return self.__crossingDf



  