# Using loader

The loader has some helper methods that add more annotations and filters to the datasets. 

List of added columns to tracks:
1. uniqueTrackId (recordingId * 1000 + trackId)

**Create a loader**
```
from extractors.loader import Loader
dataDir = "G:AV datasets/inD-dataset-v1.0/data/"
loader = Loader(dataDir, 'inD')
```

**reading all the recording meta**
```
rMeta = loader.getAllRecordingMeta()
```
**location recording dictionary**
```
print(loader.locationToRecordingIds)
/*
{4: [0, 1, 2, 3, 4, 5, 6],
 1: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
 2: [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
 3: [30, 31, 32]}
*/
```
**all the recording Ids of a location**
```
print(loader.getRecordingIdsOfALocation(1))
# [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
```
**get recording data by id**

This method returns a [RecordingData](RecordingData.py) object.
```
recordingData = loader.getRecordingData('30')
```

# RecordingData
properties:

1. recordingId
2. recordingMeta
3. tracksMetaDf
4. tracksDf

**get dataframe with crossing pedestrians only**
This api requires that the tracks meta file has the crossing annotations in a "crossing" column. This hasn't been released by us yet.
```
recordingData = loader.getRecordingData('18')
crossingDf = recordingData.getCrossingDf() # all the tracks are pedestrians who crossed a road.

```

**get pedestrian Ids**
```
pedIds = recordingData.getPedIds() # all pedestrians
crossingIds = recordingData.getCrossingPedIds() # pedestrians crossing
```

**dataset selection by track Ids**
```
selectedDf = recordingData.getDfByTrackIds(trackIds) # trackIds is a set of actor ids.
```

# LocationData
Aggregation over recording data of a location

**get all the unique crossing ids**
```
loc2data = loader.getLocationData(2)
loc2data.getUniqueCrossingIds()
```

**get all the crossing data**
```
loc2data = loader.getLocationData(2)
crossingDf = loc2data.getCrossingDf()
```