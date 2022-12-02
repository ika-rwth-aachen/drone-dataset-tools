# Using loader
[Loader example notebook](../src/notebooks/extractor-test.ipynb)

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
```

**dataset selection by track Ids**
```
selectedDf = recordingData.getDfByTrackIds(trackIds) # trackIds is a set of actor ids.
```

### Crossing data
Currently we can extract pedestrian frames (only peds, no other traffic participants) who crosses the road in two approaches:
1. By annotation. We manually annotated some track meta files. It's robust to errors.
2. By scene config. We picked some scenes for a location. Using geometric intersection, we extract crossing trajectories. But this cannot filter out trajectories which were possibly wrongly classified as pedestrians. Also, it cannot capture the trajectories that leaves the bounding box of the scene very early. The scene configs can be found [here](../data/scenes/ind.json).
```
recordingData.getCrossingDfByAnnotations() 
recordingData.getCrossingDfBySceneConfig(sceneConfigs, refresh=False, fps=FPS)
```
# LocationData
Aggregation over recording data of a location. Crossing data in a location can be extracted in two different ways:
1. By annotation: if the dataset has crossing annotation for pedestrian, this approach is fast 
2. By scene config: A scene config defines a bounding box, the pedestrian trajectories that have overlap with the bounding boxes are considerred crossing ones. This is prone to errors if a pedestrian does not cross the road (just walks along the road).

**Bulding location data will all scene annotations"
```
loc2data = loader.getLocationData(2)
loc2data.buildLocalInformationForScenes() # builds the local coordinate data.
```

**get all the unique crossing ids**
```
loc2data = loader.getLocationData(2)
loc2data.getUniqueCrossingIds()
```

**get all the crossing data**
```
loc2data = loader.getLocationData(2) # by annotation
loc2data = loader.getLocationData(2, useSceneConfigToExtract=True) by scene config

crossingDf = loc2data.getCrossingDf()
```

**Processed data in files**
We have different types of preprocessed data and formats. For all the examples please go through the notebook:
[Reading from preprocessed data](../src/notebooks/read-from-preprocessed.ipynb)
```
loc2data = loader.getLocationData(2, useSceneConfigToExtract=True) by scene config
loc2data.buildLocalInformationForScenes() # builds the local coordinate data.

loc2data.saveCrossingDf("../data") # only crossing dataframes. 
loc2data.save("../data") # whole thing as a dill object. 
loc2data.saveSceneDataOnly("../data")

# loading data
loc2dataFromFile = LocationData.load("../data/location-2", "dill file name")

```

# SceneData

This class holds the data of a scene. The purpose of this class is to crop trajectories to the scene area and transfrom them to scene coordinate system. **getClippedPedDfs** method returns the clipped trajectories. Here goes the important columns:
- sceneId: a numeric id represent a scene location. It will be the same value for a SceneData object
- sceneX, sceneY: xCenter, yCenter transformed into scene origin.

**Extracting ccene data from location data**
```
  sceneData = loc2data.getSceneData(1, 10, 5, refresh=False)
  sceneData.buildLocalInformation() # Optional. see notes below.
  visualizer.showLocationSceneData(SceneData, showLocal=True)
```

If you didn't call **buildLocalInformationForScenes()** on location data object, you will need to call buildLocalInformation on the sceneData object. Otherwise, no information on scene coordinates system will be generated.

**Get clipped trajectories**
Vehicle trajectories are clipped by about 50 meter + bounding box width. Can be changed by setting **OTHER_CLIP_LENGTH** constant in the SceneData class
```
sceneLocalDf = scene6Data.getClippedPedDfs() # pedestrians crossing
sceneLocalDf.head()

sceneLocalDf = scene6Data.getClippedOtherDfs() # others (vehicles, bicycles, trucks)
sceneLocalDf.head()
```

# Exporting data to formats for different purposes
## [Trajectory transformer]('https://github.com/FGiuliari/Trajectory-Transformer')

```
transformerData = loc2data.getCrossingDataForTransformer()
```



