As the preprocessing takes a long time, it's best to preprocess the location data and later user them from saved files.


## Step 1: Preprocess and save the location data:

```

from extractors.loader import Loader
dataDir = "E:/Datasets/inD-dataset-v1.0/data/" # path to the InD data directory
loader = Loader(dataDir, 'inD')
loc2data = loader.getLocationData(2, useSceneConfigToExtract=True)
loc2data.buildLocalInformationForScenes() # data in scene coordinate system
loc2data.saveCrossingDf("../data") # only crossing dataframes
loc2data.save("../data") # whole thing as a dill object
loc2data.saveSceneDataOnly("../data") # individual scenes
```

You will find several data files prefixed with the date.

## Step 2: Load preprocessed data from disk 

**Notebook with updated examples**: 
[Reading from preprocessed data](../src/notebooks/read-from-preprocessed.ipynb)


The csv files can be read with pandas DataFrame. To load the pickled location data:

```
from extractors.LocationData import LocationData
loc2dataFromFile = LocationData.load("../data/location-2", "2022-11-12-all.dill")
```

## Step 2.1: Visualize a scene data:

```
from tools.TrajectoryUtils import TrajectoryUtils
from tools.TrajectoryVisualizer import TrajectoryVisualizer
import matplotlib.pyplot as plt
import numpy as np
visualizer = TrajectoryVisualizer(None)

sceneData = loc2dataFromFile.getSceneData(10, 10, 5, refresh=False)
visualizer.showLocationSceneData(sceneData, onlyClipped=True, showOthers=True)
```