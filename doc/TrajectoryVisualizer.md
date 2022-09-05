# Track Visualizer

It shows trajectories of tracks.
## Visualising all pedestrian tracks

```
dataDir = "........../data/"
loader = Loader(dataDir, 'inD')
recordingData = loader.getRecordingData(recordingId)
visualizer = TrajectoryVisualizer(loader)
for pedId in loader.getSortedPedIds(tracksMetaDf):
    visualizer.showTrack(recordingId, pedId)
```

## Visualising all crossing pedestrian tracks

```
dataDir = "........../data/"
loader = Loader(dataDir, 'inD')
recordingData = loader.getRecordingData(recordingId)
visualizer = TrajectoryVisualizer(loader)
for pedId in recordingData.getCrossingPedIds(tracksMetaDf):
    visualizer.showTrack(recordingId, pedId)
```