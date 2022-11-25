# Pedestrian trajectory extraction and processing

## Architecture & Tutorials
1. [Methodology for Extraction](extractors-methodology.md) 
2. [Quick start](how-to-use.md)
3. [Extractors](extractors.md)
4. [Visualization tools](visualization.md)
5. [Threads to validity](threats-to-validity.md)


## Scene coordinate system:
<img src="./images/scene-coordinate-system.PNG" width="600">

The original data has position and dynamics in the image coordinate system which is the top-left pixel of the background image. Our extracted data transforms trajectories into scene coordinate system where the origin is at the center of the scene bounding box and x axis is rotated counter-clockwise to aligh with the road reference line (length).

So, our scene data looks like this (plotted in the image coordinate system)
<img src="./images/scene-data.PNG" width="600">

## Derived Data

**Additional attributes for track**

| Attribute | Description |
| --- | ------ |
| uniqueTrackId | Unique track identifier in a location. First two digits denote the recordingId and last three digits denote the trackId in the recording. |
| sceneId | Every scene in a location has a sceneId |
| roadWidth | Approximate road width along the scene y-axis |
| sceneX | track x position in the scene coordinate system |
| sceneY | track y position in the scene coordinate system |

**Scene meta data**

The meta data is developed in clipped trajectories in the scene coordinate system.

| Attribute | Description |
| --- | ------ |
| uniqueTrackId | Unique track identifier in a location. First two digits denote the recordingId and last three digits denote the trackId in the recording. |
| initialFrame | starting frame in the recording |
| finalFrame |  ending frame in the recording  |
| numFrames | Life span in frames. Depends on the FPS of the data. |
| class | type of the actor |
| horizontalDirection | Positive x is EAST |
| verticalDirection | Positive y is NORTH |


