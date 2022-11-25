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