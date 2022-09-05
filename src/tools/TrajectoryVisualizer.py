from extractors.loader import Loader
from loguru import logger
import matplotlib.pyplot as plt
import os
import cv2

class TrajectoryVisualizer:

  def __init__(self, loader: Loader):

    self.loader = loader
  
  def initPlot(self, recordingId, trackId):
  
    # Create figure and axes
    self.fig, self.ax = plt.subplots(1, 1)
    self.fig.set_size_inches(15, 8)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.10, top=1.00)
    self.fig.canvas.set_window_title(f"Trajectory for track {trackId}")
    self.ax.set_title(f"Trajectory for track {trackId}")

    background_image_path = self.loader.getBackgroundImagePath(recordingId)
    if background_image_path and os.path.exists(background_image_path):
        logger.info("Loading background image from {}", background_image_path)
        self.background_image = cv2.cvtColor(cv2.imread(background_image_path), cv2.COLOR_BGR2RGB)
        (self.image_height, self.image_width) = self.background_image.shape[:2]
    else:
        logger.warning("No background image given or path not valid. Using fallback black background.")
        self.image_height, self.image_width = 1700, 1700
        self.background_image = np.zeros((self.image_height, self.image_width, 3), dtype="uint8")
    self.ax.imshow(self.background_image)
  


  def plot(self, recordingMeta, tracksDf):
    ortho_px_to_meter = recordingMeta["orthoPxToMeter"] * 12 # TODO fixed to scale down factor of inD

    self.ax.plot(tracksDf['xCenter'] / ortho_px_to_meter, -tracksDf['yCenter'] / ortho_px_to_meter, 'r')


  def showTrack(self, tracksDf, recordingMeta, trackId):
    self.initPlot(recordingMeta["recordingId"], trackId)

    pedTracksDf = tracksDf[tracksDf.trackId == trackId]
    self.plot(recordingMeta, pedTracksDf)
    return pedTracksDf

  
