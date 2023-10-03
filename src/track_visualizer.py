import matplotlib
from matplotlib.backend_bases import MouseButton

matplotlib.use('qt5agg')

import json
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from loguru import logger
from matplotlib import animation
from matplotlib.widgets import Button, TextBox

from sortedcontainers import SortedList


class TrackVisualizer(object):
    def __init__(self, config: dict, tracks: List[dict], tracks_meta: List[dict], recording_meta: dict):
        self.config = config
        self.input_path = config["dataset_dir"]
        self.dataset = config["dataset"].lower()
        self.location_id = recording_meta["locationId"]
        self.recording_name = config["recording"]
        self.playback_speed = config["playback_speed"]
        self.pedestrianFrameRanges = 0 # range of frames having pedestrians.
        self.currentPedRange = 0

        if config['ped_only']:
            self.pedestrianFrameRanges = self.getFramesWithPedestrians(tracks_meta)
            print(f'number of pedestrians {len(self.pedestrianFrameRanges)}')
            
        self.suppress_track_window = config["suppress_track_window"]

        # Currently clicked vehicle
        self.clicked_track_id = None
        # Color mapping for surrounding vehicles
        self.surrounding_vehicles_colors = {
            "leadId": "red",
            "rightLeadId": "orange",
            "rightAlongsideId": "black",
            "rightRearId": "purple",
            "rearId": "blue",
            "leftRearId": "green",
            "leftAlongsideId": "brown",
            "leftLeadId": "yellow"
        }
        vehicle_keys = list(self.surrounding_vehicles_colors.keys())
        self.surrounding_vehicles_ids = dict(zip(vehicle_keys, -1 * np.ones(len(vehicle_keys), dtype=int)))

        # Load dataset specific visualization parameters from file
        dataset_params_path = Path(config["visualizer_params_dir"]) / "visualizer_params.json"

        if not dataset_params_path.exists():
            logger.error("Could not find dataset visualization parameters in {}", dataset_params_path)
            sys.exit(-1)

        with open(dataset_params_path) as f:
            self.dataset_params = json.load(f)

        if self.dataset not in self.dataset_params["datasets"]:
            logger.error("Visualization parameters for dataset {} not found in {}. Please make sure, that the needed "
                         "parameters are given", self.dataset, dataset_params_path)
            sys.exit(-1)

        self.dataset_params = self.dataset_params["datasets"][self.dataset]
        self.scale_down_factor = self.dataset_params["scale_down_factor"]

        self.tracks = tracks
        self.tracks_meta = tracks_meta
        self.recording_meta = recording_meta

        # Check whether tracks and tracks_meta match each other
        error_message = "The tracks file and the tracksMeta file is not matching each other. " \
                        "Please check whether you modified any of these files."
        if len(tracks) != len(tracks_meta):
            logger.error(error_message)
            raise DataError("Failed", error_message)
        for track, track_meta in zip(tracks, tracks_meta):
            if track["trackId"] != track_meta["trackId"]:
                logger.error(error_message)
                raise DataError("Failed", error_message)

        # Determine the first and last frame
        self.minimum_frame = min(meta["initialFrame"] for meta in self.tracks_meta)
        self.maximum_frame = max(meta["finalFrame"] for meta in self.tracks_meta)
        logger.info("The recording contains tracks from frame {} to {}.", self.minimum_frame, self.maximum_frame)

        # Create a mapping between frame and idxs of tracks for quick lookup during playback
        self.frame_to_track_idxs = {}
        for i_frame in range(self.minimum_frame, self.maximum_frame + 1):
            indices = [i_track for i_track, track_meta in enumerate(self.tracks_meta)
                       if track_meta["initialFrame"] <= i_frame <= track_meta["finalFrame"]]
            self.frame_to_track_idxs[i_frame] = indices

        # Initialize data variables
        self.plot_handles = []
        self.track_info_figures = {}

        # Create figure and axes
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.set_size_inches(15, 8)
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.10, top=1.00)

        # Remove unwanted toolbar buttons
        toolbar = plt.get_current_fig_manager().toolbar
        unwanted_buttons = ['Subplots', 'Save', 'Customize', 'Forward', 'Back']
        for x in toolbar.actions():
            if x.text() in unwanted_buttons:
                toolbar.removeAction(x)

        self.fig.canvas.set_window_title("Tracks Visualizer - Dataset {}, Recording {}".format(self.dataset,
                                                                                               self.recording_name))

        # Show background image
        background_image_path = self.config["background_image_path"]
        if background_image_path and os.path.exists(background_image_path):
            logger.info("Loading background image from {}", background_image_path)
            self.background_image = cv2.cvtColor(cv2.imread(background_image_path), cv2.COLOR_BGR2RGB)
            (self.image_height, self.image_width) = self.background_image.shape[:2]
        else:
            logger.warning("No background image given or path not valid. Using fallback black background.")
            self.image_height, self.image_width = 1700, 1700
            self.background_image = np.zeros((self.image_height, self.image_width, 3), dtype="uint8")
        self.ax.imshow(self.background_image)

        # Find correct text font size
        track_label_font_size = 4
        if "orthoPxToMeter" in recording_meta:
            if recording_meta["orthoPxToMeter"] < 0.1:
                # For an urban area, we need smaller font sizes because the relevant areas are smaller
                track_label_font_size = 4
            else:
                # For highway areas, we need bigger font sizes because the relevant areas are bigger
                track_label_font_size = 6

        # Dictionaries for the style of the different objects that are visualized
        self.bbox_style = dict(fill=True, alpha=0.4, zorder=19)
        self.orientation_style = dict(facecolor="k", fill=True, edgecolor="k", lw=0.1, alpha=0.6, zorder=20)
        self.text_style = dict(picker=True, size=track_label_font_size, color='k', zorder=22, ha="center")
        self.text_box_style = dict(boxstyle="round,pad=0.2", alpha=.6, ec="black", lw=0.2, zorder=21)
        self.trajectory_style = dict(linewidth=1, zorder=10)
        self.future_trajectory_style = dict(color="linen", linewidth=1, alpha=0.7, zorder=10)
        self.centroid_style = dict(fill=True, edgecolor="black", lw=0.1, alpha=1, radius=0.5, zorder=30)
        # self.class_colors = dict(car="lightblue", van="purple", truck_bus="orange", bus="orange", truck="orange",
        #                          pedestrian="red", bicycle="yellow", motorcycle="yellow", default="green")
        self.class_colors = dict(car="lightblue", van="lightblue", truck_bus="lightblue", bus="lightblue",
                                 truck="lightblue",
                                 pedestrian="red", bicycle="lightblue", motorcycle="lightblue",
                                 default="lightblue")

        # Create legend
        self.legend_visible = False

        # Define axes for the widgets
        self.ax_textbox = self.fig.add_axes([0.27, 0.035, 0.04, 0.04])
        self.ax_button_previous2 = self.fig.add_axes([0.32, 0.035, 0.06, 0.04])
        self.ax_button_previous = self.fig.add_axes([0.39, 0.035, 0.06, 0.04])
        self.ax_button_play = self.fig.add_axes([0.46, 0.035, 0.06, 0.04])
        self.ax_button_next = self.fig.add_axes([0.53, 0.035, 0.06, 0.04])
        self.ax_button_next2 = self.fig.add_axes([0.60, 0.035, 0.06, 0.04])
        self.ax_button_reset = self.fig.add_axes([0.67, 0.035, 0.06, 0.04])

        if config['ped_only']:
            self.ax_textbox_ped = self.fig.add_axes([0.22, 0.035, 0.04, 0.04])
            self.ax_button_nextPed = self.fig.add_axes([0.74, 0.035, 0.09, 0.04])

        # Define the widgets
        self.textbox_frame = TextBox(self.ax_textbox, 'Set Frame ', initial=str(self.minimum_frame))

        self.button_previous2 = Button(self.ax_button_previous2, '',
                                       image=plt.imread("../assets/button_icons/previous2.png"))
        self.button_previous2.ax.axis('off')

        self.button_previous = Button(self.ax_button_previous, '',
                                      image=plt.imread("../assets/button_icons/previous.png"))
        self.button_previous.ax.axis('off')

        self.button_next = Button(self.ax_button_next, '', image=plt.imread("../assets/button_icons/next.png"))
        self.button_next.ax.axis('off')

        self.button_next2 = Button(self.ax_button_next2, '', image=plt.imread("../assets/button_icons/next2.png"))
        self.button_next2.ax.axis('off')

        self.play_image = plt.imread("../assets/button_icons/play.png")
        self.stop_image = plt.imread("../assets/button_icons/stop.png")
        self.button_play = Button(self.ax_button_play, '', image=self.play_image)
        self.button_play.ax.axis('off')
        self.button_reset = Button(self.ax_button_reset, 'Reset')
        if config['ped_only']:
            self.textbox_ped = TextBox(self.ax_textbox_ped, 'Ped ', initial=str(self.minimum_frame))
            self.button_nextPed = Button(self.ax_button_nextPed, 'Next Ped')

        # Define the callbacks for the widgets' actions
        self.button_previous.on_clicked(self._on_click_button_previous)
        self.button_previous2.on_clicked(self._update_button_previous2)
        self.button_next.on_clicked(self._on_click_button_next)
        self.button_next2.on_clicked(self._on_click_button_next2)
        self.button_play.on_clicked(self._start_stop_animation)
        self.button_reset.on_clicked(self._reset)
        if config['ped_only']:
            self.button_nextPed.on_clicked(self._jumpToNextPed)
        self.fig.canvas.mpl_connect('key_press_event', self._on_keypress)

        # Initialize main axes
        self.ax.set_autoscale_on(False)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        if "relevant_areas" in self.dataset_params and \
                str(recording_meta["locationId"]) in self.dataset_params["relevant_areas"]:
            limits = self.dataset_params["relevant_areas"][str(recording_meta["locationId"])]
            limits["x_lim"][0] = int(limits["x_lim"][0] / self.scale_down_factor)
            limits["x_lim"][1] = int(limits["x_lim"][1] / self.scale_down_factor)
            limits["y_lim"][0] = int(limits["y_lim"][0] / self.scale_down_factor)
            limits["y_lim"][1] = int(limits["y_lim"][1] / self.scale_down_factor)
            self.ax.set_xlim(limits["x_lim"])
            self.ax.set_ylim(limits["y_lim"])

        self.ax.axis('off')

        # Initialize visualization options
        self.current_frame = self.minimum_frame

        # Do not start the animation by default
        self.animation_running = False
        self._set_controls_activation(True)

        # Create animation instance•
        self.track_animation = animation.FuncAnimation(self.fig, self._update_figure, interval=20, blit=True,
                                                       init_func=self._clear_figure, cache_frame_data=False)

        
        # Add listener to figure so that clicks on tracks open a plot window
        self.fig.canvas.mpl_connect('pick_event', self._open_track_plots_window)

    def show(self):
        """
        Show the main windows of the Track visualizer.
        """
        # Show window maximized
        if self.config["show_maximized"]:
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()

        plt.show()

    def _update_figure(self, *args):
        """
        Main function to draw all tracks and selected annotations for the current frame.
        :param args: Should be unused if called manually. If called by FuncAnimation, args contains a call counter.
        :return: List of artist handles that have been updated. Needed for blitting.
        """
        # if self.animation_running == False:
        #     self._clear_figure()
        #     return self.plot_handles
        # Detect if the function was called manually (as FuncAnimation always adds a call counter). If the function
        # is called manually, draw all objects directly.
        animate = len(args) != 0

        # First remove all existing drawings
        self._clear_figure()

        # Plot the bounding boxes, their text annotations and direction arrow
        plot_handles = []
        for track_idx in self.frame_to_track_idxs[self.current_frame]:
            track = self.tracks[track_idx]

            track_id = track["trackId"]
            track_meta = self.tracks_meta[track_idx]
            initial_frame = track_meta["initialFrame"]
            current_index = self.current_frame - initial_frame

            # print("track_id", track_id)
            # print("current_index", current_index)

            object_class = track_meta["class"]
            if track["bboxVis"] is not None:
                bounding_box = track["bboxVis"][current_index] / self.scale_down_factor
            else:
                bounding_box = None
            center_points = track["centerVis"] / self.scale_down_factor
            center_point = center_points[current_index]

            color = self.class_colors.get(object_class, self.class_colors["default"])

            if self.clicked_track_id and track_id == self.clicked_track_id:
                current_frame = self.current_frame - track_meta["initialFrame"]
                self._find_surrounding_vehicles(current_frame, track, show_log=False)

            if self.config["show_bounding_box"]:
                edge_color = None
                bbox_color = color
                for vehicle_key, vehicle_id in self.surrounding_vehicles_ids.items():
                    if isinstance(vehicle_id, list) and track_id in vehicle_id:
                        bbox_color = self.surrounding_vehicles_colors[vehicle_key]
                        break
                    elif vehicle_id == track_id:
                        bbox_color = self.surrounding_vehicles_colors[vehicle_key]
                        break

                if track_id == self.clicked_track_id:
                    edge_color = "red"

                if bounding_box is not None:
                    if edge_color is not None:
                        bbox = plt.Polygon(bounding_box, True, facecolor=bbox_color, edgecolor=edge_color,
                                           **self.bbox_style)
                    else:
                        bbox = plt.Polygon(bounding_box, True, facecolor=bbox_color, edgecolor="k", **self.bbox_style)
                else:
                    bbox = plt.Circle(center_point, radius=2, facecolor=bbox_color)

                bbox.set_animated(animate)

                # Make bbox clickable to open track info window
                bbox.set_picker(True)
                bbox.track_id = track["trackId"]

                self.ax.add_patch(bbox)
                plot_handles.append(bbox)

            if self.config["show_orientation"] and bounding_box is not None:
                # Add triangles that display the direction of the cars
                triangle_factor = 0.25
                a_x = bounding_box[3, 0] + ((bounding_box[2, 0] - bounding_box[3, 0]) * triangle_factor)
                b_x = bounding_box[0, 0] + ((bounding_box[1, 0] - bounding_box[0, 0]) * triangle_factor)
                c_x = bounding_box[0, 0] + ((bounding_box[3, 0] - bounding_box[0, 0]) * 0.5)
                triangle_x_position = np.array([a_x, b_x, c_x])

                a_y = bounding_box[3, 1] + ((bounding_box[2, 1] - bounding_box[3, 1]) * triangle_factor)
                b_y = bounding_box[0, 1] + ((bounding_box[1, 1] - bounding_box[0, 1]) * triangle_factor)
                c_y = bounding_box[0, 1] + ((bounding_box[3, 1] - bounding_box[0, 1]) * 0.5)
                triangle_y_position = np.array([a_y, b_y, c_y])

                # Differentiate between vehicles that drive on the upper or lower lanes
                triangle_info = np.array([triangle_x_position, triangle_y_position])
                polygon = plt.Polygon(np.transpose(triangle_info), True, **self.orientation_style)
                polygon.set_animated(animate)
                self.ax.add_patch(polygon)
                plot_handles.append(polygon)

            if self.config["show_trajectory"]:
                centroid = plt.Circle((center_point[0], center_point[1]),
                                      facecolor=color, **self.centroid_style)
                centroid.set_animated(animate)
                self.ax.add_patch(centroid)
                plot_handles.append(centroid)
                if center_points.shape[0] > 0:
                    plotted_past_line = plt.Polygon(center_points[0:current_index + 1:2], closed=False, color=color,
                                                    fill=False, **self.trajectory_style)
                    plotted_past_line.set_animated(animate)
                    self.ax.add_patch(plotted_past_line)
                    plot_handles.append(plotted_past_line)
                    if self.config["show_future_trajectory"]:
                        # Check track direction
                        plotted_centroids_future = plt.Polygon(center_points[current_index::2], closed=False,
                                                               fill=False, **self.future_trajectory_style)
                        plotted_centroids_future.set_animated(animate)
                        self.ax.add_patch(plotted_centroids_future)
                        plot_handles.append(plotted_centroids_future)

            # Compose annotation
            annotation_text = ''
            if self.config["annotate_track_id"]:
                # Plot the text annotation
                annotation_text = "ID{}".format(track_id)
            if self.config["annotate_class"]:
                if annotation_text != '':
                    annotation_text += '|'
                annotation_text += "{}".format(object_class[0])
            if self.config["annotate_speed"]:
                if annotation_text != '':
                    annotation_text += '|'
                current_velocity = np.sqrt(
                    track["xVelocity"][current_index] ** 2 + track["yVelocity"][current_index] ** 2) * 3.6
                annotation_text += "{:.2f}km/h".format(current_velocity)
            if self.config["annotate_orientation"]:
                if annotation_text != '':
                    annotation_text += '|'
                current_rotation = track["heading"][current_index]
                annotation_text += "Deg%.2f" % current_rotation
            if self.config["annotate_age"]:
                if annotation_text != '':
                    annotation_text += '|'
                age = track_meta["numFrames"]
                annotation_text += "Age%d/%d" % (current_index + 1, age)

            if annotation_text:
                text_patch = self.ax.text(center_point[0], center_point[1] - 2.5, annotation_text,
                                          bbox={"fc": color, **self.text_box_style}, animated=animate,
                                          **self.text_style)

                # Make text clickable to open track info window
                text_patch.set_picker(True)
                text_patch.track_id = track["trackId"]

                plot_handles.append(text_patch)

        # Draw current frame number
        x = self.ax.get_xlim()[0] + 5
        y = self.ax.get_ylim()[1] + int((self.ax.get_ylim()[0] - self.ax.get_ylim()[1]) * 0.05)
        label_current_frame = self.ax.text(x, y, "Frame: {}/{}".format(self.current_frame, self.maximum_frame),
                                           fontsize=12, color="white", animated=animate)
        plot_handles.append(label_current_frame)

        # we are gonna jump to the next frame with pedestrian
        

        # Update current frame
        if self.animation_running:
            self.current_frame = self.getNextFrame(self.current_frame)
            # print("current_frame", self.current_frame)
            self.textbox_frame.set_val(self.current_frame)

        # if self.current_frame == self.maximum_frame:
        #     self.current_frame = self.minimum_frame
        # elif self.animation_running:
        #     # This is the "play-speed"
        #     self.current_frame = min(self.current_frame + self.playback_speed, self.maximum_frame)
            
        #     print("current_frame", self.current_frame)

        #     # Update the textbox to new current frame
        #     self.textbox_frame.set_val(self.current_frame)

        self.plot_handles = plot_handles
        return plot_handles

    def _clear_figure(self):
        """
        Remove all dynamic objects (tracks including texts, bboxes, trajectories etc.)
        """
        for figure_object in self.plot_handles:
            if isinstance(figure_object, list):
                figure_object[0].remove()
            else:
                figure_object.remove()
        self.plot_handles = []
        return []  # Matplotlib requires returning a list of artists to draw

    def _set_controls_activation(self, active: bool):
        """
        Allows to (de-)activate and show/hide some controls depending on the run state of the animation.
        """
        self.textbox_frame.set_active(active)
        self.button_previous2.set_active(active)
        self.button_previous2.ax.set_visible(active)
        self.button_previous.set_active(active)
        self.button_previous.ax.set_visible(active)
        self.button_next.set_active(active)
        self.button_next.ax.set_visible(active)
        self.button_next2.set_active(active)
        self.button_next2.ax.set_visible(active)

    def _on_keypress(self, evt):
        """
        Handle updates in the frame textbox as well as keyboard hotkeys.
        """

        # When the textbox is focused, only accept the keypress for "enter" as we use this for the submission of the
        # current frame. This is due to the fact that when losing focus of the textbox, the textbox submits its value
        # anyways. However, we only want to update the visualization, when the user confirms it with a "enter".
        if self.textbox_frame.capturekeystrokes:
            if evt.key != "enter":
                return

            try:
                new_frame = int(self.textbox_frame.text)
            except ValueError:
                return

            if 0 <= new_frame <= self.maximum_frame:
                self.current_frame = new_frame
            else:
                logger.warning("The entered frame does not exist. Maximum frame is {}.", self.maximum_frame)

        if self.textbox_ped.capturekeystrokes:
            if evt.key != "enter":
                return

            try:
                pedId = int(self.textbox_ped.text)
                self.jumpToPed(pedId)
            except ValueError:
                return


        # Time navigation for left and right arrow, fast backward and fast forward
        if evt.key == "right" and self.current_frame + self.playback_speed < self.maximum_frame:
            self.current_frame = self.current_frame + self.playback_speed
        elif evt.key == "left" and self.current_frame - self.playback_speed >= 0:
            self.current_frame = self.current_frame - self.playback_speed
        elif evt.key == " ":
            self._start_stop_animation(None)

    def _on_click_button_next(self, _):
        if self.current_frame + 1 < self.maximum_frame:
            self.current_frame = self.current_frame + 1
        else:
            logger.warning("There are no frames available with an index higher than {}.", self.maximum_frame)

    def _on_click_button_next2(self, _):
        if self.current_frame + self.playback_speed < self.maximum_frame:
            self.current_frame = self.current_frame + self.playback_speed
        else:
            logger.warning("There are no frames available with an index higher than {}.", self.maximum_frame)

    def _on_click_button_previous(self, _):
        if self.current_frame - 1 >= 0:
            self.current_frame = self.current_frame - 1
        else:
            logger.warning("There are no frames available with an index lower than 0.")

    def _update_button_previous2(self, _):
        if self.current_frame - self.playback_speed >= 0:
            self.current_frame = self.current_frame - self.playback_speed
        else:
            logger.warning("There are no frames available with an index lower than 0.")

    def _start_stop_animation(self, _):
        if not self.animation_running:
            # Start animation
            self.ax_button_play.images[0].set_data(self.stop_image)
            self.button_play.canvas.draw_idle()

            self._set_controls_activation(False)
            self.animation_running = True
        else:
            # Stop animation
            self.ax_button_play.images[0].set_data(self.play_image)
            self.button_play.canvas.draw_idle()

            self.animation_running = False
            self._set_controls_activation(True)

    def _reset(self, _):
        
        self.ax_button_play.images[0].set_data(self.play_image)
        self.button_play.canvas.draw_idle()
        self._set_controls_activation(True)
        self.current_frame = self.minimum_frame
        self.animation_running = False

    def _show_legend(self):
        if self.legend_visible:
            return
        label_boxes = []
        for _, color in self.surrounding_vehicles_colors.items():
            label_boxes.append(self.ax.bar([np.nan], [np.nan], color=color, edgecolor="k", **self.bbox_style))
        self.ax.legend(label_boxes, self.surrounding_vehicles_colors.keys(), bbox_to_anchor=(1.05, 1), loc='upper left',
                       borderaxespad=0.)
        self.legend_visible = True

    def _open_track_plots_window(self, event):
        """
        Create and show a window visualizing the fields of a clicked track.
        """
        # Restrict events to only left mouse clicks
        if event.mouseevent.button != MouseButton.LEFT:
            return
        # Get clicked artist and check if it belongs to a shown track
        artist = event.artist
        if "track_id" not in artist.__dict__:
            return
        track_id = artist.track_id

        # Get track by id
        for track in self.tracks:
            if track["trackId"] == track_id:
                track = track
                break
        else:
            logger.error("No track with the ID {} was found. Nothing to show.", track_id)
            return

        # Get track meta by id
        track_meta = [track_meta for track_meta in self.tracks_meta if track_meta["trackId"] == track_id]
        if len(track_meta) != 1:
            logger.error("No track meta information was found for the ID {} was found. Nothing to show.", track_id)
            return
        track_meta = track_meta[0]

        # Get information of the selected track
        centroids = track["center"]
        rotations = track["heading"]
        centroids = np.transpose(centroids)
        initial_frame = track_meta["initialFrame"]
        final_frame = track_meta["finalFrame"]
        x_limits = [initial_frame, final_frame]
        track_frames = np.linspace(initial_frame, final_frame, centroids.shape[1], dtype=np.int64)

        max_relevant_lead_ttc = 10

        current_local_frame = self.current_frame - track_meta["initialFrame"]
        self.clicked_track_id = track_id
        self._find_surrounding_vehicles(current_local_frame, track)

        if self.suppress_track_window:
            return

        # Create a new figure that pops up
        fig = plt.figure(np.random.randint(0, 5000, 1))
        fig.canvas.mpl_connect('close_event', lambda evt: self._on_close_track_plots_window(evt, track_id))
        fig.canvas.mpl_connect('resize_event', lambda evt: fig.tight_layout())
        fig.set_size_inches(12, 7)
        fig.canvas.set_window_title("Recording {}, Track {} ({})".format(self.recording_name,
                                                                         track_id, track_meta["class"]))

        extra_plots = {
            "leadId": "Lead Vehicle Presence (1=present)",
            "leadDHW": "Lead Distance Headway [m]",
            "lonVelocity": "Longitudinal-Velocity [m/s]",
            "leadDV": "Lead Relative Velocity [m/s]",
            "leadTTC": f"Lead Time-To-Collision [s] (capped at {max_relevant_lead_ttc}s)",
            "lonAcceleration": "Longitudinal-Acceleration [m/s^2]"
        }

        extra_plots_available = all([track.get(extra_plot_key, None) is not None for extra_plot_key in extra_plots.keys()])
        if not extra_plots_available:
            extra_plots = {
                "xVelocity": "X-Velocity [m/s]",
                "yVelocity": "Y-Velocity [m/s]",
                "xAcceleration": "Longitudinal-Velocity [m/s]",
                "yAcceleration": "Y-Acceleration [m/s^2]",
                "lonVelocity": "Longitudinal-Velocity [m/s]",
                "lonAcceleration": "Longitudinal-Acceleration [m/s^2]"
            }

        # Check availability of fields to create correct plot layout
        num_plots = 3
        for key in extra_plots:
            if track.get(key, None) is not None:
                num_plots += 1

        subplot_index = 311
        if 3 < num_plots <= 6:
            subplot_index = 321
        elif 6 < num_plots <= 9:
            subplot_index = 331

        borders_list = []
        subplot_list = []

        def create_subplot(subplot_index, title, values, borders=None):
            sub_plot = plt.subplot(subplot_index, title=title)
            subplot_list.append(sub_plot)
            if borders is None:
                borders = [np.amin(values), np.amax(values)]
                borders[0] = borders[0] - np.sign(borders[0]) * 0.1 * borders[0]
                borders[1] = borders[1] + np.sign(borders[1]) * 0.1 * borders[1]
            if borders[0] == borders[1]:
                borders = [borders[0] - 0.1, borders[1] + 0.1]
            plt.plot(track_frames, values)
            plt.plot([self.current_frame, self.current_frame], borders, "--r")
            borders_list.append(borders)
            plt.xlim(x_limits)
            plt.ylim(borders)
            sub_plot.grid(True)
            plt.xlabel('Frame')
            return sub_plot

        if "traveledDistance" in track:
            create_subplot(subplot_index, "Traveled Distance [m]", track["traveledDistance"])
        else:
            create_subplot(subplot_index, "X-Position [m]", centroids[0, :])
        subplot_index += 1

        if "latLaneCenterOffset" in track:
            sub_plot = create_subplot(subplot_index, "Lateral Lane Center Offset [m]", track["latLaneCenterOffset"][:, 0])
            if "laneChange" in track:
                lane_change_signal = track["laneChange"]
                lane_change_idcs = np.nonzero(lane_change_signal)
                if lane_change_idcs:
                    lane_change_idcs = lane_change_idcs[0].tolist()
                for lane_change_idx in lane_change_idcs:
                    sub_plot.plot(lane_change_idx + track_meta["initialFrame"], 0, "rx")
                    sub_plot.text(lane_change_idx + track_meta["initialFrame"] + 1, 0.1, "LC")
        else:
            create_subplot(subplot_index, "Y-Position [m]", centroids[1, :])
        subplot_index += 1

        create_subplot(subplot_index, "Heading [deg]", np.unwrap(rotations, discont=360), borders=[-10, 400])
        subplot_index += 1

        lead_id_changes = {}
        if "leadId" in extra_plots:
            lead_id_signal = track["leadId"]
            lead_id_change_idcs = np.nonzero(np.diff(lead_id_signal))
            if lead_id_change_idcs:
                lead_id_change_idcs = (lead_id_change_idcs[0] + 1).tolist()
            else:
                lead_id_change_idcs = []
            lead_id_change_idcs.append(0)
            lead_id_changes = {lead_id_change_idx: lead_id_signal[lead_id_change_idx] for lead_id_change_idx in lead_id_change_idcs}

        for extra_plot_key, extra_plot_name in extra_plots.items():
            if track.get(extra_plot_key, None) is not None:
                plot_data = track[extra_plot_key]
                borders = None

                if extra_plot_key == "leadId":
                    plot_data = plot_data != -1
                    borders = [-0.5, 1.5]
                elif extra_plot_key == "leadTTC":
                    # Cap TTC value
                    plot_data[plot_data > max_relevant_lead_ttc] = max_relevant_lead_ttc
                    borders = [-1.5, max_relevant_lead_ttc + 0.5]
                elif extra_plot_key == "leadDV":
                    if np.all(plot_data == -1000):
                        borders = [0, 1]
                    else:
                        plot_data[plot_data == -1000] = np.nan
                        borders = [np.nanmin(plot_data) - 0.5, np.nanmax(plot_data) + 0.5]

                sub_plot = create_subplot(subplot_index, extra_plot_name, plot_data, borders)
                subplot_index = subplot_index + 1

                if extra_plot_key in ("leadId", "leadDHW", "leadDV", "leadTTC"):
                    for lead_id_change_idx, new_lead_id in lead_id_changes.items():
                        if new_lead_id == -1:
                            continue
                        sub_plot.plot(lead_id_change_idx + track_meta["initialFrame"], plot_data[lead_id_change_idx], "rx")
                        sub_plot.text(lead_id_change_idx + track_meta["initialFrame"], plot_data[lead_id_change_idx] + 0.1, str(new_lead_id))

        self.track_info_figures[track_id] = {"main_figure": fig,
                                             "borders": borders_list,
                                             "subplots": subplot_list}
        plt.show()

    def _find_surrounding_vehicles(self, current_frame: int, track: dict, show_log: bool = True):
        track_id = track["trackId"]
        header_log = False
        for surrounding_vehicle_key in self.surrounding_vehicles_ids.keys():
            surrounding_id = track.get(surrounding_vehicle_key, {current_frame: -1})[current_frame]
            if isinstance(surrounding_id, list) and len(surrounding_id) == 0:
                surrounding_id = -1
            self.surrounding_vehicles_ids[surrounding_vehicle_key] = surrounding_id
            if show_log and surrounding_id != -1:
                if not header_log:
                    logger.info(f"--- Surrounding vehicles for track {track_id} ---")
                    self._show_legend()
                    header_log = True
                logger.info(f"{surrounding_vehicle_key} "
                            f"({self.surrounding_vehicles_colors[surrounding_vehicle_key]}) "
                            f"surrounding vehicle for track {track_id}: {surrounding_id}")

    def _on_close_track_plots_window(self, _, track_id: int):
        if track_id in self.track_info_figures:
            self.track_info_figures[track_id]["main_figure"].canvas.mpl_disconnect('close_event')
            self.track_info_figures.pop(track_id)
    
    def _jumpToNextPed(self, _):
        running = self.animation_running

        if running:
            self._start_stop_animation(None)


        nextRange = self.currentPedRange + 1
        if nextRange >= len(self.pedestrianFrameRanges):
            nextRange = 0


        self.current_frame = self.pedestrianFrameRanges[nextRange][0]
        self.textbox_frame.set_val(self.current_frame)

        self.currentPedRange = nextRange

        if running:
            self._start_stop_animation(None)
    
    def jumpToPed(self, pedId):
        """
        jumps to the ped with id in box 
        """
        self.current_frame = self.getInitiaFrameFor(pedId)


    def getNextFrame(self, curFrame):

        if self.config['ped_only']:
            return self.getNextFrameWithPedestrians(curFrame, self.playback_speed)

        if self.current_frame == self.maximum_frame:
            return self.minimum_frame

        return min(self.current_frame + self.playback_speed, self.maximum_frame)


    def getFramesWithPedestrians(self, tracks_meta):
        """
        Attributes:
            tracks_meta: list of dictionary of tracks_meta_file.
        Returns: a list or ranges. Each range for each pedestrian. Not optimized for multiple pedestrians in a frame.
        """
        frames = SortedList([])
        for track in tracks_meta:
            if track['class'] == 'pedestrian':
                frames.add((track['initialFrame'], track['finalFrame']))
                
        return frames

    def getPedIds(self):
        ids = []
        for track in self.tracks_meta:
            if track['class'] == 'pedestrian':
                ids.append(track['trackId'])
        return ids
    
    def getInitiaFrameFor(self, trackId):
        for track in self.tracks_meta:
            if track['trackId'] == trackId:
                return track['initialFrame']
        return 0



    def getNextFrameWithPedestrians(self, curFrame, skip=1):
        # TODO use bijection for faster performance. use sorted containers.

        if len(self.pedestrianFrameRanges) == 0:
            raise Exception('No frame with pedestrian found')
        

        nextFrame = curFrame + skip
        if nextFrame >= self.maximum_frame:
            self.currentPedRange = 0
            return self.pedestrianFrameRanges[0][0]
        
        for idx, range in enumerate(self.pedestrianFrameRanges):

            if nextFrame <= range[0]:
                self.currentPedRange = 0
                return range[0]

            if range[0] <= nextFrame and  range[1] >= nextFrame:
                if idx < self.currentPedRange:
                    continue
                self.currentPedRange = idx
                return nextFrame


        # next frame is greater than all final frames
        self.currentPedRange = 0
        return self.pedestrianFrameRanges[0][0]


class DataError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
