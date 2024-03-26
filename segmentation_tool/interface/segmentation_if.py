import abc
import io
import pickle

import PIL.Image
import PySimpleGUI as sg
import cv2
import numpy as np
import shapely
from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage
import itertools

CONTOUR_COLOR_CYCLE = itertools.cycle([
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
])


class SegmentationInterface(abc.ABC):
    def _is_slow(self):
        return False

    @abc.abstractmethod
    def _get_controls(self):
        pass

    @abc.abstractmethod
    def _segment(self, image, mask, kwargs):
        pass

    def _get_interface(self):
        controls = self._get_controls()

        interface = []
        for control in controls:
            interface_control = None

            if control.control_type == "number":
                interface_control = sg.Slider(
                    range=control.allowed_values, default_value=control.default, enable_events=True,
                    resolution=control.step,
                    orientation='horizontal', key=control.kwarg
                )

            if control.control_type == "color":
                interface_control = sg.Button("Select", key=control.kwarg, enable_events=True)

            interface.append([
                sg.Text(control.display_name), interface_control
            ])

        return interface

    def start(self, image_path: str):
        def update_image(image_array, polygons=None):
            nonlocal last_image
            nonlocal latest_segments

            latest_segments = polygons

            if polygons is not None:

                for segment in polygons:
                    swapped_coords = np.dstack((segment[:, 1], segment[:, 0]))

                    cv2.fillPoly(image_array, np.array([swapped_coords]).astype(int), contour_color)

            last_image = Image.fromarray(image_array.astype(np.uint8))
            last_image.thumbnail((750, 500))

            with io.BytesIO() as output:
                last_image.save(output, format="PNG")
                data = output.getvalue()

            graph.draw_image(data=data, location=(0, graph_height))

        segments = None
        last_image: PIL.Image.Image = None

        controls = self._get_controls()
        control_kwargs = [c.kwarg for c in controls]

        color_button_keys = [c.kwarg for c in controls if c.control_type == "color"]
        color_button_selections = {c.kwarg: None for c in controls if c.control_type == "color"}
        awaiting_color_button = None

        if isinstance(image_path, list):
            im = Image.fromarray(image_path[0].astype(np.uint8))
            mask = image_path[1]
            attached = True

        else:
            im = Image.open(image_path)
            mask = None
            attached = False

        array = np.array(im, dtype=np.uint8)
        array = array[:, :, :3]

        original_array = array.copy()

        latest_segments = []

        contour_color = CONTOUR_COLOR_CYCLE.__next__()

        graph_ref = im.copy()
        graph_ref.thumbnail((750, 500))
        graph_width, graph_height = graph_ref.size

        layout = [
            [
                sg.Graph(
                    canvas_size=(graph_width, graph_height),
                    graph_bottom_left=(0, 0),
                    graph_top_right=(graph_width, graph_height),
                    key="-GRAPH-",
                    # change_submits=False,  # mouse click events
                    background_color='black',
                    # drag_submits=True,
                    enable_events=True
                )
            ]
        ]
        layout.extend(self._get_interface())

        if self._is_slow():
            layout.append([
                sg.Button("Home", key="--home"),
                sg.Button("Save Image", key="--save"),
                sg.Button("Polygon Histogram", key="--poly-histo"),
                sg.Button("Export Segmentation", key="--export"),
                sg.Button("Contour Color", key="--contour-color"),
                sg.Button("Reset", key="--reset"),
                sg.Button("Update", key="--update")
            ])

        else:
            layout.append([
                sg.Button("Home", key="--home"),
                sg.Button("Save Image", key="--save"),
                sg.Button("Polygon Histogram", key="--poly-histo"),
                sg.Button("Export Segmentation", key="--export"),
                sg.Button("Contour Color", key="--contour-color"),
                sg.Button("Reset", key="--reset"),
            ])

        layout.append([
            sg.Slider(
                range=[0, 30], default_value=0, enable_events=False,
                resolution=1,
                orientation='horizontal', key="--median-filter-size"
            ),
            sg.Button("Median Filter", key="--median-filter"),
        ])

        window = sg.Window("Image Segmentation - {}".format(self.__class__.__name__), layout, finalize=True)
        graph = window["-GRAPH-"]

        update_image(array)

        while True:
            event, values = window.read()
            update = False
            if event is None:
                break

            if event in color_button_keys:
                awaiting_color_button = event

            if event == '-GRAPH-':
                if awaiting_color_button is not None:
                    x, y = values[event]
                    color = last_image.getpixel((x, last_image.height - y))

                    window[awaiting_color_button].update(button_color=('black', '#%02x%02x%02x' % color))
                    color_button_selections[awaiting_color_button] = color

                    awaiting_color_button = None

                    if not self._is_slow():
                        update = True

            if (
                    (event in control_kwargs and not self._is_slow()) or (self._is_slow() and event == "--update") or update
            ) and event not in color_button_keys:

                for v in color_button_selections.values():
                    if v is None:
                        continue

                kwargs = {k: values[k] for k in control_kwargs if k not in color_button_keys}
                kwargs.update(color_button_selections)

                new_image = array.copy()

                segments = self._segment(new_image, mask, kwargs)

                if not isinstance(segments, list) and segments.shape == new_image.shape:
                    update_image(segments)

                else:
                    update_image(new_image, segments)

            if event == "--home":
                break

            if event == "--save":
                path = sg.filedialog.asksaveasfilename(
                    filetypes=(("Image Files", "*.png *.jpeg"),),
                    defaultextension="png",
                    parent=window.TKroot,
                    title="Save As"
                )

                if path != "":
                    last_image.save(path)

            if event == "--export":
                if attached:
                    window.close()
                    return latest_segments

                else:
                    path = sg.filedialog.asksaveasfilename(
                        filetypes=(("Segmentation Data", "*.dat"),),
                        defaultextension="dat",
                        parent=window.TKroot,
                        title="Save As"
                    )

                    if path != "":
                        with open(path, 'wb') as f:
                            f.write(pickle.dumps(latest_segments))

            if event == "--reset":
                new_image = array.copy()
                update_image(new_image, [])

            if event == "--contour-color":
                contour_color = CONTOUR_COLOR_CYCLE.__next__()
                new_image = array.copy()
                update_image(new_image, segments)

            if event == "--median-filter":
                size = int(values["--median-filter-size"])
                array = original_array.copy()

                if size > 0:
                    array = ndimage.median_filter(array, size=size)

                update_image(array.copy(), segments)

            if event == "--poly-histo":
                polygons = [p for p in segments if len(p) > 4]

                if len(polygons) > 0:
                    areas = [shapely.Polygon(p).area for p in polygons]

                    bin_edges = np.histogram_bin_edges(areas, bins='scott')
                    print("Generating histogram with {} bins...".format(len(bin_edges)))

                    plt.hist(areas, bins=bin_edges)

                    plt.yscale("log")
                    plt.title("Extracted Polygon Area Distribution")
                    plt.ylabel("Count")
                    plt.xlabel("Area")

                    plt.show()

        window.close()


class Control:
    def __init__(self, display_name, control_type, kwarg, allowed_values: list = None, default=None, step=1):
        self.display_name = display_name
        self.control_type = control_type
        self.kwarg = kwarg
        self.allowed_values = allowed_values
        self.default = default
        self.step = step
