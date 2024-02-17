import abc
import io

import PySimpleGUI as sg
import cv2
import numpy as np
from PIL import Image


class SegmentationInterface(abc.ABC):
    def _is_slow(self):
        return False

    @abc.abstractmethod
    def _get_controls(self):
        pass

    @abc.abstractmethod
    def _segment(self, image, kwargs):
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

            interface.append([
                sg.Text(control.display_name), interface_control
            ])

        return interface

    def start(self, image_path: str):
        def update_image(image_array, polygons=None, thickness=1):
            nonlocal last_image

            if polygons is not None:

                for segment in polygons:
                    swapped_coords = np.dstack((segment[:, 1], segment[:, 0]))

                    cv2.drawContours(
                        image_array, np.array([swapped_coords]).astype(int), -1, [255, 0, 0],
                        thickness=int(thickness)
                    )

                    # draw.polygon(polygon)

            last_image = Image.fromarray(image_array.astype(np.uint8))

            with io.BytesIO() as output:
                last_image.save(output, format="PNG")
                data = output.getvalue()

            graph.draw_image(data=data, location=(0, height))

        segments = None
        last_image = None

        controls = self._get_controls()
        control_kwargs = [c.kwarg for c in controls]

        im = Image.open(image_path)
        im.thumbnail((750, 500))

        width, height = im.size
        array = np.array(im, dtype=np.uint8)
        array = array[:, :, :3]

        layout = [
            [
                sg.Graph(
                    canvas_size=(width, height),
                    graph_bottom_left=(0, 0),
                    graph_top_right=(width, height),
                    key="-GRAPH-",
                    change_submits=False,  # mouse click events
                    background_color='black',
                    drag_submits=True
                )
            ],
            [
                sg.Text("Line Thickness"),
                sg.Slider(
                    range=[1, 30], default_value=1, enable_events=True, key="--thickness", orientation='horizontal'
                )
            ]
        ]
        layout.extend(self._get_interface())

        if self._is_slow():
            layout.append([
                sg.Button("Home", key="--home"),
                sg.Button("Save Image", key="--save"),
                sg.Button("Update", key="--update")
            ])

        else:
            layout.append([
                sg.Button("Home", key="--home"),
                sg.Button("Save Image", key="--save")
            ])

        window = sg.Window("Image Segmentation - {}".format(self.__class__.__name__), layout, finalize=True)
        graph = window["-GRAPH-"]

        update_image(array)

        while True:
            event, values = window.read()
            if event is None:
                break

            if (event in control_kwargs and not self._is_slow()) or (self._is_slow() and event == "--update"):
                kwargs = {k: values[k] for k in control_kwargs}
                new_image = array.copy()

                segments = self._segment(new_image, kwargs)

                if not isinstance(segments, list) and segments.shape == new_image.shape:
                    update_image(segments)

                else:
                    update_image(new_image, segments, values["--thickness"])

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

            if event == "--thickness":
                new_image = array.copy()
                update_image(new_image, segments, values["--thickness"])

        window.close()


class Control:
    def __init__(self, display_name, control_type, kwarg, allowed_values: list = None, default=None, step=1):
        self.display_name = display_name
        self.control_type = control_type
        self.kwarg = kwarg
        self.allowed_values = allowed_values
        self.default = default
        self.step = step
