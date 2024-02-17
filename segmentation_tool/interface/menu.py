import PySimpleGUI as sg


class Menu:
    def __init__(self, segmentation_methods: list):
        self.segmentation_methods = segmentation_methods

    def start(self, default_path: str):
        layout = [
            [sg.FileBrowse(initial_folder=default_path), sg.Text("Choose a file: ", size=(50, None))]
        ]

        for method in self.segmentation_methods:
            layout.append([sg.Button(method, key=method)])

        window = sg.Window("Image Segmentation", layout, finalize=True)

        while True:
            event, values = window.read()
            if event is None:
                break

            if event in self.segmentation_methods:
                if values["Browse"] != "":
                    window.close()
                    return values["Browse"], event

        window.close()
