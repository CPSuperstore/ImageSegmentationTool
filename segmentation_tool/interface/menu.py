import PySimpleGUI as sg


class Menu:
    def __init__(self, segmentation_methods: list, loaded_data=None):
        self.segmentation_methods = segmentation_methods
        self.loaded_data = loaded_data

    def start(self, default_path: str):
        if self.loaded_data is None:
            layout = [
                [sg.FileBrowse(initial_folder=default_path), sg.Text("Choose a file: ", size=(50, None))]
            ]
        else:
            layout = [
                [sg.Text("Connected to external application")]
            ]

        for method in self.segmentation_methods:
            layout.append([sg.Button(method, key=method)])

        window = sg.Window("Image Segmentation", layout, finalize=True)

        while True:
            event, values = window.read()
            if event is None:
                break

            if event in self.segmentation_methods:
                if self.loaded_data is not None:
                    window.close()
                    return self.loaded_data, event

                if values["Browse"] != "":
                    window.close()
                    return values["Browse"], event

        window.close()
