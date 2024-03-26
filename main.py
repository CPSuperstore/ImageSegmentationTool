import pickle
import sys

import segmentation_tool.interface.morph_acwe as morph_acwe
import segmentation_tool.interface.morph_gac as morph_gac
import segmentation_tool.interface.slic as slic
import segmentation_tool.interface.graph_cut as graph_cut
import segmentation_tool.interface.menu as menu_if
import segmentation_tool.interface.manual_segmentation as manual_segmentation
import segmentation_tool.interface.threshold_segmentation as threshold_segmentation
import segmentation_tool.interface.color_proximity_segmentation as color_proximity_segmentation

MENU_REGISTER = {
    "MorphACWE": morph_acwe.MorphACWE(),
    "MorphGAC": morph_gac.MorphGAC(),
    "SLIC": slic.SLIC(),
    "GraphCut": graph_cut.GraphCut(),
    "ManualSegmentation": manual_segmentation,
    "ThresholdSegmentation": threshold_segmentation.ThresholdSegmentation(),
    "ColorProximitySegmentation": color_proximity_segmentation.ColorProximitySegmentation()
}


def locked_menu(loaded_data=None):
    menu = menu_if.Menu(MENU_REGISTER.keys(), loaded_data=loaded_data)
    path = None

    while True:
        try:
            path, segment = menu.start(path)
        except TypeError:
            break

        seg_screen = MENU_REGISTER[segment]
        result = seg_screen.start(path)

        if result is not None:
            return result

        break


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        if sys.argv[1] == "--connect":
            with open(sys.argv[2], 'rb') as f:
                loaded_data = pickle.loads(f.read())

            result = locked_menu(loaded_data)

            with open(sys.argv[2], 'wb') as f:
                f.write(pickle.dumps(result))

        else:
            method = MENU_REGISTER[sys.argv[1]]
            method.start(sys.argv[2])

    else:
        locked_menu()
