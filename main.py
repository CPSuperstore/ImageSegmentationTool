import sys

import segmentation_tool.interface.morph_acwe as morph_acwe
import segmentation_tool.interface.morph_gac as morph_gac
import segmentation_tool.interface.slic as slic
import segmentation_tool.interface.menu as menu_if

MENU_REGISTER = {
    "MorphACWE": morph_acwe.MorphACWE(),
    "MorphGAC": morph_gac.MorphGAC(),
    "SLIC": slic.SLIC()
}

if len(sys.argv) >= 3:
    method = MENU_REGISTER[sys.argv[1]]
    method.start(sys.argv[2])

else:
    menu = menu_if.Menu(MENU_REGISTER.keys())
    path = None

    while True:
        try:
            path, segment = menu.start(path)
        except TypeError:
            break

        seg_screen = MENU_REGISTER[segment]
        seg_screen.start(path)