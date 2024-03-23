import pickle

import PySimpleGUI as sg
import pygame

pygame.init()


def start(image_path: str):
    scale = 1
    image = pygame.image.load(image_path)
    pygame.display.set_caption("Image Segmentation - Manual Segmentation")

    image = pygame.transform.scale(
        image, (image.get_width() * scale, image.get_height() * scale)
    )

    screen = pygame.display.set_mode(image.get_size())

    polygons = []
    editing_polygon = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:

                    if not editing_polygon:
                        polygons.append([])
                        editing_polygon = True

                    polygons[-1].append(event.pos)

                if event.button == 3:
                    editing_polygon = False

            if event.type == pygame.KEYDOWN:
                if event.key == 13:
                    path = sg.filedialog.asksaveasfilename(
                        filetypes=(("Segmentation Data", "*.dat"),),
                        defaultextension="dat",
                        title="Save As"
                    )
                    restored_polygons = []
                    for polygon in polygons:
                        restored_polygons.append([
                            (p[1] / scale, p[0] / scale) for p in polygon
                        ])

                    if path != "":
                        with open(path, 'wb') as f:
                            f.write(pickle.dumps(restored_polygons))

                if event.key == 27:
                    if editing_polygon:
                        editing_polygon = False
                        del polygons[-1]

        screen.blit(image, (0, 0))

        for index, locations in enumerate(polygons):
            color = (200, 0, 0) if index != len(polygons) - 1 else (255, 0, 0)
            if len(locations) > 1:
                pygame.draw.lines(screen, color, True, locations, width=3)

            else:
                pygame.draw.circle(screen, color, locations[0], 3)

        pygame.display.update()
