import subprocess
import pygame
import os
class Mdog_Owner_system():
    def __init__(self):
        pass
    def train(self):
        pass
    def predict(
            self,
            model_path: str = 'yolov5s.pt',
            image_path: str = 'data/images',
            resize_img: int = 640,
            image_size: tuple = (1920,1080),
            output_path: str = 'test',
            owner_id: int = 0,
            show: bool = False,
        ):
        args = (
            f'--weights {model_path} '
            f'--source {image_path} '
            f'--data data/coco128.yaml '
            f'--imgsz {resize_img} '
            f'--name output '
            f'--project {output_path} '
            f'--save-txt '
            f'--max-det 1 '
            f'--classes {owner_id} '
            f'--conf 0.7 '
            f'--hide-conf '
            f'--exist-ok '
        )
        if show:
            pygame.init()
            w,h = image_size
            screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Mdog Image")
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                subprocess.run(f'python3 detect.py {args}',shell=True)
                image = pygame.image.load('output/test2.jpg')
                if image: screen.blit(image, (0, 0))
                pygame.display.flip()
            pygame.quit()

        else:
            subprocess.run(f'python3 detect.py {args}',shell=True)



mdog = Mdog_Owner_system()

mdog.predict(
    model_path='best2.pt',
    image_path='test_img/test2.jpg',
    resize_img=640,
    image_size=(1920,1080),
    output_path='./',
    owner_id=1,
    show=True,
    )