import flet
from flet import UserControl, Row, Container, Column, icons, colors, IconButton, AlertDialog, Text, Stack, alignment, Image, transform, animation, AnimationCurve
from image_card import ImageCard
from math import pi

paths = [
    {
        "ori": "/images/shapes/sample1.jpg",
        "result": "/images/shapes/sample1 - result.jpg",
        "mask": "/images/shapes/sample1 - mask.jpg",
    },
    {
        "ori": "/images/shapes/sample2.jpg",
        "result": "/images/shapes/sample2 - result.jpg",
        "mask": "/images/shapes/sample2 - mask.jpg",
    },
    {
        "ori": "/images/shapes/sample3.jpg",
        "result": "/images/shapes/sample3 - result.jpg",
        "mask": "/images/shapes/sample3 - mask.jpg",
    }
]

class ShapesContainer(UserControl):
    def __init__(self, page):
        super().__init__()
        self.clicked = False
        self.page

    def build(self):
        self.clicked = False
        
        sampleOriginalCards = [ImageCard(paths[i]["ori"], "Originalbild", 292, 150) for i in range(3)]
        sampleResultCards = [ImageCard(paths[i]["result"], "Unser Ansatz", 292, 150) for i in range(3)]
        sampleMaskCards = [ImageCard(paths[i]["mask"], "Vorlage", 292, 150) for i in range(3)]

        image1 = Image(
            "/images/Heart_alpha.png",
            width=150,
            height=150
        )

        image2 = Image(
            "/images/Heart_alpha.png",
            width=150,
            height=150,
            rotate=transform.Rotate(-pi / 2.0, alignment=alignment.center),
            animate_rotation=animation.Animation(4000, AnimationCurve.LINEAR_TO_EASE_OUT),
        )
        rotation_container = Container(
            content=Stack(
                [
                    image1,
                    image2
                ] 
            ),
            alignment=alignment.center,
            width=150,
            height=150,
            visible=False
        )

        def on_result(e):
            if self.clicked: return
            self.clicked = True

            for i, con in enumerate(self.content.controls):
                con.controls.append(sampleMaskCards[i])
                con.controls.append(sampleResultCards[i])            
            
            second_sample.controls.append(self.rotation_button)
            second_sample.controls.append(rotation_container)

            button.disabled = True

            self.update()
            
        def on_rotate(e):
            rotation_container.visible = True
            image2.rotate.angle += pi * 1.25
            self.update()

        button = IconButton(
            icon=icons.ARROW_RIGHT,
            icon_color=colors.GREEN_100,
            icon_size=48,
            tooltip="Erkennen",
            on_click=on_result,
            disabled=False
        )

        self.rotation_button = IconButton(
            icon=icons.ROTATE_RIGHT,
            icon_color=colors.GREEN_100,
            icon_size=48,
            tooltip="Rotation",
            on_click=on_rotate,
            disabled=False
        )

        first_sample = Row(
            [
                sampleOriginalCards[0],
                Container(width=64),
            ],
            vertical_alignment="center"
        )

        second_sample = Row(
            [
                sampleOriginalCards[1],
                button,
            ],
            vertical_alignment="center"
        )

        third_sample = Row(
            [
                sampleOriginalCards[2],
                Container(width=64),
            ],
            vertical_alignment="center"
        )

        self.content = Column(
            [
                first_sample,
                second_sample,
                third_sample
            ],
            horizontal_alignment="start",
            spacing=20
        )

        return Container(self.content)