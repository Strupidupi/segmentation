import flet
from flet import UserControl, Row, Container, Column, icons, colors, IconButton, margin
from image_card import ImageCard

paths = [
    {
        "ori": "/images/pathology/Bild1/original.jpg",
        "result": "/images/pathology/Bild1/output.jpg",
        "mask": "/images/pathology/Bild1/label.jpg",
    },
    {
        "ori": "/images/pathology/Bild2/original.jpg",
        "result": "/images/pathology/Bild2/output.jpg",
        "mask": "/images/pathology/Bild2/label.jpg",
    },
    {
        "ori": "/images/pathology/Bild3/original.jpg",
        "result": "/images/pathology/Bild3/output.jpg",
        "mask": "/images/pathology/Bild3/label.jpg",
    }
]

class PathologyContainer(UserControl):
    def __init__(self):
        super().__init__()
        self.clicked = False

    def build(self):
        self.clicked = False

        sampleOriginalCards = [ImageCard(paths[i]["ori"], "Originalbild", 384, 150) for i in range(3)]
        sampleResultCards = [ImageCard(paths[i]["result"], "Unser Ansatz", 384, 150) for i in range(3)]
        sampleMaskCards = [ImageCard(paths[i]["mask"], "Vorlage", 384, 150) for i in range(3)]


        def on_result(e):
            if self.clicked: return
            self.clicked = True

            for i, con in enumerate(self.content.controls):
                con.controls.append(sampleMaskCards[i])
                con.controls.append(sampleResultCards[i])
            
            button.disabled = True

            self.update()

        button = IconButton(
            icon=icons.ARROW_RIGHT,
            icon_color=colors.GREEN_100,
            icon_size=48,
            tooltip="Erkennen",
            on_click=on_result,
            disabled=False
        )

        first_sample = Row(
            [
                sampleOriginalCards[0],
                Container(width=64)
            ],
            vertical_alignment="center",
        )

        second_sample = Row(
            [
                sampleOriginalCards[1],
                button
            ],
            vertical_alignment="center"
        )

        third_sample = Row(
            [
                sampleOriginalCards[2],
                Container(width=64)
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