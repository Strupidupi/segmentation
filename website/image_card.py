import flet
from flet import UserControl, Text, Container, Image, Column, Card


class ImageCard(UserControl):
    def __init__(self, image_path, title, width, height):
        super().__init__()
        self.image_path = image_path
        self.title = title
        self.img_width = width
        self.img_height = height

    def build(self):
        return Card(
			content=Container(
				content=Column(
					[
						Text(
							self.title,
							text_align="center",
                            weight="bold"
						),
						Image(
							src=self.image_path,
							width=self.img_width,
							height=self.img_height,
						),
					],
					horizontal_alignment="center",
				),
				padding=20,
			),
		)