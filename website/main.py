import flet
from flet import Page, Row, AppBar, Text, colors, Dropdown, dropdown, Container, TextThemeStyle, alignment
from patholoy import PathologyContainer
from shapes import ShapesContainer

PATHOLOGY_KEY = "Zellen"
SHAPES_KEY = "Formen + Orientierungen"

def main(page: Page):
    page.title = "Dashboard"
    page.vertical_alignment = "start"
    page.theme_mode = "light"
    page.padding = 20
    page.window_full_screen = True

    container_pathology = PathologyContainer()

    container_shapes = ShapesContainer(page)

    def dropdown_changed(e):
        if e.control.value == SHAPES_KEY:
            page.remove(container_pathology)
            page.add(container_shapes)
        else:
            page.remove(container_shapes)
            page.add(container_pathology)
        page.update()
    
    leading = Row(
        [
            Container(width=20,height=70),
            Text("Erkennung von", style=TextThemeStyle.TITLE_LARGE),
            Dropdown(
                width=300,
                height=56,
                options=[
                    dropdown.Option(PATHOLOGY_KEY),
                    dropdown.Option(SHAPES_KEY),
                ],
                value=PATHOLOGY_KEY,
                on_change=dropdown_changed,
                text_size=20,
                alignment=alignment.center_left
            ),
        ],
        vertical_alignment=alignment.center
    )
        
    page.appbar = AppBar(
        leading=leading,
        bgcolor=colors.LIGHT_GREEN_600,
        leading_width=600,
        toolbar_height=70,
        center_title=False
    )
    page.add(container_pathology)

flet.app(
    target=main,
    assets_dir="assets"
)