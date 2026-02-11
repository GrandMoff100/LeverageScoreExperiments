from manim import *
from manim_slides import Slide  # type: ignore

from datetime import datetime

from matplotlib.pyplot import title

config.background_color = BLACK  # Set the background color of the slides
Text.set_default(color=BLUE_C)  # Controls the color of text in Manim's Text objects 
Tex.set_default(color=BLUE_C)  # Controls the color of text in LaTeX formulas
MathTex.set_default(color=BLUE_C)  # Affects non-text LaTeX as well, like bullets and formulas
BulletedList.set_default(color=BLUE_C)  # Controls the text color of bulleted lists

HEADER_FONT_SIZE = 36


class MySlide(Slide):
    skip_reversing = True


class MyMovingCameraScene(MovingCameraScene):
    camera: MovingCamera  # type: ignore

class TitleSlide(MySlide, MyMovingCameraScene):
    def construct(self):
        title = Text(
            "Assessing Representation Sensitivity\n"
            "of Leverage Scores in Active Learning",
            font_size=HEADER_FONT_SIZE,
            line_spacing=0.7,
            weight=BOLD,
        )
        line = Line(LEFT * 4, RIGHT * 4).next_to(title, DOWN)

        self.play(Write(title), run_time=4)
        self.play(Create(line))

        confdate = datetime(2026, 2, 28).strftime("%B %-d, %Y")
        today = datetime.now().strftime("%B %-d, %Y")
        subtitle = Text(
            f"Nate Larsen | {today} | Student Research Conference",
            font_size=18,
            color=YELLOW_C,
        )
        subtitle.next_to(title, DOWN, buff=1.5)
        subsubtitle = Text(
            "Advised by Dr. Kevin Miller",
            font_size=14,
            color=YELLOW_C,
        ).next_to(subtitle, DOWN)

        self.play(FadeIn(subtitle, shift=UP))
        self.play(FadeIn(subsubtitle, shift=UP))
        self.next_slide()
        self.play(self.camera.frame.animate.scale(0.01).shift(DOWN * 0.5), run_time=2)


class IntroSlide(MySlide, MyMovingCameraScene):
    def construct(self):
        title = Text("What is Active Learning?", font_size=HEADER_FONT_SIZE, weight=BOLD).to_edge(UP)
        line = Line(LEFT * 2, RIGHT * 2).next_to(title, DOWN)

        body = BulletedList(
            "Active learning is a powerful technique for reducing labeling costs in machine learning.",
            "Leverage scores are a popular method for selecting informative samples in active learning.",
            "However, the sensitivity of leverage scores to representation changes is not well understood.",
            font_size=24,
        )

        body.next_to(title, DOWN, buff=0.5)
        self.play(Create(line), Write(title), run_time=2)
        self.play(self.camera.frame.animate.scale(0.8).shift(UP * 0.5), run_time=1)
        self.next_slide()
        for item in body:
            self.play(Write(item))
            self.next_slide()


class Slide1(MySlide, MyMovingCameraScene):
    def construct(self):
        # Draw some math
        title = Text("Problem Statement", font_size=HEADER_FONT_SIZE, weight=BOLD).to_edge(UP)
        line = Line(LEFT * 2, RIGHT * 2).next_to(title, DOWN)
        subtitle = Text(
            "Given a dataset and a model representation, \n how do leverage scores change when \n the representation is altered?",
            font_size=24,
            color=YELLOW_C,
        ).next_to(line, DOWN, buff=0.5)

        # Draw a math formula
        formula = MathTex(
            r"\text{Leverage Score}(x) = \|P x\|^2", font_size=36
        ).next_to(subtitle, DOWN, buff=0.5)

        self.play(Write(title), run_time=2)
        self.play(Create(line))
        self.next_slide()
        self.play(FadeIn(subtitle, shift=UP))
        self.next_slide()
        self.play(Write(formula))
        self.next_slide()

if __name__ == "__main__":
    print(*(cls.__name__ for cls in MySlide.__subclasses__()))
