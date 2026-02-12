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
        # title = Text(
        #     "Assessing Representation Sensitivity\n"
        #     "of Leverage Scores in Active Learning",
        #     font_size=HEADER_FONT_SIZE,
        #     line_spacing=0.7,
        #     weight=BOLD,
        #     t2c={"Representation Sensitivity": ORANGE, "Leverage Scores": YELLOW, "Active Learning": GREEN}, # type: ignore
        # )
        title = VGroup(
            VGroup(
                Text("Assessing", font_size=HEADER_FONT_SIZE, weight=BOLD),
                Text("Representation Sensitivity", font_size=HEADER_FONT_SIZE, weight=BOLD, color=RED),
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Text("of", font_size=HEADER_FONT_SIZE, weight=BOLD),
                Text("Leverage Scores", font_size=HEADER_FONT_SIZE, weight=BOLD, color=YELLOW),
                Text("in", font_size=HEADER_FONT_SIZE, weight=BOLD),
                Text("Active Learning", font_size=HEADER_FONT_SIZE, weight=BOLD, color=GREEN),
            ).arrange(RIGHT, buff=0.2)
        ).arrange(DOWN, buff=0.25)
        for m in title[1]:
            m.align_to(title[1][0], UP)

        line = Line(LEFT * 4, RIGHT * 4).next_to(title, DOWN)

        self.play(Write(title), run_time=4)
        self.play(Create(line))
    
        confdate = datetime(2026, 2, 28).strftime("%B %-d, %Y")
        name, _, date, _, event = subtitle = VGroup(
            Text("Nate Larsen", font_size=18, color=YELLOW_C),
            Text("|", font_size=18, color=YELLOW_C),
            Text(confdate, font_size=18, color=YELLOW_C),
            Text("|", font_size=18, color=YELLOW_C),
            Text("Student Research Conference", font_size=18, color=YELLOW_C)
        ).arrange(RIGHT, buff=0.2)
        subtitle.next_to(title, DOWN, buff=1.5)
        subsubtitle = Text(
            "Advised by Dr. Kevin Miller",
            font_size=14,
            color=YELLOW_C,
        ).next_to(subtitle, DOWN)

        self.play(FadeIn(subtitle, shift=UP))
        self.play(FadeIn(subsubtitle, shift=UP))
        self.next_slide()

        self.play(Wiggle(title[0][1], scale_value=1.1, rotation_angle=0), run_time=1.5)
        self.play(Wiggle(title[1][1], scale_value=1.1, rotation_angle=0), run_time=1.5)
        self.play(Wiggle(title[1][3], scale_value=1.1, rotation_angle=0), run_time=1.5)

        # self.play(Circumscribe(name, color=WHITE), run_time=2)
        self.next_slide()
        self.play(self.camera.frame.animate.scale(0.01).shift(DOWN * 0.5), run_time=2)
        self.next_slide()


class IntroSlide(MySlide, MyMovingCameraScene):
    def construct(self):
        title = Text("What is Active Learning?", font_size=HEADER_FONT_SIZE, weight=BOLD).to_edge(UP)
        line = Line(LEFT * 2, RIGHT * 2).next_to(title, DOWN)

        # Active Learning assumes that labeling data is expensive, and seeks to minimize the number of labeled samples needed to train a model.
        # <Animate Linear Regression diagram>
        # Leverage scores are a common method for selecting informative samples in active learning, but they can be sensitive to changes in the model's representation.
        # Question 1: How sensitive are leverage scores to changes in the data's representation? (neural network pov)
        # Question 2: How does this sensitivity impact the performance of active learning algorithms that rely on leverage scores for sample selection?
        

# class Slide1(MySlide, MyMovingCameraScene):
#     def construct(self):
#         # Draw some math
#         title = Text("Problem Statement", font_size=HEADER_FONT_SIZE, weight=BOLD).to_edge(UP)
#         line = Line(LEFT * 2, RIGHT * 2).next_to(title, DOWN)
#         subtitle = Text(
#             "Given a dataset and a model representation, \n how do leverage scores change when \n the representation is altered?",
#             font_size=24,
#             color=YELLOW_C,
#         ).next_to(line, DOWN, buff=0.5)

#         # Draw a math formula
#         formula = MathTex(
#             r"\text{Leverage Score}(x) = \|P x\|^2", font_size=36
#         ).next_to(subtitle, DOWN, buff=0.5)

#         self.play(Write(title), run_time=1)
#         self.play(Create(line))
#         self.next_slide()
#         self.play(FadeIn(subtitle, shift=UP))
#         self.next_slide()
#         self.play(Write(formula))
#         self.next_slide()

if __name__ == "__main__":
    print(*(cls.__name__ for cls in MySlide.__subclasses__()))
