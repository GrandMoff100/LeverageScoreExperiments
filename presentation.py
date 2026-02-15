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

        # self.play(Wiggle(title[0][1], scale_value=1.1, rotation_angle=0), run_time=1.5)
        # self.play(Wiggle(title[1][1], scale_value=1.1, rotation_angle=0), run_time=1.5)
        # self.play(Wiggle(title[1][3], scale_value=1.1, rotation_angle=0), run_time=1.5)

        # # self.play(Circumscribe(name, color=WHITE), run_time=2)
        # self.next_slide()
        self.play(self.camera.frame.animate.scale(0.001).shift(DOWN * 0.5), run_time=2)
        self.next_slide()


class IntroSlide(MySlide, MyMovingCameraScene):
    def construct(self):
        title = Text("What is Active Learning?", font_size=HEADER_FONT_SIZE, weight=BOLD).to_edge(UP)
        line = Line(LEFT * 3, RIGHT * 3).next_to(title, DOWN)

        self.play(Write(title), run_time=1)
        self.play(Create(line))

        self.next_slide()

        # Active Learning assumes that labeling data is expensive, and seeks to minimize the number of labeled samples needed to train a model.
        hypothesis = Tex(
            "We assume that labeling data is expensive",
            font_size=36,
            color=YELLOW_C,
        ).next_to(line, DOWN)
        implication = Tex(
            "\\textbf{Goal:} Minimize the number of labeled samples needed to train a model",  # We want to find the most informative samples to label, so we can train an accurate model with fewer labeled examples.
            font_size=36,
            color=YELLOW_C,
        ).next_to(line, DOWN)
        self.play(
            Write(hypothesis),
        )
        self.next_slide()
        self.play(
            Transform(hypothesis, implication)
        )
        self.next_slide()

        linear_regression = VGroup(Text("Linear Regression:", font_size=HEADER_FONT_SIZE, weight=BOLD), Tex("$\\hat{y} \\approx X\\beta$"))
        linear_regression.arrange(RIGHT, buff=0.5)
        linear_regression.to_edge(UP)
        self.play(Transform(title, linear_regression))

        # Fake "large" data matrix
        X = Matrix([
            ["x_{11}", "x_{12}", "\\cdots", "x_{1n}"],
            ["x_{21}", "x_{22}", "\\cdots", "x_{2n}"],
            ["x_{31}", "x_{32}", "\\cdots", "x_{3n}"],
            ["x_{41}", "x_{42}", "\\cdots", "x_{4n}"],
            ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
            ["x_{m1}", "x_{m2}", "\\cdots", "x_{mn}"],
        ])
        X_label = Tex("Data Matrix $X$", font_size=24)
        X_group = VGroup(X, X_label).arrange(DOWN)
        X_group.to_edge(LEFT, buff=2.5)
        X_group.shift(DOWN * 0.75)
        X_group.scale(0.8)

        # Coefficient vector
        beta = Matrix([
            ["\\beta_1"],
            ["\\beta_2"],
            ["\\vdots"],
            ["\\beta_n"]
        ])
        beta_label = Tex("Coefficients $\\beta$", font_size=24)
        beta_group = VGroup(beta, beta_label).arrange(DOWN)
        beta_group.scale(0.8)
        beta_group.next_to(X_group, RIGHT)

        # Equal sign
        equal_sign = Tex("$\\approx$", font_size=45)
        equal_sign.next_to(beta_group, RIGHT, buff=0.25)

        # Label vector
        y = Matrix([
            ["y_1"],
            ["y_2"],
            ["y_3"],
            ["y_4"],
            ["\\vdots"],
            ["y_m"],
        ])
        y_label = Tex("Labels $y$", font_size=24)
        y_group = VGroup(y, y_label).arrange(DOWN)
        y_group.scale(0.8)
        y_group.next_to(equal_sign, RIGHT, buff=0.25)

        self.play(
            FadeIn(X_group),
            FadeIn(beta_group),
            FadeIn(equal_sign),
            FadeIn(y_group)
        )

        self.next_slide()

        # Indices to highlight
        highlight_rows = [1, 3]  # rows 2 and 4 (0-indexed)

        boxes = []

        for r in highlight_rows:
            # Highlight X row
            row = X.get_rows()[r]
            box = SurroundingRectangle(row, color=YELLOW, buff=0.1)
            boxes.append(box)

            # Highlight y entry
            y_entry = y.get_rows()[r]
            y_box = SurroundingRectangle(y_entry, color=RED, buff=0.1)
            boxes.append(y_box)

        # Highlight all beta entries
        beta_boxes = [
            SurroundingRectangle(beta.get_rows()[i], color=BLUE, buff=0.1)
            for i in [0,1,3]
        ]
        boxes.extend(beta_boxes)

        self.play(*[Create(b) for b in boxes])
        self.next_slide()

        # # Emphasize mapping
        # arrows = []
        # for i in range(4):
        #     arrow = Arrow(
        #         X.get_rows()[1][i].get_center(),
        #         beta.get_rows()[i].get_center(),
        #         buff=0.1,
        #         color=GREEN
        #     )
        #     arrows.append(arrow)

        # self.play(*[GrowArrow(a) for a in arrows])
        # self.next_slide()

        # Introduce Leverage Scores
        question = Tex(
            "How do we pick which samples we should use in our model?",
            font_size=28,
            color=YELLOW_C,
        ).to_edge(DOWN)
        self.play(Write(question))
        self.next_slide()

        # Reduce matrix and labels to just highlighted rows
        reduced_X = Matrix([
            ["x_{21}", "x_{22}", "\\cdots", "x_{2n}"],
            ["x_{41}", "x_{42}", "\\cdots", "x_{4n}"],
        ]).scale(0.8)
        reduced_X.next_to(X_group, DOWN, buff=1.5)
        reduced_y = Matrix([
            ["y_2"],
            ["y_4"],
        ]).scale(0.8)
        reduced_y.next_to(reduced_X, RIGHT, buff=0.25)
        self.play(
            Transform(X, reduced_X),
            Transform(y, reduced_y)
        )
        self.next_slide()

        # Clean up
        self.play(
            FadeOut(VGroup(*boxes)),
            # FadeOut(VGroup(*arrows)),
        )
        self.next_slide()

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
