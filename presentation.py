from manim import *
from manim_slides import Slide, ThreeDSlide  # type: ignore

from datetime import datetime

from matplotlib.pyplot import title

config.background_color = BLACK  # Set the background color of the slides
Text.set_default(color=BLUE_C)  # Controls the color of text in Manim's Text objects
Tex.set_default(color=BLUE_C)  # Controls the color of text in LaTeX formulas
MathTex.set_default(
    color=BLUE_C
)  # Affects non-text LaTeX as well, like bullets and formulas
BulletedList.set_default(color=BLUE_C)  # Controls the text color of bulleted lists

HEADER_FONT_SIZE = 36


class MySlide(Slide):
    skip_reversing = True


class MyThreeDSlide(ThreeDSlide):
    skip_reversing = True


class MyMovingCameraScene(MovingCameraScene):
    camera: MovingCamera  # type: ignore


class TitleSlide(MySlide, MyMovingCameraScene):
    def construct(self):
        title = VGroup(
            VGroup(
                Text(
                    "Representation Sensitivity",
                    font_size=HEADER_FONT_SIZE,
                    weight=BOLD,
                    color=RED,
                ),
                Text("of", font_size=HEADER_FONT_SIZE, weight=BOLD),
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Text(
                    "Leverage Scores",
                    font_size=HEADER_FONT_SIZE,
                    weight=BOLD,
                    color=YELLOW,
                ),
                Text("in", font_size=HEADER_FONT_SIZE, weight=BOLD),
                Text(
                    "Active Learning",
                    font_size=HEADER_FONT_SIZE,
                    weight=BOLD,
                    color=GREEN,
                ),
            ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.25)
        for m in title[1]:
            m.align_to(title[1][0], UP)

        line = Line(LEFT * 4, RIGHT * 4).next_to(title, DOWN)

        self.play(Write(title), run_time=4)
        self.play(Create(line))

        confdate = datetime(2026, 2, 28).strftime("%B %-d, %Y")
        name = Text("Nate Larsen | BYU Student Research Conference", font_size=18, color=YELLOW_C)
        name.next_to(title, DOWN, buff=1.5)
        subsubtitle = Text(
            "Advised by Dr. Kevin Miller",
            font_size=14,
            color=YELLOW_C,
        ).next_to(name, DOWN)

        self.play(FadeIn(name))
        self.play(FadeIn(subsubtitle))
        self.next_slide()

        self.play(self.camera.frame.animate.scale(0.001).shift(DOWN * 0.75), run_time=2)


class IntroSlide(MySlide, MyMovingCameraScene):
    def construct(self):
        title = Text(
            "What is Active Learning?",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
            t2c={"Active Learning": GREEN},
        ).to_edge(UP)
        line = Line(LEFT * 3, RIGHT * 3).next_to(title, DOWN)

        self.play(Write(title), run_time=1)
        self.play(Create(line))

        # Active Learning assumes that labeling data is expensive, and seeks to minimize the number of labeled samples needed to train a model.
        hypothesis = Tex(
            "Often labeling data (making measurements, classifying things, etc.) is expensive",
            font_size=30,
            color=YELLOW_C,
        ).next_to(line, DOWN)
        implication = Tex(
            "\\textbf{Goal:} Minimize the number of labeled samples needed to train a model",  # We want to find the most informative samples to label, so we can train an accurate model with fewer labeled examples.
            font_size=30,
            color=YELLOW_C,
        ).next_to(line, DOWN)
        self.play(
            Write(hypothesis),
        )
        self.next_slide()
        self.play(Transform(hypothesis, implication))
        self.next_slide()

        linear_regression = VGroup(
            Text(
                "Active Linear Regression:",
                font_size=HEADER_FONT_SIZE,
                weight=BOLD,
                t2c={"Active": GREEN},
            ),
            Tex("$\\hat{y} \\approx X\\beta$"),
        )
        linear_regression.arrange(RIGHT, buff=0.5)
        linear_regression.to_edge(UP)

        # Label vector
        y = Matrix(
            [
                ["y_1"],
                ["y_2"],
                ["y_3"],
                ["y_4"],
                ["\\vdots"],
                ["y_m"],
            ]
        )
        y.scale(0.8)
        y.to_edge(LEFT, buff=3)
        y.shift(DOWN * 0.5)
        y_label = Tex("Labels $\\hat{y}$", font_size=24)
        y_label.scale(0.8)
        y_label.next_to(y, DOWN, buff=0.25)

        # Equal sign
        equal_sign = Tex("$\\approx$", font_size=45)
        equal_sign.next_to(y, RIGHT, buff=0.25)

        # Fake "large" data matrix
        X = Matrix(
            [
                ["x_{11}", "x_{12}", "\\cdots", "x_{1n}"],
                ["x_{21}", "x_{22}", "\\cdots", "x_{2n}"],
                ["x_{31}", "x_{32}", "\\cdots", "x_{3n}"],
                ["x_{41}", "x_{42}", "\\cdots", "x_{4n}"],
                ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                ["x_{m1}", "x_{m2}", "\\cdots", "x_{mn}"],
            ]
        )
        X.scale(0.8)
        X.next_to(equal_sign, RIGHT, buff=0.25)

        X_label = Tex("Data Matrix $X$", font_size=24)
        X_label.scale(0.8)
        X_label.next_to(X, DOWN, buff=0.25)

        # Coefficient vector
        beta = Matrix([["\\beta_1"], ["\\beta_2"], ["\\vdots"], ["\\beta_n"]])
        beta_label = Tex("Coefficients $\\beta$", font_size=24)
        beta.scale(0.8)
        beta.next_to(X, RIGHT, buff=0.25)
        beta_label.scale(0.8)
        beta_label.next_to(beta, DOWN, buff=0.25)

        self.play(
            Transform(title, linear_regression),
            FadeIn(X),
            FadeIn(X_label),
            FadeIn(beta),
            FadeIn(beta_label),
            FadeIn(equal_sign),
            FadeIn(y),
            FadeIn(y_label),
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

        self.play(*[Create(b) for b in boxes])
        self.next_slide()

        # Reduce matrix and labels to just highlighted rows
        reduced_X = Matrix(
            [
                ["x_{21}", "x_{22}", "\\cdots", "x_{2n}"],
                ["x_{41}", "x_{42}", "\\cdots", "x_{4n}"],
            ]
        ).scale(0.8)
        reduced_X.move_to(X.get_center())  # Move to the center of the highlighted rows
        reduced_y = Matrix(
            [
                ["y_2"],
                ["y_4"],
            ]
        ).scale(0.8)
        reduced_y.move_to(
            y.get_center()
        )  # Move to the center of the highlighted entries

        # Transform the highlight boxes into the reduced matrices
        new_highlight_boxes = []
        for i, r in enumerate(highlight_rows):
            # Transform X row box
            new_box = SurroundingRectangle(
                reduced_X.get_rows()[i], color=YELLOW, buff=0.1
            )
            new_highlight_boxes.append(new_box)

            # Transform y entry box
            new_y_box = SurroundingRectangle(
                reduced_y.get_rows()[i], color=RED, buff=0.1
            )
            new_highlight_boxes.append(new_y_box)

        self.play(
            Transform(X, reduced_X),
            Transform(y, reduced_y),
            *[
                Transform(box, new_box)
                for box, new_box in zip(boxes, new_highlight_boxes)
                if box in boxes
            ],
        )
        self.next_slide()

        # Clean up
        self.play(
            FadeOut(VGroup(*boxes)),
        )
        self.next_slide()


def compute_leverage_scores(X):
    Q, _ = np.linalg.qr(X)
    return np.sum(Q**2, axis=1)


class LeverageScores(MySlide, MyMovingCameraScene):
    def construct(self):
        # Leverage scores are a common method for selecting informative samples in active learning, but they can be sensitive to changes in the model's representation.
        title = Text(
            "Leverage Scores",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
            t2c={"Leverage Scores": YELLOW},
        ).to_edge(UP)
        self.play(Write(title), run_time=1)

        description = Tex(
            'Leverage scores measure how much ``leverage" each data point has on the fitted model',
            font_size=28,
        ).next_to(title, DOWN)
        self.play(Write(description))
        self.next_slide()

        # Define Leverage Scores
        definition = MathTex(
            "\\text{Leverage Score}(x_i) = x_i^T (X^T X)^{-1} x_i = \\|Q^T e_i\\|^2 \\quad \\text{(Classic Definition)}",
            font_size=28,
            color=YELLOW,
        ).next_to(description, DOWN)
        self.play(Write(definition))
        self.next_slide()

        alternate_description = Tex(
            "Leverage scores also describe how easily data points can be reconstructed by other points in the dataset! (minimal norm reconstruction)",
            font_size=28,
        ).next_to(definition, DOWN, buff=1)
        self.play(Write(alternate_description))
        alternate_definition = MathTex(
            "\\text{Leverage Score}(x_i) = \\min_{\\mathbf{c}\\in \\mathbb{R}^n} \\|\\mathbf{c}\\|^2 \\text{ such that } X^T \\mathbf{c} = x_i^T \\quad \\text{(Interpretable Definition)}",
            font_size=28,
            color=YELLOW,
        ).next_to(alternate_description, DOWN)
        self.play(Write(alternate_definition))
        self.next_slide()

        # Quick example of leverage scores for a simple dataset
        X = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 10],
            ]
        )
        matrix = Matrix(X).scale(0.7).to_edge(DOWN + LEFT, buff=1)
        arrow = MathTex("\\Longrightarrow").scale(1).next_to(matrix, RIGHT, buff=0.5)
        arrow_label = Tex("Leverage Scores", font_size=18).next_to(arrow, UP * 0.5)
        scores = Matrix(map(lambda x: np.round(x, 2), compute_leverage_scores(X)[:,None])).scale(0.7).next_to(arrow, RIGHT, buff=0.5)
        self.play(FadeIn(matrix))
        self.next_slide()

        # Animate minimal norm reconstruction of the third point using the first two points
        reconstruction = MathTex(
            "\\left(0\\right) \\begin{bmatrix} 1 \\\\ 0 \\end{bmatrix} + (1)\\begin{bmatrix} 0 \\\\ 1 \\end{bmatrix} + (0)\\begin{bmatrix} 1 \\\\ 10 \\end{bmatrix} = \\begin{bmatrix} 0 \\\\ 1 \\end{bmatrix} \\implies \\mathbf{c} = \\begin{bmatrix} 0 \\\\ 1 \\\\ 0 \\end{bmatrix} \\text{ and } \\|\\mathbf{c}\\|^2 = 1",
            font_size=24,
        ).next_to(scores, RIGHT, buff=0.5)
        better_reconstruction = MathTex(
            "\\left(-0.1\\right) \\begin{bmatrix} 1 \\\\ 0 \\end{bmatrix} + (0)\\begin{bmatrix} 0 \\\\ 1 \\end{bmatrix} + (0.1)\\begin{bmatrix} 1 \\\\ 10 \\end{bmatrix} = \\begin{bmatrix} 0 \\\\ 1 \\end{bmatrix} \\implies \\mathbf{c} = \\begin{bmatrix} -0.1 \\\\ 0 \\\\ 0.1 \\end{bmatrix} \\text{ and } \\|\\mathbf{c}\\|^2 = 0.02",
            font_size=24,
        ).next_to(scores, RIGHT, buff=0.5)

        self.play(Write(reconstruction))
        self.next_slide()

        self.play(Transform(reconstruction, better_reconstruction))
        self.next_slide()

        self.play(FadeIn(arrow), FadeIn(arrow_label), FadeIn(scores))
        self.next_slide()

        # Draw example of leverage scores in 1D


class BigTheme(MyThreeDSlide):
    def construct(self):
        # Introduce neural networks as non-linear representations

        ## Do leverage scores remain a reliable measure of sample importance when we change the representation of our data, such as by using a neural network to learn a non-linear embedding?
        title = Text(
            "Big Idea",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
            t2c={"Big Idea": RED},
        ).to_edge(UP)
        self.play(Write(title), run_time=1)
        self.next_slide()

        ## Question 1: How sensitive are leverage scores to changes in the data's representation? (neural network pov)

        ## Do they give us important samples to train on?

        ## Question 2: How does this sensitivity impact the performance of active learning algorithms that rely on leverage scores for sample selection?

        questions = BulletedList(
            "How sensitive are leverage scores to changes in the data's representation? (neural networks)",
            "Do they give us important samples to train on?",
            font_size=24,
        ).next_to(title, DOWN)
        for q in questions:
            self.play(Write(q))
            self.next_slide()

class Question1(MySlide, MyMovingCameraScene):
    def construct(self):
        title = Text(
            "Question 1",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
            t2c={"Question 1": RED},
        ).to_edge(UP)
        self.play(Write(title), run_time=1)


class Question2(MySlide, MyMovingCameraScene):
    def construct(self):
        title = Text(
            "Question 2",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
            t2c={"Question 2": RED},
        ).to_edge(UP)
        self.play(Write(title), run_time=1)


class Conclusion(MySlide, MyMovingCameraScene):
    def construct(self):
        title = Text(
            "Conclusion",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
            t2c={"Conclusion": RED},
        ).to_edge(UP)
        self.play(Write(title), run_time=1)


if __name__ == "__main__":
    print(
        "TitleSlide",
        "IntroSlide",
        "LeverageScores",
        "BigTheme",
        "Question1",
        "Question2",
        "Conclusion",
    )
