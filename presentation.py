import torch
from manim import *  # type: ignore
from manim_slides import Slide  # type: ignore

from mnist_embeddings import MnistConvNet
from datetime import datetime

from manim_ml.neural_network import NeuralNetwork, FeedForwardLayer
import numpy as np

from matplotlib import pyplot as plt

config.background_color = BLACK  # Set the background color of the slides
Text.set_default(color=WHITE)  # Controls the color of text in Manim's Text objects
Tex.set_default(color=WHITE)  # Controls the color of text in LaTeX formulas
MathTex.set_default(
    color=WHITE
)  # Affects non-text LaTeX as well, like bullets and formulas
BulletedList.set_default(color=WHITE)  # Controls the text color of bulleted lists

HEADER_FONT_SIZE = 36
SUBHEADER_FONT_SIZE = 28

__all__ = (
    "TitleSlide",
    "MNISTSlide",
    "IntroSlide",
    "LeverageScores",
    "LeverageScoresOnMNIST",
    "AlternativeRepresentations",
    "LeverageScoresOfNNRepresentation",
    "LeverageScoresForCoreSetSelection",
    "KeyFindings",
    "ThankYouSlide",
)


import torch
from torchvision.datasets import MNIST

MNIST_TRAIN = MNIST(
    root="~/Desktop/AliasingOperatorExperiments/data", train=True, download=True
)
MNIST_TEST = MNIST(
    root="~/Desktop/AliasingOperatorExperiments/data", train=False, download=True
)

mnist_X = MNIST_TRAIN.data.float().reshape(-1, 1, 28, 28) / 255.0
mnist_y = MNIST_TRAIN.targets
test_mnist_X = MNIST_TEST.data.float().reshape(-1, 1, 28, 28) / 255.0
test_mnist_y = MNIST_TEST.targets


def compute_leverage_scores(X):
    Q, _ = np.linalg.qr(X)
    return np.sum(Q**2, axis=1)


def embed_dataset(X, model, device, basics_functions: int):
    # Embed the data using the convolutional layers of the network
    embeddings = torch.tensor(
        np.zeros(
            (
                X.shape[0],
                basics_functions,
            )
        )
    ).to(device)

    with torch.no_grad():
        for batch_start in range(0, X.shape[0], 256):
            batch_end = min(batch_start + 256, X.shape[0])
            batch = X[batch_start:batch_end].to(device)
            batch_embeddings = model.embed(batch)
            embeddings[batch_start:batch_end] = batch_embeddings
    return embeddings


leverage_scores = compute_leverage_scores(mnist_X.reshape(60000, 28 * 28).numpy())


class MySlide(Slide):
    skip_reversing = True


class TitleSlide(MySlide):
    def construct(self):
        title = VGroup(
            VGroup(
                Text("Assessing", font_size=HEADER_FONT_SIZE, weight=BOLD),
                Text(
                    "Representation Sensitivity",
                    font_size=HEADER_FONT_SIZE,
                    weight=BOLD,
                    color=RED,
                ),
            ).arrange(RIGHT, buff=0.2),
            VGroup(
                Text("of", font_size=HEADER_FONT_SIZE, weight=BOLD),
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
        for row in title:
            for m in row:
                m.align_to(row[0], UP)

        line = Line(LEFT * 4, RIGHT * 4).next_to(title, DOWN)

        confdate = datetime(2026, 2, 28).strftime("%B %-d, %Y")
        name = Text(
            "Nate Larsen | BYU Student Research Conference",
            font_size=18,
        )
        name.next_to(title, DOWN, buff=1.5)
        subsubtitle = Text(
            "Advised by Dr. Kevin Miller",
            font_size=14,
        ).next_to(name, DOWN)

        self.play(
            FadeIn(title), FadeIn(line), FadeIn(name), FadeIn(subsubtitle), run_time=1
        )
        self.next_slide()


class MNISTSlide(MySlide):
    def construct(self):
        # Introduce MNIST dataset
        title = Text(
            "Our data...",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
        ).to_edge(UP)
        digits = Group(
            *[
                ImageMobject(
                    (mnist_X[i].numpy().reshape(28, 28) * 255).astype(np.uint8)
                ).scale(5)
                for i in range(16)
            ]
        )
        digits.scale(0.7).arrange_in_grid(rows=4, cols=4, buff=0.1).next_to(
            title, DOWN, buff=1
        ).to_edge(LEFT, buff=1)
        label_top = Tex("MNIST Dataset", font_size=24).next_to(digits, UP)
        label_bottom = Tex(
            """
        \\begin{itemize}
        \\item 60,000 ($28 \\times 28$) handwritten digits
        \\item 10,000 additional digits for testing
        \\item 10 classes (0-9)
        \\item \\textbf{One Hot Encoded Labels}
        \\end{itemize}
        """,
            font_size=24,
        ).next_to(digits, DOWN)

        arrow1 = MathTex("\\rightarrow").scale(1).next_to(digits, RIGHT, buff=0.25)
        digit = (
            Matrix(
                (mnist_X[0].numpy().reshape(28, 28) * 255).astype(np.uint8),
                left_bracket="(",
                right_bracket=")",
                h_buff=1,
            )
            .scale(0.20)
            .next_to(arrow1, RIGHT, buff=0.25)
        )

        arrow2 = MathTex("\\rightarrow").scale(1).next_to(digit, RIGHT, buff=0.25)
        flat = (mnist_X[0].numpy().reshape(-1) * 255).astype(np.uint8)
        flattened_digit = (
            MathTex(
                f"""
                \\underbrace{{
                    \\begin{{bmatrix}}
                    {' & '.join([str(e) for e in flat[:4]] + ['\\cdots'] + [str(e) for e in flat[-4:]])}
                    \\end{{bmatrix}}
                }}
                """
            )
            .scale(0.25)
            .next_to(arrow2, RIGHT, buff=0.25)
        )
        pixel_count = Tex("784 pixels", font_size=14).next_to(flattened_digit, DOWN)
        flattened_digit_label = Tex("Flattened Digit Vector", font_size=18).next_to(
            flattened_digit, UP
        )

        one_hot_label = Tex("One Hot Encoded Label (5)", font_size=18).next_to(
            flattened_digit, DOWN, buff=0.5
        )
        one_hot = (
            torch.nn.functional.one_hot(mnist_y[0:1], num_classes=10)
            .numpy()
            .reshape(-1)
        )
        one_hot_vector = (
            Matrix(one_hot[None, :], left_bracket="(", right_bracket=")", h_buff=0.8)
            .scale(0.25)
            .next_to(one_hot_label, DOWN)
        )
        index_label = MathTex(
            "\\overbrace{\\text{ Index 5 } }_{}", font_size=14
        ).next_to(one_hot_vector.get_columns()[5], DOWN * 0.3)

        self.play(
            FadeIn(title),
            FadeIn(digits),
            FadeIn(label_top),
            FadeIn(label_bottom),
            FadeIn(arrow1),
            FadeIn(digit),
            FadeIn(arrow2),
            FadeIn(flattened_digit),
            FadeIn(flattened_digit_label),
            FadeIn(pixel_count),
            FadeIn(one_hot_vector),
            FadeIn(one_hot_label),
            FadeIn(index_label),
        )
        self.next_slide()


class IntroSlide(MySlide):
    def construct(self):
        title = Text(
            "Active Learning",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
            t2c={"Active Learning": GREEN},  # pyright: ignore[reportArgumentType]
        ).to_edge(UP + LEFT)

        # Active Learning assumes that labeling data is expensive, and seeks to minimize the number of labeled samples needed to train a model.
        hypothesis, implication = (
            BulletedList(
                "Often it is expensive to label data (making measurements, classifying things, etc.)",
                "\\textbf{Goal:} Minimize the number of labeled samples needed",
                font_size=24,
            )
            .arrange(DOWN, aligned_edge=LEFT)
            .next_to(title, DOWN)
            .to_edge(LEFT, buff=1)
        )

        linear_regression = VGroup(
            Text(
                "Linear Regression:",
                font_size=SUBHEADER_FONT_SIZE,
                weight=BOLD,
            ),
            Tex("$\\hat{y} \\approx X\\beta$"),
        )
        linear_regression.arrange(RIGHT, buff=0.5)
        linear_regression.next_to(implication, DOWN, buff=1).to_edge(LEFT, buff=1)

        SCALE = 0.6

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
        y.scale(SCALE)
        y.next_to(linear_regression, DOWN, buff=0.5).to_edge(LEFT, buff=1.5)
        # y.shift(DOWN * 0.5)
        y_label = Tex("\\textbf{Labels $\\hat{y}$}", font_size=24)
        y_label.scale(SCALE)
        y_label.next_to(y, DOWN, buff=0.25)

        # Equal sign
        equal_sign = Tex("$\\approx$", font_size=45)
        equal_sign.next_to(y, RIGHT, buff=0.5)

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
        X.scale(SCALE)
        X.next_to(equal_sign, RIGHT, buff=0.5)
        columns_label = Tex("\\textit{Features}", font_size=18).next_to(
            X.get_rows()[0], UP, buff=0.2
        )
        rows_label = (
            Tex("\\textit{Data Points}", font_size=18)
            .next_to(X.get_columns()[0], LEFT, buff=0)
            .rotate(PI / 2)
        )

        X_label = Tex("\\textbf{Data Matrix $X$}", font_size=24)
        X_label.scale(SCALE)
        X_label.next_to(X, DOWN, buff=0.25)

        # Coefficient vector
        beta = Matrix([["\\beta_1"], ["\\beta_2"], ["\\vdots"], ["\\beta_n"]])
        beta_label = Tex("\\textbf{Coefficients $\\beta$}", font_size=24)
        beta.scale(SCALE)
        beta.next_to(X, RIGHT, buff=0.25)
        beta_label.scale(SCALE)
        beta_label.next_to(beta, DOWN, buff=0.25)

        self.play(
            FadeIn(title),
            FadeIn(hypothesis),
            FadeIn(implication),
        )
        self.next_slide()

        self.play(
            FadeIn(linear_regression),
            FadeIn(X),
            FadeIn(columns_label),
            FadeIn(rows_label),
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
            box = SurroundingRectangle(row, color=PINK, buff=0.1)
            boxes.append(box)

            # Highlight y entry
            y_entry = y.get_rows()[r]
            y_box = SurroundingRectangle(y_entry, color=PINK, buff=0.1)
            boxes.append(y_box)

        self.play(*[Create(b) for b in boxes])
        self.next_slide()

        # Reduce matrix and labels to just highlighted rows
        reduced_X = Matrix(
            [
                ["x_{21}", "x_{22}", "\\cdots", "x_{2n}"],
                ["x_{41}", "x_{42}", "\\cdots", "x_{4n}"],
            ]
        ).scale(SCALE)
        reduced_X.move_to(X.get_center())  # Move to the center of the highlighted rows
        reduced_y = Matrix(
            [
                ["y_2"],
                ["y_4"],
            ]
        ).scale(SCALE)
        reduced_y.move_to(
            y.get_center()
        )  # Move to the center of the highlighted entries

        # Transform the highlight boxes into the reduced matrices
        new_highlight_boxes = []
        for i, r in enumerate(highlight_rows):
            # Transform X row box
            new_box = SurroundingRectangle(
                reduced_X.get_rows()[i], color=PINK, buff=0.1
            )
            new_highlight_boxes.append(new_box)

            # Transform y entry box
            new_y_box = SurroundingRectangle(
                reduced_y.get_rows()[i], color=PINK, buff=0.1
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


class LeverageScores(MySlide): 
    def construct(self):
        # Leverage scores are a common method for selecting informative samples in active learning, but they can be sensitive to changes in the model's representation.
        title = Text(
            "Leverage Scores",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
            t2c={"Leverage Scores": YELLOW},  # pyright: ignore[reportArgumentType]
        ).to_edge(UP + LEFT)

        description = (
            Tex("Sensitivity of the coefficients to each data point", font_size=28)
            .next_to(title, DOWN)
            .to_edge(LEFT, buff=1)
        )

        # Define Leverage Scores
        definition = (
            MathTex(
                "\\ell_i (\\mathbf{x}_i) = \\mathbf{x}_i^T (X^T X)^{-1} \\mathbf{x}_i = \\|Q^T \\mathbf{e}_i\\|^2 \\quad \\text{(Classic Definition)}",
                font_size=28,
                color=YELLOW,
            )
            .next_to(description, DOWN)
            .to_edge(LEFT, buff=2)
        )

        alternate_description = (
            Tex(
                "How easily each data point can be reconstructed by other points",
                font_size=28,
            )
            .next_to(definition, DOWN, buff=1)
            .to_edge(LEFT, buff=1)
        )
        alternate_definition = (
            MathTex(
                "\\ell_i (\\mathbf{x}_i) = \\min_{\\mathbf{c}\\in \\mathbb{R}^n} \\|\\mathbf{c}\\|^2 \\text{ such that } X^T \\mathbf{c} = \\mathbf{x}_i \\quad \\text{(Optimization Definition)}",
                font_size=28,
                color=YELLOW,
            )
            .next_to(alternate_description, DOWN)
            .to_edge(LEFT, buff=2)
        )

        self.play(
            FadeIn(title),
            FadeIn(description),
            FadeIn(definition),
            FadeIn(alternate_description),
            FadeIn(alternate_definition),
        )
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
        scores = (
            Matrix(map(lambda x: np.round(x, 2), compute_leverage_scores(X)[:, None]))
            .scale(0.7)
            .next_to(arrow, RIGHT, buff=0.5)
        )
        self.play(FadeIn(matrix))
        self.next_slide()

        # Animate minimal norm reconstruction of the third point using the first two points
        reconstruction = (
            MathTex(
                "\\left(0\\right) \\begin{bmatrix} 1 \\\\ 0 \\end{bmatrix} + (1)\\begin{bmatrix} 0 \\\\ 1 \\end{bmatrix} + (0)\\begin{bmatrix} 1 \\\\ 10 \\end{bmatrix} = \\begin{bmatrix} 0 \\\\ 1 \\end{bmatrix}",
                font_size=24,
            )
            .next_to(scores, RIGHT, buff=1)
            .shift(UP * 0.5)
        )
        implies = (
            MathTex(
                "\\implies \\mathbf{c} = \\begin{bmatrix} 0 \\\\ 1 \\\\ 0 \\end{bmatrix} \\text{ and } \\|\\mathbf{c}\\|^2 = 1",
                font_size=24,
            )
            .next_to(reconstruction, DOWN)
            .align_to(reconstruction, LEFT)
        )
        better_reconstruction = (
            MathTex(
                "\\left(-0.1\\right) \\begin{bmatrix} 1 \\\\ 0 \\end{bmatrix} + (0)\\begin{bmatrix} 0 \\\\ 1 \\end{bmatrix} + (0.1)\\begin{bmatrix} 1 \\\\ 10 \\end{bmatrix} = \\begin{bmatrix} 0 \\\\ 1 \\end{bmatrix}",
                font_size=24,
            )
            .next_to(scores, RIGHT, buff=1)
            .shift(UP * 0.5)
        )
        better_implies = (
            MathTex(
                "\\implies \\mathbf{c} = \\begin{bmatrix} -0.1 \\\\ 0 \\\\ 0.1 \\end{bmatrix} \\text{ and } \\|\\mathbf{c}\\|^2 = 0.02",
                font_size=24,
            )
            .next_to(better_reconstruction, DOWN)
            .align_to(better_reconstruction, LEFT)
        )

        self.play(FadeIn(reconstruction), FadeIn(implies))
        self.next_slide()

        self.play(
            FadeTransform(reconstruction, better_reconstruction),
            FadeTransform(implies, better_implies),
        )
        self.next_slide()

        self.play(FadeIn(arrow), FadeIn(arrow_label), FadeIn(scores))
        self.next_slide()


class LeverageScoresOnMNIST(MySlide):
    def construct(self):
        title = Text(
            "Leverage Scores of MNIST",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
        ).to_edge(UP + LEFT)

        # Show leverage scores for MNIST data points, and how they change when we apply a non-linear transformation to the data (e.g. a neural network representation).
        k = 60
        rows = 3
        top_k_digit_indices = np.argsort(leverage_scores)[-k:]
        bottom_k_digit_indices = np.argsort(leverage_scores)[:k]

        top_digits = Group(
            *[
                Group(
                    ImageMobject(
                        (mnist_X[i].numpy().reshape(28, 28) * 255).astype(np.uint8)
                    ).scale(3.5),
                    Tex(f"{leverage_scores[i]:.4f}", font_size=18),
                ).arrange(DOWN, buff=0.1)
                for i in top_k_digit_indices[::-1]
            ]
        )
        top_digits.scale(0.7).arrange_in_grid(
            rows=rows, cols=k // rows, buff=0.1
        ).next_to(title, DOWN, buff=1).to_edge(LEFT, buff=1)
        top_label = (
            Tex("Top Leverage Scores", font_size=24)
            .next_to(top_digits, UP)
            .to_edge(LEFT, buff=1)
        )
        bottom_digits = Group(
            *[
                Group(
                    ImageMobject(
                        (mnist_X[i].numpy().reshape(28, 28) * 255).astype(np.uint8)
                    ).scale(3.5),
                    Tex(f"{leverage_scores[i]:.4f}", font_size=18),
                ).arrange(DOWN, buff=0.1)
                for i in bottom_k_digit_indices[::-1]
            ]
        )
        bottom_digits.scale(0.7).arrange_in_grid(
            rows=rows, cols=k // rows, buff=0.1
        ).next_to(top_digits, DOWN, buff=1).to_edge(LEFT, buff=1)
        bottom_label = (
            Tex("Bottom Leverage Scores", font_size=24)
            .next_to(bottom_digits, UP)
            .to_edge(LEFT, buff=1)
        )

        self.play(
            FadeIn(title),
            FadeIn(top_digits),
            FadeIn(top_label),
            FadeIn(bottom_digits),
            FadeIn(bottom_label),
            run_time=0.5,
        )
        self.next_slide()


class AlternativeRepresentations(MySlide):
    def construct(self):
        title = Text(
            "Alternative Representations",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
            t2c={"Representations": RED},  # pyright: ignore[reportArgumentType]
        ).to_edge(UP + LEFT)

        bullets = (
            BulletedList(
                "Linear Regression on MNIST: 85\\% test accuracy (not great, but better than random guessing)",
                "Neural Network on MNIST: 99\\% test accuracy",
                font_size=28,
            )
            .next_to(title, DOWN)
            .to_edge(LEFT, buff=1)
        )

        # Draw log-log plot of polynomial data, and how with a non-linear transformation, a linear model can fit it well.
        
        plt.figure(figsize=(4, 4))
        x_axis = np.linspace(1.1, 4, 100)
        y = x_axis**4 - x_axis**2 + np.random.normal(0, 0.05, size=x_axis.shape)
        plt.scatter(x_axis, y, color="blue", label="Data Points")
        m,b = np.polyfit(x_axis, y, 1)
        plt.plot(x_axis, m * x_axis + b, color="red", label="Linear Fit")
        plt.title("Polynomial Data")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.savefig("figures/polynomial_data.png")
        plt.close()

        plot = ImageMobject("figures/polynomial_data.png").to_edge(LEFT).scale(0.7).shift(DOWN)
        plot_arrow1 = MathTex("\\rightarrow").scale(1).next_to(plot, RIGHT, buff=0.25)

        plt.figure(figsize=(4, 4))
        ln_x_axis = np.log(x_axis)
        ln_y = np.log(y)
        m, b = np.polyfit(ln_x_axis, ln_y, 1)
        plt.scatter(ln_x_axis, ln_y, color="blue", label="Transformed Data Points")
        plt.plot(ln_x_axis, m * ln_x_axis + b, color="red", label="Linear Fit")
        plt.title("Log-Log Plot")
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.savefig("figures/log_log_plot.png")
        plt.close()

        plot2 = ImageMobject("figures/log_log_plot.png").next_to(plot, DOWN, buff=0.5).scale(0.7).next_to(plot_arrow1, RIGHT, buff=0.25)
        

        self.play(
            FadeIn(title),
            FadeIn(bullets),
            FadeIn(plot),
            FadeIn(plot_arrow1),
            FadeIn(plot2),
        )
        self.next_slide()


        # Draw neural network architecture
        digit = (
            ImageMobject((mnist_X[0].numpy().reshape(28, 28) * 255).astype(np.uint8))
            .scale(6)
            .next_to(plot2, RIGHT, buff=1)
        )
        digit_label = Tex("Input Image", font_size=18).next_to(digit, DOWN)
        arrow1 = MathTex("\\rightarrow").scale(1).next_to(digit, RIGHT)

        nn = NeuralNetwork(
            [
                FeedForwardLayer(num_nodes=14),
                FeedForwardLayer(num_nodes=13),
                FeedForwardLayer(num_nodes=12),
                FeedForwardLayer(num_nodes=11, rectangle_color="red"),
                FeedForwardLayer(num_nodes=10),
            ],
        ).next_to(arrow1, RIGHT, buff=0.25)
        nn.scale(0.5)

        nn_label = Tex("Neural Network", font_size=18).next_to(nn, DOWN)
        nn_function = MathTex(
            "f(\\mathbf{x}) = W\\underbrace{\\sigma(...)}_{\\text{Embedding} } +~b", font_size=18
        ).next_to(nn_label, DOWN)
        arrow2 = MathTex("\\rightarrow").scale(1).next_to(nn, RIGHT)


        state = torch.load(
            "networks/mnist-cnn-B1-84b328155fa96389e73cce5eed35404f17fa625a35db8b0beb7da142a279277f.pth",
            map_location=torch.device("cpu"),
        )
        net = MnistConvNet()
        net.load_state_dict(state)
        net.eval()
        prediction = net(mnist_X[0:1]).detach().numpy().round(4)

        output = (
            Matrix(prediction.reshape(-1, 1))
            .scale(0.3)
            .next_to(arrow2, RIGHT, buff=0.5)
        )
        output_label = Tex("Output", font_size=18).next_to(output, DOWN)
        ideal = (
            Matrix([[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]])
            .scale(0.3)
            .next_to(output, RIGHT, buff=0.5)
        )
        ideal_label = Tex("Ideal Output", font_size=18).next_to(ideal, DOWN)

        self.play(
            FadeIn(digit),
            FadeIn(digit_label),
            FadeIn(arrow1),
            FadeIn(nn),
            FadeIn(nn_label),
            FadeIn(nn_function),
            FadeIn(arrow2),
            FadeIn(output_label),
            FadeIn(output),
            FadeIn(ideal),
            FadeIn(ideal_label),
        )
        self.next_slide()


class LeverageScoresOfNNRepresentation(MySlide):
    def construct(self):
        title = Text(
            "Leverage Scores of Neural Network Representations",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
        ).to_edge(UP + LEFT)

        state = torch.load(
            "networks/mnist-cnn-B1-84b328155fa96389e73cce5eed35404f17fa625a35db8b0beb7da142a279277f.pth",
            map_location=torch.device("cpu"),
        )
        net = MnistConvNet()
        net.load_state_dict(state)
        net.eval()
        embedding = embed_dataset(mnist_X, net, torch.device("cpu"), basics_functions=200).numpy()
        leverage_scores_nn = compute_leverage_scores(embedding)

        # Show the top leverage digits of the neural network representation
        top_k = 60
        rows = 3
        top_k_digit_indices = np.argsort(leverage_scores_nn)[-top_k:]
        top_digits = Group(
            *[
                Group(
                    ImageMobject(
                        (mnist_X[i].numpy().reshape(28, 28) * 255).astype(np.uint8)
                    ).scale(3.5),
                    Tex(f"{leverage_scores_nn[i]:.4f}", font_size=18),
                ).arrange(DOWN, buff=0.1)
                for i in top_k_digit_indices[::-1]
            ]
        )
        top_digits.scale(0.7).arrange_in_grid(
            rows=rows, cols=top_k // rows, buff=0.1
        ).next_to(title, DOWN, buff=1).to_edge(LEFT, buff=1)
        top_label = (
            Tex("Top Leverage Scores of NN Representation", font_size=24)
            .next_to(top_digits, UP)
            .to_edge(LEFT, buff=1)
        )

        # Show the bottom leverage digits
        bottom_k_digit_indices = np.argsort(leverage_scores_nn)[:top_k]
        bottom_digits = Group(
            *[
                Group(
                    ImageMobject(
                        (mnist_X[i].numpy().reshape(28, 28) * 255).astype(np.uint8)
                    ).scale(3.5),
                    Tex(f"{leverage_scores_nn[i]:.4f}", font_size=18),
                ).arrange(DOWN, buff=0.1)
                for i in bottom_k_digit_indices[::-1]
            ]
        )
        bottom_digits.scale(0.7).arrange_in_grid(
            rows=rows, cols=top_k // rows, buff=0.1
        ).next_to(top_digits, DOWN, buff=1).to_edge(LEFT, buff=1)
        bottom_label = (
            Tex("Bottom Leverage Scores of NN Representation", font_size=24)
            .next_to(bottom_digits, UP)
            .to_edge(LEFT, buff=1)
        )


        self.play(FadeIn(title), FadeIn(top_label), FadeIn(top_digits), FadeIn(bottom_label), FadeIn(bottom_digits))
        self.next_slide()


class LeverageScoresForCoreSetSelection(MySlide):
    def construct(self):
        title = Text(
            "Leverage Scores for Core Set Selection",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
        ).to_edge(UP + LEFT)

        # Show how both the leverage scores of the RAW embedding and of the 99% accurate neural network embedding
        # don't actually select the most informative samples in comparison to uniformly selecting samples from the dataset,
        # and how this is because leverage scores are only looking at linear independence, which is not the same thing as importance for training a model.

        bullets = BulletedList(
            "Uniformly random selection beats leverage score selection",
            "Unique points are not necessarily useful points",
            font_size=24,
        ).next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=1)

        plot1 = ImageMobject("figures/neural_networks_with_leverage_scores.png").scale(0.80).to_edge(DOWN + LEFT, buff=0.5)
        plot2 = ImageMobject("figures/neural_networks_with_nn_leverage_scores.png").scale(0.80).next_to(plot1, RIGHT, buff=0.25)

        self.play(FadeIn(title), FadeIn(bullets), FadeIn(plot1), FadeIn(plot2))
        self.next_slide()


class KeyFindings(MySlide):
    def construct(self):
        title = Text(
            "Key Findings",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
        ).to_edge(UP + LEFT)

        bullets = BulletedList(
            'Leverage scores find unique points (however ``linearly unique point" $\\neq$ ``useful point")',
            "We need to use more techniques for selecting informative samples",
            font_size=28,
        ).to_edge(LEFT, buff=1)
        self.play(FadeIn(title), FadeIn(bullets))
        self.next_slide()


class ThankYouSlide(MySlide):
    def construct(self):
        title = Text(
            "Thank You!",
            font_size=HEADER_FONT_SIZE,
            weight=BOLD,
        )
        self.play(FadeIn(title), run_time=1)


if __name__ == "__main__":
    print(*__all__)
