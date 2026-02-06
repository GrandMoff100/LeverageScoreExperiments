from manim import *
from manim_slides import Slide  # type: ignore
from datetime import datetime


config.background_color = GREEN_A
Text.set_default(color=BLUE_A)
Tex.set_default(color=RED_A)
MathTex.set_default(color=RED_E)
BulletedList.set_default(color=YELLOW_A)


class MySlide(Slide):
    skip_reversing = True


class TitleSlide(MySlide):
    def construct(self):
        title = Text(
            "Assessing Representation Sensitivity\n"
            "of Leverage Scores in Active Learning",
            font_size=48,
            line_spacing=1,
            weight=BOLD,
        )

        confdate = datetime(2026, 2, 28).strftime("%B %-d, %Y")
        today = datetime.now().strftime("%B %-d, %Y")
        subtitle = Text(
            f"Nate Larsen | {today} | Student Research Conference",
            font_size=28,
            color=GRAY,
        )
        subtitle.next_to(title, DOWN, buff=0.9)
        subsubtitle = Text(
            "Advised by Dr. Kevin Miller",
            font_size=18,
            color=GRAY,
        ).next_to(subtitle, DOWN, buff=0.5)

        line = Line(LEFT * 4, RIGHT * 4).next_to(title, DOWN)

        self.play(Write(title), run_time=4)
        self.play(Create(line))
        self.play(FadeIn(subtitle, shift=UP))
        self.play(FadeIn(subsubtitle, shift=UP))
        self.next_slide()


class IntroSlide(MySlide):
    def construct(self):
        title = Text("Motivation", font_size=48, weight=BOLD).to_edge(UP)
        line = Line(LEFT * 4, RIGHT * 4).next_to(title, DOWN)

        body = BulletedList(
            "Active learning is a powerful technique for reducing labeling costs in machine learning.",
            "Leverage scores are a popular method for selecting informative samples in active learning.",
            "However, the sensitivity of leverage scores to representation changes is not well understood.",
            font_size=28,
        )

        body.next_to(title, DOWN, buff=0.5)
        self.play(Write(title))
        self.play(Create(line))
        self.next_slide()
        for item in body:
            self.play(Write(item))
            self.next_slide()


class Slide1(MySlide):
    def construct(self):
        # Draw some math
        title = Text("Problem Statement", font_size=48, weight=BOLD).to_edge(UP)
        line = Line(LEFT * 4, RIGHT * 4).next_to(title, DOWN)
        subtitle = Text(
            "Given a dataset and a model representation, how do leverage scores change when the representation is altered?",
            font_size=28,
            color=GRAY,
        ).next_to(line, DOWN, buff=0.5)

        # Draw a math formula
        formula = MathTex(
            r"\text{Leverage Score}(x) = \|P x\|^2", font_size=36
        ).next_to(subtitle, DOWN, buff=0.5)

        self.play(Write(title), run_time=4)
        self.play(Create(line))
        self.play(FadeIn(subtitle, shift=UP))
        self.play(Write(formula))
        self.next_slide()

if __name__ == "__main__":
    print(*(cls.__name__ for cls in MySlide.__subclasses__()))
