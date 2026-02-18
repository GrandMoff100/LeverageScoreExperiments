#!/bin/bash


. ./.venv/bin/activate

# Check if arguments are provided
if [ "$#" -eq 0 ]; then
    slides=$(python presentation.py)
else
    slides="$@"
fi

# Render the slides
manim-slides render -q m -v WARNING presentation.py $slides

# Convert the slides to HTML
manim-slides \
    convert --to html --one-file \
    $slides presentation.html
