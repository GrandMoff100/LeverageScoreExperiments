#!/bin/bash


. ./.venv/bin/activate

# Check if arguments are provided
if [ "$#" -eq 0 ]; then
    slides=$(python presentation.py)
else
    slides="$@"
fi

# Render the slides
manim-slides render -q h -v WARNING presentation.py $slides

# 1 minute - l
# 2 minutes - m
# 4 minutes - h
# 10 minutes - k

# Convert the slides to HTML
manim-slides \
    convert --to html --one-file --open \
    $slides index.html
