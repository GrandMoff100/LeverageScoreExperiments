#!/bin/bash


. ./.venv/bin/activate

# Check if arguments are provided
if [ "$#" -eq 0 ]; then
    slides=$(python presentation.py)
else
    slides="$@"
fi

# Render the slides
manim-slides render -q k -v WARNING presentation.py $slides

# 1 minute - l
# 2 minutes - m
# 4 minutes - h
# 10 minutes - k

# Convert the slides to HTML
manim-slides \
    convert --to html --one-file --open \
    $slides index.html

sed -i 's/<title>Manim Slides<\/title>/<title>Assessing Representation Sensitivity of Leverage Scores<\/title>/g' index.html