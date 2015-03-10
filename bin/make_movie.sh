#!/bin/bash
# cf. http://pages.uoregon.edu/noeckel/MakeMovie.html

echo "Generating rod Gif"
convert -delay 20 -loop 0 results/tmp/rod_1_*.jpg results/rod_1.gif

echo "Animations generated"
