---
layout: post
title: "A tutorial for visualizing climate data in 3D"
author: "Dani Lafarga"
categories: journal
tags: [documentation,sample]
image: Jan_EOF_mode1.png
---

# Motivation
My previous 3D visualizations used Plotly, which was convenient and interactive, but it was limited in the amount of data it could visualize. For [my previous project](https://dlafarga.github.io/journal/CodeToplotfigures.html), this was not an issue as the data was small.

Once I started working with larger data, I found I was only able to plot one altitude layer of high-resolution data before crashing. This is when I turned to Python's Matplotlib library and its 3D projection. It took a lot less memory to create a figure, and it generated figures faster. Though I lost out on the interactive aspect (rotating, zooming in and out, etc.) I was still able to perform these tasks manually. 

The following code is a tutorial for how to is used to visualize 3D climate data using Matplotlib 3D projections. As an example, I use the first empirical orthogonal function (EOF) computed from the high-resolution Global Ocean Physics Reanalysis (GLORYS) output for all latitude-longitude-depth dimensions (3D) at once. The complete calculation for the GLORYS 3D EOFs can be found at [this repository](https://github.com/dlafarga/PIM-for-Computing-3D-EOFs). The repository for this full code is found at [this Github](https://github.com/dlafarga/3D-Visualization-Tutorial-using-GLORYS).


