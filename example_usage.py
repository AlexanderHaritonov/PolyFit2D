import numpy as np
from skimage import measure

from src.fit_to_points_sequence import FitterToPointsSequence
from src.plotting import show_fitted_polygon

"""Create a small bitmap with a simple shape"""
bitmap = np.zeros((100, 100), dtype=np.uint8)
bitmap[20:80, 30:70] = 1
bitmap[30:50, 50:80] = 1

"""extract contour - a dense points sequence"""
contour = measure.find_contours(bitmap, level=0.5)[0]

""" fit a polygon """
segments = FitterToPointsSequence(contour, is_closed=True).fit()

""" plot """
show_fitted_polygon(bitmap, contour, segments, filename='fitted_segments.png')