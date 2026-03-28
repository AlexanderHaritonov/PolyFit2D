import sys
import os
import matplotlib
if not os.environ.get('SHOW_PLOTS'):
    matplotlib.use('Agg')  # non-interactive backend: plt.show() is a no-op

sys.path.insert(0, os.path.dirname(__file__))
