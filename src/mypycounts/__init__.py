# read version from installed package
from importlib.metadata import version
__version__ = version("mypycounts")

# populate package namespace
from mypycounts.mypycounts import count_words
from mypycounts.plotting import plot_words