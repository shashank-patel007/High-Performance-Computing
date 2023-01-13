from gc import callbacks
import numpy as np
from tkinter import *
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import pandas as pd
from scipy.interpolate import make_interp_spline

df = pd.read_csv("exp8_output.csv")
x = np.arange(7,12)

df.plot(subplots=True)

plt.tight_layout()
plt.show()