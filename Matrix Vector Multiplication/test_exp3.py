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

df = pd.read_csv("exp3_output.csv")
x = np.arange(10,1100,10) # floyd

def plot():
    global canvas
    global plot1
    var = IntVar()
    scale = Scale(window,variable=var,from_=2,to=5,command=updateScale,orient=HORIZONTAL)
    scale.pack(anchor=CENTER)
    # the figure that will contain the plot
    fig = Figure(figsize = (5, 5),
                 dpi = 100)
  
    
    # adding the subplot
    plot1 = fig.add_subplot(111,xlabel='Size of matrix',ylabel='Time (seconds)')
  
    # plotting the graph
    spline = make_interp_spline(x, df[str(var.get())])
    xsp = np.linspace(x.min(),x.max(),500)
    ysp = spline(xsp)
    plot1.plot(xsp,ysp)
    

    canvas = FigureCanvasTkAgg(fig,
                               master = window)  
    canvas.draw()


    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()
  
    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   window)
    toolbar.update()
  
    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()

def updateScale(var):
    plot1.clear()
    spline = make_interp_spline(x, df[var])
    xsp = np.linspace(x.min(),x.max(),500)
    ysp = spline(xsp)
    plot1.plot(xsp,ysp)
    canvas.draw()

window = Tk()       
window.title('Plot')
plot()
window.mainloop()