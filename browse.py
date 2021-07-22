from tkinter                           import Frame, Tk,Listbox, TOP, BOTH, BOTTOM, LEFT
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from mri3d                             import Labelled_MRI_Dataset
from os.path                           import join
from matplotlib.figure import Figure
import numpy as np
from pydicom           import dcmread
from matplotlib.pyplot import close, imshow, show
study      = None
series     = None
slice_path = None
canvas = None
def onselect_studies(evt):
    global study
    widget = evt.widget
    selection = widget.curselection()
    if selection:
        index  = int(selection[0])
        key    = widget.get(index)
        study  = training[key]
        print (f'Study: {study.name}')
        LB_series.delete(0,'end')
        LB_slices.delete(0,'end')
        for i,series in enumerate(study.get_series()):
            LB_series.insert(i,series.description)

def onselect_series(evt):
    global series
    widget    = evt.widget
    selection = widget.curselection()
    if selection:
        index  = int(selection[0])
        key    = widget.get(index)
        series = study[key]
        print (f'Study: {study.name}, Series: {series.description}')
        LB_slices.delete(0,'end')
        for i,slice in enumerate(series.seqs):
            LB_slices.insert(i,slice)
fig = Figure(figsize=(5, 4), dpi=100)
ax=fig.add_subplot(111)
def onselect_slice(evt):
    global slice_path, fig
    widget     = evt.widget
    selection  = widget.curselection()
    if selection:
        if fig!=None:
            close(fig)
        index        = int(selection[0])
        slice_number = widget.get(index)
        slice_path   = join(series.dirpath,f'Image-{slice_number}.dcm')
        print (f'Study: {study}, Series: {series}, Slice: {slice_number} -- {slice_path}')
        dcim = dcmread(slice_path)
        fig = Figure(figsize=(5, 4), dpi=100)
        fig.add_subplot(111).imshow(dcim.pixel_array)#plot(t, 2 * np.sin(2 * np.pi * t))

        canvas = FigureCanvasTkAgg(fig, master=bottomframe)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        # imshow(dcim.pixel_array)

if __name__=='__main__':
    training = Labelled_MRI_Dataset(r'D:\data\rsna','train')

    top = Tk()
    top.title('Layout Test')
    top.geometry('1200x720+300+300')
    top.resizable(True, True)
    frame = Frame(top)
    frame.pack()
    bottomframe = Frame(top)
    bottomframe.pack( side = BOTTOM )
    LB_studies = Listbox(frame)
    LB_series  = Listbox(frame)
    LB_slices  = Listbox(frame)
    LB_studies.pack( side = LEFT)
    LB_series.pack( side = LEFT)
    LB_slices.pack( side = LEFT)

    LB_studies.bind('<<ListboxSelect>>', onselect_studies)
    LB_series.bind('<<ListboxSelect>>',  onselect_series)
    LB_slices.bind('<<ListboxSelect>>',  onselect_slice)

    # t = np.arange(0, 3, .01)
    # fig = Figure(figsize=(5, 4), dpi=100)
    # fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

    # canvas = FigureCanvasTkAgg(fig, master=bottomframe)  # A tk.DrawingArea.
    # canvas.draw()
    # canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    for i,study in enumerate(training.get_studies()):
        LB_studies.insert(i, study)

    top.mainloop()
