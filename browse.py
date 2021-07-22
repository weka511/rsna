
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure                 import Figure
from matplotlib.pyplot                 import close, imshow, show
from mri3d                             import Labelled_MRI_Dataset
from os.path                           import join
from pydicom                           import dcmread
from tkinter                           import BOTH, BOTTOM, END, Frame, LEFT, Listbox, Scrollbar, Tk, TOP

study      = None
series     = None
slice_path = None
canvas     = None

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
        for series in study.get_series():
            LB_series.insert(END,series.description)

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
        for slice in series.seqs:
            LB_slices.insert(END,slice)

def onselect_slice(evt):
    global slice_path, canvas
    widget     = evt.widget
    selection  = widget.curselection()
    if selection:
        try: # https://stackoverflow.com/questions/56154065/using-tkinter-how-to-clear-figurecanvastkagg-object-if-exists-or-similar
            canvas.get_tk_widget().pack_forget()
        except AttributeError:
            pass
        index        = int(selection[0])
        slice_number = widget.get(index)
        slice_path   = join(series.dirpath,f'Image-{slice_number}.dcm')
        print (f'Study: {study}, Series: {series}, Slice: {slice_number} -- {slice_path}')
        dcim = dcmread(slice_path)
        fig  = Figure(figsize=(10, 10), dpi=100)
        fig.add_subplot(111).imshow(dcim.pixel_array)

        canvas = FigureCanvasTkAgg(fig, master=bottomframe)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

if __name__=='__main__':
    root = Tk()
    root.title('Browse MIR images')
    root.geometry('1200x720+300+300')
    root.resizable(True, True)

 #  Build UI elements

    frame           = Frame(root,height=50)
    bottomframe     = Frame(root, height=100)
    scrollbar_left  = Scrollbar(frame)
    LB_studies      = Listbox(frame, bg='#90e4c1')
    LB_series       = Listbox(frame, bg='#90e4c1')
    LB_slices       = Listbox(frame, bg='#029386')
    scrollbar_right = Scrollbar(frame, bg='#029386')

#   pack...

    frame.pack()
    bottomframe.pack(side = BOTTOM )
    scrollbar_left.pack(side = LEFT, fill = BOTH)
    LB_studies.pack(side = LEFT)
    LB_series.pack(side = LEFT)
    LB_slices.pack(side = LEFT)
    scrollbar_right.pack(side = LEFT, fill = BOTH)

#   Wire up scrollbars

    LB_studies.config(yscrollcommand = scrollbar_left.set)
    scrollbar_left.config(command = LB_studies.yview)
    LB_slices.config(yscrollcommand = scrollbar_right.set)
    scrollbar_right.config(command = LB_slices.yview)

#   Bind event handlers

    LB_studies.bind('<<ListboxSelect>>', onselect_studies)
    LB_series.bind('<<ListboxSelect>>',  onselect_series)
    LB_slices.bind('<<ListboxSelect>>',  onselect_slice)

    training = Labelled_MRI_Dataset(r'D:\data\rsna','train')

    for study in training.get_studies():
        LB_studies.insert(END, study)

    root.mainloop()
