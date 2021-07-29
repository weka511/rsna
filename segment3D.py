# MIT License

# Copyright (c) 2021 Simon Crase -- simon@greenweaves.nz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from argparse          import ArgumentParser
from matplotlib.pyplot import figure, savefig, show
from mri3d             import Labelled_MRI_Dataset, MRI_Geometry, Study
from numpy             import array, matmul

def get_3d(dcim):
    xs = []
    ys = []
    zs = []
    cs = []
    if dcim.pixel_array.sum()>0:
        Matrix = MRI_Geometry.create_matrix(dcim)
        for i in range(0,dcim.Rows,args.stride):
            for j in range(0,dcim.Columns,args.stride):
                if dcim.pixel_array[i,j]>0:
                    vector_ij = array([i,j,0,1])
                    vector_XYZ = matmul(Matrix,vector_ij)
                    xs.append(vector_XYZ[0])
                    ys.append(vector_XYZ[1])
                    zs.append(vector_XYZ[2])
                    cs.append(dcim.pixel_array[i,j])
    return xs,ys,zs,cs

if __name__=='__main__':
    parser = ArgumentParser('Visualize & segment in 3D')
    parser.add_argument('--path',     default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--figs',     default = './figs',                     help = 'Path to store plots')
    parser.add_argument('--show',     default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    parser.add_argument('--study',    default = '00098',                      help = 'Name of Studies to be processed' )
    parser.add_argument('--cmap',     default = 'gray',                       help = 'Colour map for displaying greyscale images')
    parser.add_argument('--stride',   default = 8)
    args = parser.parse_args()

    dataset = Labelled_MRI_Dataset(args.path,'train')
    study   = dataset[args.study]
    label   = dataset.get_label(args.study)

    fig = figure(figsize=(20,20))
    axs  = {series.description:fig.add_subplot(2,2,i+1,projection='3d') for i,series in enumerate(study.get_series())}

    for series in study.get_series():
        for dcim in series.dcmread():
            xs,ys,zs,cs = get_3d(dcim)
            if len(xs)>0:
                axs[series.description].scatter(xs,ys,zs,c=cs,s=1)
        axs[series.description].set_xlabel('X')
        axs[series.description].set_ylabel('Y')
        axs[series.description].set_zlabel('Z')
        axs[series.description].set_title(series.description)
    fig.suptitle(f'{study.name} {label}')
    if args.show:
        show()

