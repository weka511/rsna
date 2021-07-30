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
from matplotlib.cm     import viridis
from matplotlib.pyplot import figure, savefig, show
from mri3d             import Labelled_MRI_Dataset, MRI_Geometry, Study
from numpy             import array, matmul, mean, std
from os.path           import join

def get_3d(series):
    xs = []
    ys = []
    zs = []
    cs = []
    for dcim in series.dcmread():
        Matrix = MRI_Geometry.create_matrix(dcim)
        for i in range(0,dcim.Rows,args.stride):
            for j in range(0,dcim.Columns,args.stride):
                if dcim.pixel_array[i,j]>0:
                    vector_ij  = array([i,j,0,1])
                    vector_XYZ = matmul(Matrix,vector_ij)
                    xs.append(vector_XYZ[0])
                    ys.append(vector_XYZ[1])
                    zs.append(vector_XYZ[2])
                    cs.append(dcim.pixel_array[i,j])
    return xs,ys,zs,cs

def  extract(xs0,ys0,zs0,cs0,n=0.7):
    mu    = mean(cs0)
    sigma = std(cs0)
    xs    = []
    ys    = []
    zs    = []
    cs    = []
    for i in range(len(xs0)):
        if cs0[i]>mu + n*sigma:
            xs.append(xs0[i])
            ys.append(ys0[i])
            zs.append(zs0[i])
            cs.append(cs0[i])
    return xs,ys,zs,cs

if __name__=='__main__':
    parser = ArgumentParser('Visualize & segment in 3D')
    parser.add_argument('--path',       default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--figs',       default = './figs',                     help = 'Path to store plots')
    parser.add_argument('--show',       default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    parser.add_argument('--study',      default = '00098',                      help = 'Name of Studies to be processed' )
    parser.add_argument('--cmap',       default = 'gray',                       help = 'Colour map for displaying greyscale images')
    parser.add_argument('--stride',     default = 1, type = int,                help = 'Stride for 2D image plane')
    parser.add_argument('--thresholds', default = ['FLAIR', 0.7, 'T1wCE', 0.9, 'T2w', 0.7], nargs='*', help = 'Cutoff values for plot')
    parser.add_argument('--hist',       default = False, action = 'store_true', help = 'Set if histograms are to be displayed')
    args       = parser.parse_args()
    dataset    = Labelled_MRI_Dataset(args.path,'train')
    study      = dataset[args.study]
    label      = dataset.get_label(args.study)
    Threshold  = {args.thresholds[i]:float(args.thresholds[i+1]) for i in range(0,len(args.thresholds),2)}
    for series in study.get_series():
        fig = figure(figsize=(20,20))
        ax1 = fig.add_subplot(2,2,1,projection='3d')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(series.description)
        xs,ys,zs,cs = get_3d(series)
        vmin        = min(cs)
        vmax        = max(cs)
        mu          = mean(cs)
        sigma       = std(cs)
        im1         = ax1.scatter(xs,ys,zs,c=cs,s=1,cmap=viridis,vmin=vmin,vmax=vmax)
        fig.colorbar(im1, ax=ax1)
        if args.hist:
            ax2 = fig.add_subplot(2,2,2)
            ax2.hist(cs)
            ax2.set_title(f'Mean = {mu:.0f}, std={sigma:.0f}')
        if series.description in Threshold:
            xs,ys,zs,cs = extract(xs,ys,zs,cs,n=Threshold[series.description])
            ax3 = fig.add_subplot(2,2,3,projection='3d')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            ax3.set_zlabel('Z')
            im3 = ax3.scatter(xs,ys,zs,c=cs,s=1,cmap=viridis,vmin=vmin,vmax=vmax)
            fig.colorbar(im3, ax=ax3)
            ax3.set_title(f'Values over {mu:.0f} + {Threshold[series.description]} * {sigma:.0f} = {mu+Threshold[series.description]*sigma:.0f}')
        fig.suptitle(f'{study.name} {label} {series.description}')
        savefig(join(args.figs,f'{args.study}-{series.description}-3D'))
    if args.show:
        show()

