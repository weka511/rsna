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
from matplotlib.pyplot import savefig, show, subplots, suptitle
from mri3d             import MRI_Dataset
from os.path           import join

if __name__=='__main__':
    parser = ArgumentParser('Determine trajectories for all studies')
    parser.add_argument('--figs',     default = './figs',                             help = 'Path to store plots')
    parser.add_argument('--plotfile', default = './sizes',                            help = 'Path to store plots')
    parser.add_argument('--path',     default = r'D:\data\rsna',                      help = 'Path for data')
    parser.add_argument('--dataset',  default = 'train', choices = ['test', 'train'], help = 'Path for data')
    args = parser.parse_args()

    training = MRI_Dataset(args.path, args.dataset)
    accumulator  = []
    consistent   = []
    for study in training.get_studies():
        xx = set()
        for series in study.get_series():
            for dcim in series.dcmread(stop_before_pixels=True):
                accumulator.append([dcim.Rows, dcim.Columns, dcim.PixelSpacing[0], dcim.PixelSpacing[1]])
                xx.add(f'{dcim.Rows} {dcim.Columns} {dcim.PixelSpacing[0]} {dcim.PixelSpacing[1]}')
                break
            consistent.append(len(xx))

    collated = list(zip(*accumulator))
    fig,axs  = subplots(nrows = 1, ncols = 3, figsize = (21,7))
    axs[0].set_title('Rows and Columns')
    axs[0].hist(collated[0], bins=25, label='Rows',    alpha=0.5)
    axs[0].hist(collated[1], bins=25, label='Columns', alpha=0.5)
    axs[0].legend()
    axs[1].set_title('Pixels')
    axs[1].hist([collated[2], collated[3]], bins=25, label='Pixels Rows')
    axs[2].set_title('Number of studies with all 4 series matching')
    axs[2].hist(consistent)
    axs[2].set_xticks(range(5))
    fig.suptitle(f'Dataset: {args.dataset}')
    savefig(join(args.figs,f'{args.plotfile}-{args.dataset}'))
    show()
