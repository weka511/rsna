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
from matplotlib.pyplot import get_cmap, savefig, show, subplots, suptitle
from mri3d             import declutter, ImagePlane, Labelled_MRI_Dataset, Study
from numpy             import array, matmul, sign
from numpy.linalg      import inv, norm
from operator          import itemgetter
from os.path           import join

def format_group(plane,group):
    return f'{plane}: [{",".join(series for series in group)}]'

# get_end_distances
#
# Check coplanar groups to see whether observations are collinear

def get_end_distances(coplanar_groups,study):

    # get_distance
    #
    # Get distance between point p and the point that corresponds to p on the line[P0,P1],
    #
    # Parameters:
    #     p         The point
    #     index     Identifies the axis used to establish correspondence
    #     P0        One of the two endpoints used to determine the line
    #     P1        One of the two endpoints used to determine the line
    #     slope     +/1, depending in whether P1[index]?P0[index]

    def get_distance(p,index,P0,P1,slope):
        def q(i):
            alpha = (p[index]-P0[index])/(P1[index]-P0[index])
            return alpha*slope*(P1[i]-P0[i])+P0[i]
        return norm([p[i]-q(i) for i in range(len(P0))])

    Distances  = {}
    for name,group in coplanar_groups.items():
        Orbits    = {series_name:study[series_name].get_orbit() for series_name in group}
        first_key = list(Orbits.keys())[0]
        P0        = Orbits[first_key][0]
        P1        = Orbits[first_key][-1]
        Delta     = [P1[i]-P0[i] for i in range(len(P0))]
        index, _  = max(enumerate([abs(delta) for delta in Delta]), key=itemgetter(1))
        slope     = sign(Delta[index])

        for series_name,Orbit in Orbits.items():
            if series_name != first_key:
                Distances[f'{first_key}-{series_name}'] = (get_distance(Orbits[series_name][0],index,P0,P1,slope),
                                                           get_distance(Orbits[series_name][-1],index,P0,P1,slope))
        return Distances

# plot_orbit
#
# Show how the patient moves through the MRI instrument during one Study

def plot_orbit(study,
               path   = './',
               weight = 10):
    fig       = figure(figsize=(20,20))
    ax        = axes(projection='3d')

    for series in study.get_series():
        orbit, trivial, SeriesDescription, PatientPosition, ImageOrientationPatient,_ = series.get_orbit_for_plot()
        ax.scatter(*orbit,
                   label = f'{SeriesDescription}: {PatientPosition} {ImagePlane.get(ImageOrientationPatient)}',
                   s     = [1 if t else weight for t in trivial])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title(f'{study}')
    ax.legend()
    savefig(join(path,f'{study}'))
    return fig

# create_cells
#
# A generator for iterating through cells, left to right, then start next row

def create_cells(ncols,axs):
    i = 0
    j = 0
    while True:
        yield axs[i,j]
        j +=1
        if j==ncols:
            j = 0
            i+= 1

# plot_series
#
# Plot an entire series of images

def plot_series(series,
               path   = './',
               study  = '',
               ncols  = 6,
               cmap   = get_cmap('coolwarm'),
               width  = 20,
               height = 20):

    _, trivial, Description, _, ImageOrientation,brightest = series.get_orbit_for_plot()
    non_trivial_slices                                     = sum([0 if t else 1 for t in trivial])
    nrows                                                  = max(2,non_trivial_slices // ncols)
    while nrows*ncols < non_trivial_slices: nrows +=1

    fig,axs   = subplots(nrows = nrows, ncols = ncols, figsize = (width,height))
    cell      = create_cells(ncols,axs)

    for k,dcim in enumerate(series.dcmread()):
        if not trivial[k]:
            next(cell).imshow(dcim.pixel_array/brightest,cmap = cmap)

    for i in range(nrows):
        for j in range(ncols):
            declutter(axs[i][j])

    suptitle(f'{study} {Description}: {ImagePlane.get(ImageOrientation)}')
    savefig(join(path,f'{study}-{Description}-{ImagePlane.get(ImageOrientation)}'))
    return fig

if __name__=='__main__':
    parser = ArgumentParser('Determine trajectories for all studies')
    parser.add_argument('--path',     default = r'D:\data\rsna',              help = 'Path for data')
    parser.add_argument('--unique',   default = 'unique.csv',                 help = 'File name for list of studies whose planes are identical')
    parser.add_argument('--figs',     default = './figs',                     help = 'Path to store plots')
    parser.add_argument('--show',     default = False, action = 'store_true', help = 'Set if plots are to be displayed')
    parser.add_argument('--studies',  default = [],    nargs = '*',           help = 'Names of Studies to be processed (omit to process all)' )
    parser.add_argument('--cmap',     default = 'gray',                       help = 'Colour map for displaying greyscale images')
    parser.add_argument('--coplanar', default = 'coplanar.txt')
    args = parser.parse_args()

    training = Labelled_MRI_Dataset(args.path,'train')

    with open(args.unique,'w') as out,              \
         open(args.coplanar,'w') as coplanar:
        for study in training.get_studies(study_names = args.studies):
            image_planes = study.get_image_planes()
            groups       = {name:[] for name in ImagePlane.get_names()}
            for series_type, image_plane in zip(Study.Series.Types,image_planes):
                groups[image_plane].append(series_type)
            coplanar_groups = {name: groups[name] for name in ImagePlane.get_names() if len(groups[name])>1}

            output_line = ';'.join(format_group(plane,coplanar_groups[plane]) for plane in sorted(coplanar_groups.keys()))
            print(f'{study.name}  {output_line}' )
            coplanar.write(f'{study.name}  {output_line}\n' )
            for key,value in get_end_distances(coplanar_groups,study).items():
                print (key,value)
                coplanar.write(f'{key}  {value}\n' )
            if len(set(image_planes))==1:
                out.write(f'{study}, {image_planes[0]}\n')
                fig = plot_orbit(study, path = args.figs)
                if not args.show:
                    close(fig)
                for series in study.get_series():
                    fig = plot_series(series,
                                      study = study,
                                      path  = args.figs,
                                      cmap  = args.cmap)
                    if not args.show:
                        close(fig)

    if args.show:
        show()
