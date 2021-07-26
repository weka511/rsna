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

from numpy             import array, matmul, sign
from numpy.linalg      import inv, norm
from operator          import itemgetter
from pydicom           import dcmread
from os                import sep, listdir, walk
from os.path           import join, normpath
from pandas            import read_csv
from re                import match

# http://www.aboutcancer.com/mri_gbm.htm

# ImagePlane

class ImagePlane:
    @staticmethod
    def get_names():
        return ImagePlane.planes.keys()

    planes = {'Sagittal' : ([0,1,0],[0,0,1]),
              'Axial'    : ([1,0,0],[0,1,0]),
              'Coronal'  : ([1,0,0],[0,0,1])}
    # get_image_plane
    #
    # http://dicomiseasy.blogspot.com/2013/06/getting-oriented-using-image-plane.html

    @staticmethod
    def get(loc):
        X = [abs(round(a)) for a in loc[:3]]
        Y = [abs(round(a)) for a in loc[3:]]
        for image_plane_name,Axes in ImagePlane.planes.items():
            if X==Axes[0] and Y==Axes[1]:
                return image_plane_name


# Study
#
# This class represents a study for one patient
#
class Study:

    # Series
    #
    # Each Study comprises several (in practic 4) Series
    class Series:

        Types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
        def __init__(self,name):
            self.name             = name
            self.missing_images   = set()
            self.dirpath          = None
            self.image_plane      = None
            self.description      = None
            self.patient_position = None
            self.slices           = []

        def __str__(self):
            return self.description

        # add_images
        #
        # Add a collection of images to Series

        def add_images(self,dirpath,filenames):
            def extract_digits(s):
                m = match(r'\D*(\d+)\D+',s)
                if m:
                    return int(m.group(1))

            self.dirpath          = dirpath
            self.seqs             = sorted([extract_digits(name) for name in filenames])
            self.N                = self.seqs[-1]
            self.missing_images   = set([i for i in range(1,self.N) if i not in self.seqs])
            dcim                  = dcmread(join(dirpath,filenames[0]))
            self.image_plane      = ImagePlane.get(dcim.ImageOrientationPatient)
            self.description      = dcim.SeriesDescription
            self.patient_position = dcim.PatientPosition

        # dcmread
        #
        # A generator to iterate through all the images.
        #
        # parameters:
        #     stop_before_pixels    Used if we just want to analyze metadata

        def dcmread(self, stop_before_pixels = False):
            for i in range(1,len(self)+1):
                if i not in self.missing_images:
                    yield  dcmread(join(self.dirpath,f'Image-{i}.dcm'), stop_before_pixels = stop_before_pixels)

        def __getitem__(self,i):
            return  dcmread(join(self.dirpath,f'Image-{i}.dcm'))


        def __len__(self):
            return self.N

        # slice_files
        #
        # Generator for iterating through image files

        def slice_files(self):
            for i in range(1,self.N+1):
                if i not in self.missing_images:
                    yield join(self.dirpath,f'Image-{i}.dcm')

        # get_values_from_meta
        #
        # Convert meta data (list of tuples of ASCII data) to a list of data
        # sequences, each being the time series for one datum
        @staticmethod
        def get_values_from_meta(orbit):
            return [[float(a) for a in p] for p in list(zip(*orbit))]


        def get_orbit(self):
            return [[float(a) for a in dcim.ImagePositionPatient] for dcim in self.dcmread(stop_before_pixels=True)]

        # get_orbit_for_plot
        #
        # Get trajectory of patient

        def get_orbit_for_plot(self):
            orbit     = []
            trivial   = []
            brightest = 1
            for dcim in self.dcmread():
                orbit.append(dcim.ImagePositionPatient)
                trivial.append(dcim.pixel_array.sum()==0)
                brightest = max(brightest,dcim.pixel_array.max())
            return self.get_values_from_meta(orbit),trivial,dcim.SeriesDescription, dcim.PatientPosition, dcim.ImageOrientationPatient,brightest

    def __init__(self,name,path):
        self.series        = None
        self.name          = name
        for dirpath, dirnames, filenames in walk(path):
            if self.series == None:
                self.series = {series_name: Study.Series(series_name) for series_name in dirnames}
            else:
                path_components = normpath(dirpath).split(sep)
                series = self.series[path_components[-1]]
                series.add_images(dirpath,filenames)

    def get_series(self):
        for name in Study.Series.Types:
            yield self.series[name]

    @staticmethod
    def get_image_plane(series):
        path_name   = next(series.slice_files())
        dcim        = dcmread(path_name,stop_before_pixels=True)
        return  ImagePlane.get(dcim.ImageOrientationPatient)

    def get_image_planes(self):
        return [self.get_image_plane(series)  for series in self.series.values()]

    def __str__(self):
        return self.name

    def __getitem__(self, key):
        return self.series[key]

# MRI_Dataset
#
# An MRI Dataset comprises sevral stidies, either test or training

class MRI_Dataset:
    def __init__(self,path,folder):
        self.studies = {name:Study(name,join(path,folder,name)) for name in listdir(join(path,folder))}

    # __getitem__
    #
    # Get value of label

    def __getitem__(self, key):
        return self.studies[key]

    # get_studies
    #
    # A generator to iterate through studies
    #
    # Parameters:
    #     study_names     Specified list of studies (or empty for all studies in dataset)

    def get_studies(self, study_names = []):
        if len(study_names)==0:
            for study in self.studies.values():
                yield study
        else:
            for study_name in study_names:
                yield self.studies[study_name]

# Labelled_MRI_Dataset
#
# A Labelled_MRI_Dataset is a MRI_Dataset accompanied by labels for training

class Labelled_MRI_Dataset(MRI_Dataset):

    def __init__(self,path,folder,labels='train_labels.csv'):
        super().__init__(path,folder)
        self.labels = read_csv(join(path,labels),dtype={'BraTS21ID':str})

    # get_counts
    #
    # Get number of negative and positive labels

    def get_counts(self):
        ones = sum(max(0,label) for label in self.labels['MGMT_value'])
        return len(self.labels) - ones, ones

    # __getitem__
    #
    # Get value of label

    def get_label(self, key):
        return int(self.labels[self.labels.BraTS21ID==key].MGMT_value)



# hide_decorations
#
# Remove boxes and other decorations from plot

def declutter(ax,spines=['top','right','bottom', 'left']):
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    for spine in spines:
        ax.spines[spine].set_visible(False)




# MRI_Geometry
#
# Utilites for mapping coordinates
#
# https://nipy.org/nibabel/dicom/dicom_orientation.html
class MRI_Geometry:

    @staticmethod
    def create_matrix(dcim):
        delta_i, delta_j       = dcim.PixelSpacing
        Sx, Sy, Sz             = dcim.ImagePositionPatient
        Xx, Xy, Xz, Yx, Yy, Yz = dcim.ImageOrientationPatient
        return array([
            [Xx*delta_i, Yx*delta_j, 0, Sx],
            [Xy*delta_i, Yy*delta_j, 0, Sy],
            [Xz*delta_i, Yz*delta_j, 0, Sz],
            [0,          0,          0, 1]
        ])

    @staticmethod
    def create_vector(i,j):
        return array([i,j,0,1]).transpose()

    @staticmethod
    def create_midpoint(dcim):
        return MRI_Geometry.create_vector(dcim.Rows/2, dcim.Columns/2)

    @staticmethod
    def create_mapping(dcim):
        def row(i):
            return [array([i,j]) for j in range(dcim.Columns)]
        return [row(i) for i in range(dcim.Rows)]

    @staticmethod
    def get_closest(series,centre_pixel,centre_pos):
        min_distance = float('inf')
        closest_dcim = None
        for dcim in series.dcmread():
            M    = MRI_Geometry.create_matrix(dcim)
            c    = matmul(M,centre_pixel)
            dist = norm(centre_pos - c)
            if dist<min_distance:
                min_distance = dist
                closest_dcim = dcim
            elif dist>min_distance:
                return closest_dcim,min_distance,norm(centre_pos[0:2] - c[0:2]),M
        return closest_dcim,min_distance,norm(centre_pos[0:2] - c[0:2]),M



if __name__=='__main__':

    def get_image_plane(loc): # https://www.kaggle.com/davidbroberts/determining-mr-image-planes

        row_x = round(loc[0])
        row_y = round(loc[1])
        row_z = round(loc[2])
        col_x = round(loc[3])
        col_y = round(loc[4])
        col_z = round(loc[5])

        if row_x == 1 and row_y == 0 and col_x == 0 and col_y == 0:
            return "Coronal"

        if row_x == 0 and row_y == 1 and col_x == 0 and col_y == 0:
            return "Sagittal"

        if row_x == 1 and row_y == 0 and col_x == 0 and col_y == 1:
            return "Axial"

        return "Unknown"

    training = MRI_Dataset(r'D:\data\rsna','train')
    for study in training.get_studies():
        for series in study.get_series():
            for dcim in series.dcmread(stop_before_pixels=True):
                new = ImagePlane.get(dcim.ImageOrientationPatient)
                old = get_image_plane(dcim.ImageOrientationPatient)
                if new!=old:
                    print (dcim.ImageOrientationPatient,old,new)
                break  # Assume all images in series have same orientation
