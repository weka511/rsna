from argparse import ArgumentParser
from mri3d    import Study
from os.path  import join

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--study')
    parser.add_argument('--path',    default = r'D:\data\rsna',              help = 'Path for data')
    args=parser.parse_args()

    study = Study(args.study,join(args.path,'train',args.study))
