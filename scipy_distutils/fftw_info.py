import os

def get_fftw_info():
    # FFTW (requires FFTW libraries to be previously installed)
    double_libraries = ['fftw_threads','rfftw_threads','fftw','rfftw']
    float_libraries = map(lambda x: 's'+x,double_libraries)

    if os.name == 'nt':
        fftw_dirs = ['c:\\fftw']
    else:
        base_dir = os.environ.get('FFTW')
        if base_dir is None:
            base_dir = os.environ['HOME']
        fftw_dirs = [os.path.join(base_dir,'lib')]
        double_libraries += ['pthread']
        float_libraries += ['pthread']

    return float_libraries, double_libraries, fftw_dirs
