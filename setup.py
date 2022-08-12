from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

iou_module = Pybind11Extension(
    'numiou',
    [str(fname) for fname in Path('numiou/ops').glob('*.cc')],
    include_dirs=['numiou/ops'],
    extra_compile_args=['-O3']
)

setup(
    name='numiou',
    version=0.1,
    author='hova88',
    author_email='84240614@qq.com',
    description='fast iou for numpy',
    ext_modules=[iou_module],
    cmdclass={"build_ext": build_ext},
    install_requires=['numpy',
                      ],
    
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',     
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.4',
    ],
    
)
