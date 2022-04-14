from setuptools import setup, find_packages
from fltk import __version__
setup(
    name="fltk",
    author="Bart Cox",
    author_email="b.a.cox@tudelft.nl",
    maintainer="Bart Cox",
    maintainer_email="b.a.cox@tudelft.nl",
    description="Federated Learning Toolkit",
    packages=find_packages(),
    version=__version__,
    entry_points={
        "console_scripts": [
            "fltk = fltk.__main__:main",
        ]
    },
    include_package_data=True,
    data_files=[('share/tudelft/fltk/configs', ['configs/experiment.yaml'])],
    install_requires=
    [
        'tqdm==4.49.0',
        'scikit-learn==0.23.2',
        'pandas==1.1.2',
        'numpy>=1.20.0',
        'torch==1.7.1',
        'torchvision==0.8.2',
        'scipy==1.4.1',
        'h5py==2.10.0',
        'requests',
        'pyyaml',
        'torchsummary',
        'dataclass-csv',
        'tensorboard'
    ]
)
