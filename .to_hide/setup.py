from os import path
from setuptools import setup, find_packages
# To use consisten encodings
from codecs import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

requirements = [
    'torch >= 1.8',
    'pyyaml',
    'easydict'
]

exec(open('bnn/version.py').read())
setup(
    name='bnn',
    version=__version__,

    description="Binarize deep convolutional neural networks using python and pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Author details
    author="Adrian Bulat",
    author_email="adrian@adrianbulat.com",
    url="https://github.com/1adrianb/binary-networks-pytorch",
    keywords=['artificial intelligence', 'convolutional neural network', 'binary networks', 'quantization', 'pytorch'],

    # Package info
    packages=find_packages(exclude=('test',)),

    install_requires=requirements,
    license='BSD',
    zip_safe=True,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',

        # Supported python versions
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
