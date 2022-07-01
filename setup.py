import os
import codecs

from setuptools import setup

base_dir = os.path.abspath(os.path.dirname(__file__))
def read(fname):
    return codecs.open(os.path.join(base_dir, fname), encoding="utf-8").read()

setup(
    name='VarMINT',
    version='2019.1',
    packages=[''],
    url='https://github.com/david-kamensky/VarMINT',
    license='GNU LGPLv3',
    author='D. Kamensky',
    author_email='',
    description="Variational Multiscale formulation for incompressible fluids for FEniCS",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
)
