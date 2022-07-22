# setup.py
from setuptools import setup
from setuptools import find_packages

setup(
    name='smirl',
    version='0.1.0',
    packages=find_packages(),
    long_description=open('README.md').read(),
    # install_requires=open('requirements.txt').read(),
)
