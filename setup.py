try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': '2-D simulation of gas particles in a box',
    'author': 'John C. Lurie',
    'url': 'https://github.com/john-lurie/boxsim',
    'version': '0.1',
    'install_requires': ['numpy', 'matplotlib'],
    'packages': ['simulation'],
    'scripts': [''],
    'name': 'boxsim'
}

setup(**config)
