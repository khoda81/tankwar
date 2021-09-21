import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'tankwar'
AUTHOR = 'Mahdi Khodabandeh'
AUTHOR_EMAIL = '20.mahdikh.0@email.com'

LICENSE = 'MIT'
DESCRIPTION = 'MultiAgent gym environment for reinforcement learning'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = ['gym', 'pygame', 'pymunk', 'pyglet', 'numpy', 'pillow', 'tankwar']

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
