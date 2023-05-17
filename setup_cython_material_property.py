# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:23:23 2022

@author: Geng Li
"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cython_material_property.pyx")
)

# To build Cython file, run the line below in cmd 
# python setup.py build_ext --inplace