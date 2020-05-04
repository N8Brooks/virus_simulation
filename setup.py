#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 10:52:08 2020

@author: nathan
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("traveltools.pyx")
)