The only purpose of using Cython is to improve the calculation efficiency.

1. Creat Cython file steps:
a. Create a file named '***.pyx'
b. Create a python Makefile(a setup file for the .pyx file, which contains the command to recomplile the .pyx file into C) named 'setup.py'
   The setup.py usually looks like below:
#**code begin**#
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("***.pyx")
)
#**code end**#
c. use the command below to build Cython file. The setup.py is the Makefile.
#**command begin**#
$ python setup.py build_ext --inplace
#**command end**#
d. if the recomplile is succusesful, two files will be generated--'***.c' and '***.cp39-win_amd64'. Otherwhise, check the command window for error. 
e. for more details, check this website below:
http://man.hubwiz.com/docset/Cython.docset/Contents/Resources/Documents/docs.cython.org/src/tutorial/cython_tutorial.html

2. Python/Cython material properties file speed tryout:
a. tried 14 ways of iterating through the material properties to find the most effective method. version 13 is the fastest way, which is very similar to version 
   9,10,14
b. 

3. NOTICE!!!
a. the function in cython file, which will be called by python file later on, should use 'def' instead of 'cdef'
c. iterating through the list somehow is faster than iterating through a numpy array. Some resource from stackoverflowed 
   might say the opposite. Do compare the different methods before finalizing the script. 
b. there is very little difference among the all the versions, especially 9,10,14 