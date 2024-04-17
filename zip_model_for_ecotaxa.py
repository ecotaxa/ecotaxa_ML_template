#!/usr/bin/env python

from zipfile import ZipFile

import os
if os.path.exists('io.zip'):
   os.remove('io.zip')

import pathlib
fe1 = [f for f in pathlib.Path('io/feature_extractor/').iterdir() if f.is_file()]
fe2 = [f for f in pathlib.Path('io/feature_extractor/variables/').iterdir() if f.is_file()]
fe = fe1 + fe2

# Create a ZipFile Object
with ZipFile('io.zip', 'w') as zip_object:
   # Adding files that need to be zipped
   zip_object.write('io/crop.txt')
   zip_object.write('io/dim_reducer.pickle')
   for f in fe:
       zip_object.write(f)
