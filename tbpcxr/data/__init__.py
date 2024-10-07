import sys

if sys.version_info < (3, 9):
    import importlib_resources
else:
    import importlib.resources as importlib_resources
import os

ref = importlib_resources.files(f"{__name__}")
# iterate over ref, and make a list of files with the pkl extension
model_list = []
for entry in ref.iterdir():
    name, ext = os.path.splitext(entry.name)
    if ext == ".pkl":
        model_list.append(name)

del os
del importlib_resources
