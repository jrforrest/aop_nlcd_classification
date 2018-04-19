# AOP ML Experimentation

Let's see what we can do with some data Kate's provided from NEON.

## Setup

I'm using Python3 here with frozen requirements.

`virtualenv -p python3 .env` will get things set up if a python `>= 3.6` is installed.

`. .env/bin/activiate` to get the proper binaries loaded up.

`pip install -r requirements.txt` to get deps.

For dev, probably run `ipython`, `%load_ext autoreload` and `%autoreload 2`
to get autoreloading of modules set up.

To see what the models do right away, just exec `./main.py`.

## Data

Index: The first column of each CSV appears to be just a line number, 0 indexed

GRID_CODE: Data was plucked from a grid with geohashed coords to provide a
           somewhat random sampling.  Grid code is just the grid it came from.
           Probably use this as a primary key.

NLCD: Classification codes from the NLCD classication database.  This identifies
      kind of cover a given grid contains.  I'm not sure if these are Murph's
      manually identified classifications or if these came from the NLCD DB.

B\*: Band data, will be our feature set for classification.
