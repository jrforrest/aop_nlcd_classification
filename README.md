# AOP ML Experimentation

Let's see what we can do with some data Kate's provided from NEON.

## Data

Index: The first column of each CSV appears to be just a line number, 0 indexed
GRID_CODE: Data was plucked from a grid with geohashed coords to provide a
           somewhat random sampling.  Grid code is just the grid it came from.
           Probably use this as a primary key.
NLCD: Classification codes from the NLCD classication database.  This identifies
      kind of cover a given grid contains.  I'm not sure if these are Murph's
      manually identified classifications or if these came from the NLCD DB.

## Contact

Kate Murphy provided the data.
