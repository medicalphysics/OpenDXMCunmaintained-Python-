Description of voxel model "GOLEM"

segm_golem .............. File with binary data of segmented voxel model
Organlist.doc ........... WINWORD document describing the relation between
                          organs and organ identification numbers


Structure of the file "segm_golem":

This file contains a header of length 4096 bytes.
Then follow binary data of the 3D matrix of organ identification numbers.
The dimensions of the matrix are: 256 x 256 x 220 (columns x rows x slices).
The data are listed slice by slice, within each slice row by row, 
within each row column by column.
That means, the column index (1-256) changes fastest, then the row index
(1-256), then the slice index (1-220).
The organ identification numbers range from 1 to 253, thus requiring
1 byte storage per voxel (unsigned character).

Slice numbers increase from the vertex of the body down to the toes.
Row numbers increase from front to back.
Column numbers increase from left to right side.

The original voxel dimensions are 2.08 x 2.08 x 8.0 mm**3 (column width x
row depth x slice height).


