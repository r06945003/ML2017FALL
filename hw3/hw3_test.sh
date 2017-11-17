#!/bin/bash
wget 'https://www.dropbox.com/s/c3vuarjcs6d3pxf/AUG_final_8.hdf5?dl=1' -O AUG_final_8.hdf5
wget 'https://www.dropbox.com/s/8nda4vymnkz891y/AUG_final_7.hdf5?dl=1' -O AUG_final_7.hdf5
wget 'https://www.dropbox.com/s/pek03x29r29573a/AUG_final_9.hdf5?dl=1' -O AUG_final_9.hdf5

python3 hw3_test.py $1 $2 