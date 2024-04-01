#!/bin/bash

source /opt/etc/bashrc

conda activate /home/pedro.silva/.conda/envs/myenvgpu

cd /home/pedro.silva/tese_ultrassom/sketch_guided_ultrasound_covid/

# Run Python parallel program
python cannyedges_sketchguided_covid.py $1 $2

exit