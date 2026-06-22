#!/bin/bash

### PARAMS for visualizing robots from the thesis "final_200" experiment ###

out_path="tmp_out"
study_name="thesis"
experiments="final_200"

# which run(s) to pull robots from
runs="1,2,3,4,5"

voxel_types="withbone"
env_conditions="none"
plastic=0

# must match the GRN params used for the final_200 runs
max_voxels=27
cube_face_size=3

evogym_steps=500
evogym_init_x=3
evogym_init_y=1
evogym_action_bias=1.0
evogym_action_amplitude=0.4
evogym_period_steps=20
evogym_render_mode="screen"
