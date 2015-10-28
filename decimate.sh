#!/bin/bash

version="c"

for sub in "BP4T" "GAET" "HJJT" "KSMT" "KSYT" "OL1T" "PL6T" "SC1T" "WSFT"; do	
	echo "running ${sub}"
	meshlabserver -i /scr/ilz3/myelinconnect/struct/surf_rh/projected_surface/${sub}_rh_mid_def.ply -o /scr/ilz3/myelinconnect/final/decimate_surface/${sub}_rh_mid_def_simple_${version}.ply -s simple_${version}.mlx -d simple_${version}.filter -l simple_${version}.log	
done 


