#!/bin/bash

version=$1

for hemi in "lh" "rh"; do	
	echo "running ${hemi}"
	meshlabserver -i /scr/ilz3/myelinconnect/groupavg/highres_${hemi}.ply -o /scr/ilz3/myelinconnect/groupavg/lowres_${hemi}_${version}.ply -s simple_${version}.mlx -d simple_${version}.filter -l simple_${version}.log	
done 
