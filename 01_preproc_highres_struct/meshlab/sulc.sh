surf=lh_lowres_new_taubin_150
hemi=lh

ConvertSurface -i_ply ${surf}.ply -o_fs ${surf}.asc
mris_convert ${surf}.asc ${surf}
mris_inflate -n 1 -sulc ${surf}.sulc ${hemi}.${surf} ${hemi}.${surf}_inf
mris_convert -c ${hemi}.${surf}.sulc ${hemi}.${surf} ${hemi}.${surf}.sulc.asc

rm ${hemi}.${surf}
rm ${hemi}.${surf}_inf
rm ${hemi}.${surf}.sulc

# Sulcal information is the 5th column of lowres_lh_d.sulc.asc

#import numpy as np
#sulc_all=np.loadtxt('lh.lh_lowres_new_taubin_150.sulc.asc')
#sulc=sulc_all[:,4]
#np.save('lh_lowres_new_taubin_150_sulc.npy', sulc)

#rm ${hemi}.${surf}.sulc.asc
