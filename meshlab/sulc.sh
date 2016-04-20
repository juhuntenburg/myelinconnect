ConvertSurface -i_ply lowres_lh_d.ply -o_fs lowres_lh_d.asc
mris_convert lowres_lh_d.asc lowres_lh_d
mris_inflate -n 1 -sulc lowres_lh_d.sulc lh.lowres_lh_d lh.lowres_lh_d_inf
mris_convert -c lh.lowres_lh_d.sulc lh.lowres_lh_d lowres_lh_d.sulc.asc

# Sulcal information is the 5th column of lowres_lh_d.sulc.asc
