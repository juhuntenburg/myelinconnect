from nipype.pipeline.engine import Node, Workflow, MapNode
import nipype.interfaces.utility as util
import nipype.interfaces.fsl as fsl
import nipype.interfaces.io as nio


def median(in_file, out_file):
    print 'calculating median'
    import nibabel as nb
    import numpy as np
    img = nb.load(in_file)
    data = np.median(img.get_data(), axis=3)
    median_img = nb.Nifti1Image(data,img.get_affine(), img.get_header())
    median_img.to_filename(out_file)
    return out_file



with open('/scr/ilz3/7tresting/surfaces/rh.txt', 'r') as f:
    surfaces=[line.strip() for line in f]
    
merge= fsl.Merge(in_files=surfaces,dimension='t').run()

out_file='/scr/ilz3/7tresting/surfaces/surf_rh/median.nii.gz'

median(merge.outputs.merged_file, out_file)


