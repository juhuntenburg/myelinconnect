import numpy as np
import nibabel as nib

surfmL = nib.freesurfer.read_geometry('S900.L.midthickness_MSMAll.32k_fs_LR.surf')
surfiL = nib.freesurfer.read_geometry('S900.L.very_inflated_MSMAll.32k_fs_LR.surf')
surffL = nib.freesurfer.read_geometry('S900.L.flat.32k_fs_LR.surf')
surfL = []
surfL.append(np.array(surfmL[0]*0.3 + surfiL[0]*0.7))
surfL.append(surfmL[1])

surfmR = nib.freesurfer.read_geometry('S900.R.midthickness_MSMAll.32k_fs_LR.surf')
surfiR = nib.freesurfer.read_geometry('S900.R.very_inflated_MSMAll.32k_fs_LR.surf')
surffR = nib.freesurfer.read_geometry('S900.R.flat.32k_fs_LR.surf')
surfR = []
surfR.append(np.array(surfmR[0]*0.3 + surfiR[0]*0.7))
surfR.append(surfmR[1])

# cortL and cortR are the indices.
res = nib.load('hcp.tmp.lh.dscalar.nii').get_data().squeeze()
cortL = np.squeeze(np.array(np.where(res != 0)[0], dtype=np.int32))
res = nib.load('hcp.tmp.rh.dscalar.nii').get_data().squeeze()
cortR = np.squeeze(np.array(np.where(res != 0)[0], dtype=np.int32))
cortLen = len(cortL) + len(cortR)

sulcL = np.zeros(len(surfL[0]))
sulcR = np.zeros(len(surfR[0]))
sulcL[cortL] = -1 * nib.load('S900.sulc_MSMAll.32k_fs_LR.dscalar.nii').get_data()[:len(cortL)].squeeze()
sulcR[cortR] = -1 * nib.load('S900.sulc_MSMAll.32k_fs_LR.dscalar.nii').get_data()[len(cortL)::].squeeze()

# Location of connectivity matrix (best to check with Marcel regarding specific
# questions, as he created this):
# /nobackup/kaiser2/dense_connectome/connectivity_matrices.hdf5

# To create connectivity matrix from scratch:
dcon = np.tanh(nib.load('/nobackup/kaiser2/dense_connectome/HCP_S900_820_rfMRI_MSMAll_groupPCA_d4500ROW_zcorr.dconn.nii ').data)

###########################################
# Code used previously to generate vectors:

# following file can be downloaded at:
# https://www.dropbox.com/s/qv4hzdcb6nait96/rsFC_eigenvectors.dscalar.nii?dl=0
emb = nib.load('rsFC_eigenvectors.dscalar.nii')
components = np.squeeze(emb.get_data()).T[0:cortLen,:]
np.save('gradients.npy', components)

myelin = nib.load('S900.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii').get_data().squeeze()
np.save('myelin.npy', myelin)

thickness = nib.load('HCP_S900_GroupAvg_v1/S900.thickness_MSMAll.32k_fs_LR.dscalar.nii').get_data().squeeze()
np.save('thickness.npy', thickness)
