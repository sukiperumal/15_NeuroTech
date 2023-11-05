import nibabel as nib
from nilearn import datasets, input_data, plotting

# Load fMRI data
fmri_img = nib.load('rest_denoised_bandpassed_norm.nii')
anat_img = nib.load('sub-032301_ses-01_acq-mp2rage_brain.nii')  # Optional: Load anatomical data

# Define and load a brain atlas (e.g., Harvard-Oxford Atlas)
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

# Create a NiftiMasker to extract time series data from ROIs
masker = input_data.NiftiMasker(mask_img=atlas.maps, sessions=None, smoothing_fwhm=None, detrend=True, standardize=True)
fmri_data = masker.fit_transform(fmri_img)


# Segment fMRI data into ROIs

# Visualize the segmented ROIs
plotting.plot_roi(atlas.maps, title='Harvard-Oxford Atlas', display_mode='ortho')

# Perform statistical analysis, connectivity analysis, or further processing
