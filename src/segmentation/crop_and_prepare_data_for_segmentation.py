import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
torch.cuda.empty_cache()
#from sympy import false
from unet_3d import Unet
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
from utils import apply_bias_correction


# --------------------------- CONFIG PATHS ---------------------------
input_root = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx//nifti_output/'
output_root = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx//segmentations/'
output_pelvis_path = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx//nifti_pelvis/'
output_lumbar_path = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx//nifti_lumbar/'
os.makedirs(output_root, exist_ok=True)
os.makedirs(output_pelvis_path, exist_ok=True)
os.makedirs(output_lumbar_path, exist_ok=True)
coord_csv = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx/slices_cortes_anatomicos.csv'
reference_path_gluteus = '/home/martin/data_imaging/Muscle/GlutealSegmentations/PelvisFOV/ManualSegmentations/MhdRegisteredDownsampled/ID00002.mhd'
reference_path_lumbar = '../../data/reference_images/lumbar_spine_reference.nii.gz'
gluteus_model_path = '/home/martin/data/UNSAM/Muscle/repoMuscleSegmentation/Data/GlutesPelvis3D/model/unet_20250807_110716_123_best_fit.pt'
lumbar_model_path = '../../models/lumbar_model.pt'

dixon_types = ['in', 'opp', 'f', 'w']
dixon_output_tag = ['I', 'O', 'F', 'W']
# --------------------------- LOAD COORDINATES AND IMAGES---------------------------
coords_df = pd.read_csv(coord_csv)
reference_image_gluteus = sitk.ReadImage(reference_path_gluteus)
reference_image_lumbar = sitk.ReadImage(reference_path_lumbar)



# --------------------------- LOAD MODELS an CREATE FILES---------------------------
multilabelNum = 8


print("Models loaded.\n")

# --------------------------- PROCESS EACH VOLUNTEER ---------------------------
#for idx, row in coords_df.iterrows():
for idx, row in coords_df.iloc[7:8].iterrows():
    volunteer_id = row['ID']
    outputPathThisSubject = os.path.join(output_root, volunteer_id )
    os.makedirs(outputPathThisSubject, exist_ok=True)
    trochanter = int(row['Trocánter menor'])
    iliac_crest = int(row['Cresta iliaca'])
    vertebra_L1 = int(row['Vértebra L1'])
    print(f"\n=== Processing volunteer: {volunteer_id} ===\nTLesser trochanter: {trochanter}\nTop Iliac Crest: {iliac_crest}\nL1: {vertebra_L1}")

    # --------------------------- LOAD IMAGE ---------------------------
    input_folder = os.path.join(input_root, volunteer_id)
    # Output paths for this volunteer
    output_pelvis_this_volunteer_path = os.path.join(output_pelvis_path, volunteer_id)
    output_lumbar_this_volunteer_path = os.path.join(output_lumbar_path, volunteer_id)
    os.makedirs(output_pelvis_this_volunteer_path, exist_ok=True)
    os.makedirs(output_lumbar_this_volunteer_path, exist_ok=True)
    images_dixon = {}
    for dixon_tag in dixon_types:
        input_file_tag = os.path.join(input_folder, f"{volunteer_id}_{dixon_tag}_dixon_concatenated.nii.gz")
        if os.path.exists(input_file_tag):
            images_dixon[dixon_tag] = sitk.ReadImage(input_file_tag)
            # Apply bias correction if needed
            if dixon_tag == 'in':
                print(f"Applying bias correction to {input_file_tag}")
                images_dixon[dixon_tag] = apply_bias_correction(images_dixon[dixon_tag], shrinkFactor=(8, 8, 4))
            # todo: get the field and apply it to the other images
        else:
            print(f"Warning: {input_file_tag} not found.")

        # --------------------------- CROP PELVIS REGION ---------------------------
        sitk_pelvis_image = images_dixon[dixon_tag][:,:, int(trochanter):int(iliac_crest)]
        dixon_index = dixon_types.index(dixon_tag)
        sitk.WriteImage(sitk_pelvis_image, f"{output_pelvis_this_volunteer_path}/{volunteer_id}_{dixon_output_tag[dixon_index]}.nii.gz")


        # --------------------------- CROP LUMBAR REGION ---------------------------
        sitk_lumbar_image = images_dixon[dixon_tag][:,:, int(trochanter):int(vertebra_L1)]
        sitk.WriteImage(sitk_lumbar_image, f"{output_lumbar_this_volunteer_path}/{volunteer_id}_{dixon_output_tag[dixon_index]}.nii.gz")

# --------------------------- END OF SCRIPT ---------------------------
print("\nEND OF SCRIPT")
