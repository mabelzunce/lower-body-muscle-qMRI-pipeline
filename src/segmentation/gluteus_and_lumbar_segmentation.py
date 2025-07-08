import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from unet_3d import Unet
from utils import multilabel, maxProb

# --------------------------- CONFIG PATHS ---------------------------
input_root = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifty_output/'
output_root = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/segmentations/'
os.makedirs(output_root, exist_ok=True)
coord_csv = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/slices_cortes_anatomicos.csv'

# Model paths
gluteus_model_path = '/data/MuscleSegmentation/Data/GluteusPelvis3D/model/gluteus_model.pt'
lumbar_model_path = '/data/MuscleSegmentation/Data/LumbarSpine3D/model/lumbar_model.pt'

# --------------------------- LOAD COORDINATES ---------------------------
coords_df = pd.read_csv(coord_csv)

# --------------------------- DEVICE SETUP ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------- LOAD MODELS ---------------------------
multilabelNum = 8

# # Gluteus model
# gluteus_model = Unet(1, multilabelNum)
# gluteus_model.load_state_dict(torch.load(gluteus_model_path, map_location=device))
# gluteus_model = gluteus_model.to(device)
# gluteus_model.eval()

# # Lumbar model
# lumbar_model = Unet(1, multilabelNum)
# lumbar_model.load_state_dict(torch.load(lumbar_model_path, map_location=device))
# lumbar_model = lumbar_model.to(device)
# lumbar_model.eval()

# --------------------------- PROCESS EACH VOLUNTEER ---------------------------
for idx, row in coords_df.iterrows():
    volunteer_id = row['ID']
    trochanter = int(row['Trocánter menor'])
    iliac_crest = int(row['Cresta iliaca'])
    vertebra_L1 = int(row['Vértebra L1'])

    input_folder = os.path.join(input_root, volunteer_id)
    input_file = os.path.join(input_folder, f"{volunteer_id}_OPP_concatenated.nii.gz")

    # --------------------------- LOAD IMAGE ---------------------------
    sitk_image = sitk.ReadImage(input_file)
    image = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

    print(f"\n=== Processing volunteer: {volunteer_id} ===")
    print(f"Original image shape: {image.shape}")

    # --------------------------- CROP GLUTEUS REGION ---------------------------
    gluteus_volume = image[trochanter:iliac_crest, :, :]
    gluteus_volume = np.expand_dims(gluteus_volume, axis=0)

    print(f"Gluteus crop slices: {trochanter} to {iliac_crest} | Cropped shape: {gluteus_volume.shape}")

    # Save cropped gluteus region for QC before segmentation
    gluteus_crop_image = sitk.GetImageFromArray(gluteus_volume.squeeze(0))
    gluteus_crop_output_path = os.path.join(output_root, f"{volunteer_id}_F_gluteus_crop.nii.gz")
    sitk.WriteImage(gluteus_crop_image, gluteus_crop_output_path)

    # --------------------------- CROP LUMBAR REGION ---------------------------
    lumbar_volume = image[trochanter:vertebra_L1, :, :]
    lumbar_volume = np.expand_dims(lumbar_volume, axis=0)

    print(f"Lumbar crop slices: {trochanter} to {vertebra_L1} | Cropped shape: {lumbar_volume.shape}")

    # Save cropped lumbar region for QC before segmentation
    lumbar_crop_image = sitk.GetImageFromArray(lumbar_volume.squeeze(0))
    lumbar_crop_output_path = os.path.join(output_root, f"{volunteer_id}_F_lumbar_crop.nii.gz")
    sitk.WriteImage(lumbar_crop_image, lumbar_crop_output_path)

    print(f"Saved cropped images for volunteer: {volunteer_id}")

# --------------------------- END OF SCRIPT ---------------------------
print("\\nPipeline completed.")
