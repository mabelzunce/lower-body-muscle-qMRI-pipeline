import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
torch.cuda.empty_cache()
#from sympy import false
from unet_3d import Unet
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import multilabel, maxProb, apply_bias_correction, FilterUnconnectedRegions

# --------------------------- CONFIG PATHS ---------------------------
input_root = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx//nifti_output/'
output_root = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx//segmentations/'
output_pelvis_path = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx//nifti_pelvis/'
os.makedirs(output_root, exist_ok=True)
os.makedirs(output_pelvis_path, exist_ok=True)
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

# --------------------------- DEVICE SETUP ---------------------------
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.cuda.set_device(1) # Forzar uso de GPU 1, que está libre
device = torch.device('cpu')


# --------------------------- LOAD MODELS an CREATE FILES---------------------------
multilabelNum = 8

print("\nLoading models...")

# Gluteus model
gluteus_model = Unet(1, multilabelNum)
gluteus_model.load_state_dict(torch.load(gluteus_model_path, map_location=device))
gluteus_model = gluteus_model.to(device)
gluteus_model.eval()

# Lumbar model
lumbar_model = Unet(1, multilabelNum)
lumbar_model.load_state_dict(torch.load(lumbar_model_path, map_location=device))
lumbar_model = lumbar_model.to(device)
lumbar_model.eval()

print("Models loaded.\n")

# --------------------------- PROCESS EACH VOLUNTEER ---------------------------
for idx, row in coords_df.iterrows():
    volunteer_id = row['ID']
    outputPathThisSubject = os.path.join(output_root, volunteer_id )
    os.makedirs(outputPathThisSubject, exist_ok=True)
    trochanter = int(row['Trocánter menor'])
    iliac_crest = int(row['Cresta iliaca'])
    vertebra_L1 = int(row['Vértebra L1'])
    print(f"\n=== Processing volunteer: {volunteer_id} ===\nTrocánter menor: {trochanter}\nCresta iliaca: {iliac_crest}\nVértebra L1: {vertebra_L1}")

    # --------------------------- LOAD IMAGE ---------------------------
    input_folder = os.path.join(input_root, volunteer_id)
    output_pelvis_this_volunteer_path = os.path.join(output_pelvis_path, volunteer_id)
    os.makedirs(os.path.join(output_pelvis_path, volunteer_id), exist_ok=True)
    images_dixon = {}
    for dixon_tag in dixon_types:
        input_file_tag = os.path.join(input_folder, f"{volunteer_id}_{dixon_tag}_dixon_concatenated.nii.gz")
        if os.path.exists(input_file_tag):
            images_dixon[dixon_tag] = sitk.ReadImage(input_file_tag)
        else:
            print(f"Warning: {input_file_tag} not found.")

        # --------------------------- CROP PELVIS REGION ---------------------------
        #original_z = image.shape[0]
        crop_start = trochanter
        crop_end = iliac_crest
        sitk_pelvis_image = images_dixon[dixon_tag][:,:, int(trochanter):int(iliac_crest)]
        dixon_index = dixon_types.index(dixon_tag)
        sitk.WriteImage(sitk_pelvis_image, f"{output_pelvis_this_volunteer_path}/{volunteer_id}_{dixon_output_tag[dixon_index]}.nii.gz")

        print(f"\nCropped gluteus region:")
        print(f"  Z range: {crop_start} to {crop_end} → slices kept: {crop_end - crop_start}")
        print("  Dimensions:   ", sitk_pelvis_image.GetDimension())
        print("  Spacing:   ", sitk_pelvis_image.GetSpacing())
        print("  Origin:    ", sitk_pelvis_image.GetOrigin())
        print("  Direction: ", sitk_pelvis_image.GetDirection())

    # --------------------------- FAT FRACTION PELVIS ---------------------------

    # --------------------------- REGISTRATION (PELVIS) ---------------------------
    parameterMapVector = sitk.VectorOfParameterMap()
    rigid_map = sitk.GetDefaultParameterMap("rigid")
    rigid_map['AutomaticTransformInitialization'] = ['true']
    rigid_map['AutomaticTransformInitializationMethod'] = ['CenterOfGravity']
    parameterMapVector.append(rigid_map)
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(reference_image_gluteus)
    elastixImageFilter.SetMovingImage(images_dixon[0]) # in phase
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.SetLogToConsole(False)
    try:
        elastixImageFilter.Execute()
        registered_image = elastixImageFilter.GetResultImage()
        sitk.WriteImage(registered_image, os.path.join(input_folder, "gluteus_registered.nii.gz"))
        elastixImageFilter.WriteParameterFile(elastixImageFilter.GetParameterMap()[0],os.path.join(input_folder, "transform_gluteus.txt"))
        print("Registración completada y guardada.")
    except RuntimeError as e:
        print("Error en la registración:", str(e))

    # --------------------------- SEGMENT PELVIS WITH TRAINED UNET ---------------------------
    print("\n⏳ Segmentando glúteos con UNet...")

    image = sitk.GetArrayFromImage(registered_image).astype(np.float32)  # [Z,Y,X]
    image = np.expand_dims(image, axis=0)

    torch.cuda.empty_cache()
    with torch.no_grad():
        input = torch.from_numpy(image).to(device)
        output = gluteus_model(input.unsqueeze(0))
        output = torch.sigmoid(output.cpu().to(torch.float32))
        outputs = maxProb(output, multilabelNum)
        output = ((output > 0.5) * 1)
        output = multilabel(output.detach().numpy())  # -> shape: [1, Z, Y, X] o [C, Z, Y, X]
        output = np.squeeze(output, axis=0)  # -> shape: [Z, Y, X] ✔️
        output_image = sitk.GetImageFromArray(output.astype(np.uint8))
        output_image.CopyInformation(registered_image)
        sitk.WriteImage(output_image, os.path.join(outputPathThisSubject, volunteer_id + '_segmentation.mhd'))

    print("✅ Segmentación de glúteos completa.")

    # --------------------------- CROP LUMBAR REGION ---------------------------
    crop_start_lumbar = trochanter
    crop_end_lumbar = vertebra_L1
    original_z = image.shape[0]

    lower_crop_lumbar = [0, 0, crop_start_lumbar]
    upper_crop_lumbar = [0, 0, original_z - crop_end_lumbar]

    lumbar_image = sitk.Crop(sitk_image, lower_crop_lumbar, upper_crop_lumbar)
    sitk.WriteImage(lumbar_image, os.path.join(input_folder, "lumbar_crop_debug.nii.gz"))

    print(f"\nCropped lumbar region:")
    print(f"  Z range: {crop_start_lumbar} to {crop_end_lumbar} → slices kept: {crop_end_lumbar - crop_start_lumbar}")
    print("  Shape:", sitk.GetArrayFromImage(lumbar_image).shape)
    print("  Spacing:   ", lumbar_image.GetSpacing())
    print("  Origin:    ", lumbar_image.GetOrigin())
    print("  Direction: ", lumbar_image.GetDirection())

    # --------------------------- FAT FRACTION LUMBAR ---------------------------
    #
    # --------------------------- REGISTRATION (LUMBAR) ---------------------------
    parameterMapVector_lumbar = sitk.VectorOfParameterMap()
    rigid_map_lumbar = sitk.GetDefaultParameterMap("rigid")
    rigid_map_lumbar['AutomaticTransformInitialization'] = ['true']
    rigid_map_lumbar['AutomaticTransformInitializationMethod'] = ['CenterOfGravity']
    parameterMapVector_lumbar.append(rigid_map_lumbar)
    elastixImageFilter_lumbar = sitk.ElastixImageFilter()
    elastixImageFilter_lumbar.SetFixedImage(reference_image_lumbar)
    elastixImageFilter_lumbar.SetMovingImage(lumbar_image)
    elastixImageFilter_lumbar.SetParameterMap(parameterMapVector_lumbar)
    elastixImageFilter_lumbar.SetLogToConsole(False)
    try:
        elastixImageFilter_lumbar.Execute()
        registered_lumbar = elastixImageFilter_lumbar.GetResultImage()
        sitk.WriteImage(registered_lumbar, os.path.join(input_folder, "lumbar_registered.nii.gz"))
        elastixImageFilter_lumbar.WriteParameterFile(
            elastixImageFilter_lumbar.GetParameterMap()[0],
            os.path.join(input_folder, "transform_lumbar.txt")
        )
        print("✅ Registración lumbar completa y guardada.")
    except RuntimeError as e:
        print("❌ Error en la registración lumbar:", str(e))

    # --------------------------- SEGMENT LUMBAR WITH TRAINED UNET ---------------------------


# --------------------------- END OF SCRIPT ---------------------------
print("\nEND OF SCRIPT")
