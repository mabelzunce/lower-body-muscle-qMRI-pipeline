import os, sys, csv
import numpy as np, matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk, torch, imageio
import DixonTissueSegmentation
from PIL import Image
from skimage.morphology import convex_hull_image
from unet_3d import Unet
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
from utils import create_segmentation_overlay_animated_gif, apply_bias_correction, multilabel, maxProb, FilterUnconnectedRegions, write_vol_ff_simple_csv
dixon_types = ['in', 'opp', 'f', 'w']
dixon_output_tag = ['I', 'O', 'F', 'W']

# --------------------------- CONFIG PATHS  ---------------------------
input_root = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifty_output/'
outputPath = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/segmentations/'
output_pelvis_path = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifti_pelvis/'
output_lumbar_path = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifti_lumbar/'
os.makedirs(outputPath, exist_ok=True)
os.makedirs(output_pelvis_path, exist_ok=True)
os.makedirs(output_lumbar_path, exist_ok=True)
coord_csv = '/home/german/lower-body-muscle-qMRI-pipeline/data/mri_info.csv'
coords_df = pd.read_csv(coord_csv)

# Modelos
lumbar_model_path  = "/home/german/lower-body-muscle-qMRI-pipeline/models/lumbarspine_unet3d_20230626_191618_173_best_fit.pt"
gluteal_model_path = "/home/german/lower-body-muscle-qMRI-pipeline/models/gluteal_unet3d_20250807_110716_123_best_fit.pt"

# Imágenes de referencia
lumbar_reference_path  = "/data/MuscleSegmentation/Data/LumbarSpine3D/ResampledData/C00001.mhd"
#lumbar_reference_path  = '/home/german/lower-body-muscle-qMRI-pipeline/data/reference_images/lumbar_spine_reference.nii.gz'
gluteus_reference_path = "/home/german/lower-body-muscle-qMRI-pipeline/data/reference_images/pelvis_reference.nii.gz"

# CONFIGURATION:
device_to_use = 'cuda' #'cpu'
preRegistration = True #TRUE: Pre-register using the next image
dataInSubdirPerSubject = True

imageNames = []
imageFilenames = []
fat_fraction_all_subjects = list() #List to then write the .csv file
volume_all_subjects = list() #List to then write the .csv file
totalvolume_all_subjects = list()
meanff_all_subjects = list()
names_subjects = list()
i = 0

# REGISTRATION PARAMETER FILES:
similarityMetricForReg = 'NMI' #NMI Metric
parameterFilesPath = '/data/MuscleSegmentation/Data/Elastix/' #Parameters path
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg

#IMAGE FORMATS AND EXTENSION:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.mhd' #'.nii.gz'
tagInPhase = '_I'

# outOfPhaseSuffix
waterSuffix = '_W'
fatSuffix = '_F'
tagAutLabels = '_aut'
tagManLabels = '_labels'

#CHECK DEVICE:
device = torch.device(device_to_use) #'cuda' uses the graphic board
print(device)
if device.type == 'cuda':
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Total memory: {0}. Reserved memory: {1}. Allocated memory:{2}. Free memory:{3}.'.format(t,r,a,f))

# Read images:
referenceImage_lumbar  = sitk.ReadImage(lumbar_reference_path)
referenceImage_gluteus = sitk.ReadImage(gluteus_reference_path)

# Parameters for image registration:
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.ElastixImageFilter().ReadParameterFile(parameterFilesPath + paramFileRigid + '.txt'))
print('Lumbar reference voxel size:', referenceImage_lumbar.GetSize())
print('Gluteus reference voxel size:', referenceImage_gluteus.GetSize())

# MODEL INIT:
multilabelNum = 8
torch.cuda.empty_cache()
lumbarModel = Unet(1, multilabelNum).to(device)
lumbarModel.load_state_dict(torch.load(lumbar_model_path, map_location=device))
glutealModel = Unet(1, multilabelNum).to(device)
glutealModel.load_state_dict(torch.load(gluteal_model_path, map_location=device))

# --------------------------- PROCESS EACH VOLUNTEER ---------------------------

#for idx, row in coords_df.iterrows():
for idx, row in coords_df.iloc[0:1].iterrows():
    inPhaseImageLumbar, fatImageLumbar, waterImageLumbar = None, None, None
    inPhaseImagePelvis, fatImagePelvis, waterImagePelvis = None, None, None
    ffLumbar, ffPelvis = None, None
    volunteer_id = row['ID']
    subject = volunteer_id
    outputPathThisSubject = os.path.join(outputPath, volunteer_id )
    os.makedirs(outputPathThisSubject, exist_ok=True)
    trochanter = int(row['Lesser Trochanter'])
    iliac_crest = int(row['Iliac Crest'])
    vertebra_L1 = int(row['L1'])
    print(f"\n=== Processing volunteer: {volunteer_id} ===\nTLesser trochanter: {trochanter}\nTop Iliac Crest: {iliac_crest}\nL1: {vertebra_L1}")

    # --------------------------- LOAD IMAGE ---------------------------
    input_folder = os.path.join(input_root, volunteer_id)
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
        if dixon_tag == 'in':
            inPhaseImagePelvis = sitk_pelvis_image
        elif dixon_tag == 'f':
            fatImagePelvis = sitk_pelvis_image
        elif dixon_tag == 'w':
            waterImagePelvis = sitk_pelvis_image
        elif dixon_tag == 'opp':
            outOfPhaseImagePelvis = sitk_pelvis_image

        # --------------------------- CROP LUMBAR REGION ---------------------------
        sitk_lumbar_image = images_dixon[dixon_tag][:,:, int(trochanter):int(vertebra_L1)]
        sitk.WriteImage(sitk_lumbar_image, f"{output_lumbar_this_volunteer_path}/{volunteer_id}_{dixon_output_tag[dixon_index]}.nii.gz")
        if dixon_tag == 'in':
            inPhaseImageLumbar = sitk_lumbar_image
        elif dixon_tag == 'f':
            fatImageLumbar = sitk_lumbar_image
        elif dixon_tag == 'w':
            waterImageLumbar = sitk_lumbar_image
        # --------------------------- END OF CROP --------------------------

    # -------------------- TISSUE / SUBCUTANEOUS FAT SEGMENTATION --------------------
    print("Running Dixon tissue segmentation...")

    # 1) Generate the tissue segmented image
    dixonImages_list = [
        images_dixon['in'],
        images_dixon['opp'],
        images_dixon['w'],
        images_dixon['f']
    ]
    segmentedImage = DixonTissueSegmentation.DixonTissueSegmentation(dixonImages_list)
    sitk.WriteImage(
        segmentedImage,
        os.path.join(outputPathThisSubject, f"{subject}_tissue_segmented{extensionImages}"),
        True
    )

    # 2) Body mask (from the fat image, usually more robust)
    bodyMask = DixonTissueSegmentation.GetBodyMaskFromFatDixonImage(
        images_dixon['f'], vectorRadius=(2, 2, 1)
    )
    sitk.WriteImage(
        bodyMask,
        os.path.join(outputPathThisSubject, f"{subject}_bodyMask{extensionImages}"),
        True
    )

    # --- GIF de BodyMask ---
    image_path = os.path.join(input_folder, f"{subject}_in_dixon_concatenated.nii.gz")
    mask_path = os.path.join(outputPathThisSubject, f"{subject}_bodyMask{extensionImages}")
    gif_output = os.path.join(outputPathThisSubject, f"{subject}_bodyMask_overlay.gif")

    sitkImage = sitk.ReadImage(image_path)
    sitkMask = sitk.ReadImage(mask_path)
    sitkMask.CopyInformation(sitkImage)  # asegurar mismo espacio

    create_segmentation_overlay_animated_gif(sitkImage, sitkMask, gif_output)


    # 3) Subcutaneous fat mask (using convex hull, slice by slice)
    skinFat = DixonTissueSegmentation.GetSkinFatFromTissueSegmentedImageUsingConvexHullPerSlice(segmentedImage)
    skinFat = sitk.And(skinFat, bodyMask)  # remove artefacts outside the body
    sitk.WriteImage(skinFat,
        os.path.join(outputPathThisSubject, f"{subject}_skinFat{extensionImages}"),
        True
    )

    # --- Subcutaneous fat mask for pelvis crop ---
    print("Running subcutaneous fat mask for pelvis crop...")
    # Generar la segmentación de tejidos solo en el recorte de pelvis
    dixonImages_pelvis = [inPhaseImagePelvis,outOfPhaseImagePelvis,waterImagePelvis,fatImagePelvis]
    segmentedPelvis = DixonTissueSegmentation.DixonTissueSegmentation(dixonImages_pelvis)
    # Body mask en el recorte de pelvis (desde fat)
    bodyMaskPelvis = DixonTissueSegmentation.GetBodyMaskFromFatDixonImage(fatImagePelvis, vectorRadius=(2, 2, 1))

    # SkinFat en pelvis
    skinFatPelvis = DixonTissueSegmentation.GetSkinFatFromTissueSegmentedImageUsingConvexHullPerSlice(segmentedPelvis)
    skinFatPelvis = sitk.And(skinFatPelvis, bodyMaskPelvis)

    # Guardar
    sitk.WriteImage(
        skinFatPelvis,
        os.path.join(outputPathThisSubject, f"{subject}_pelvis_skinFat{extensionImages}"),
        True
    )

    # 4) Muscle mask
    muscleMask = DixonTissueSegmentation.GetMuscleMaskFromTissueSegmentedImage(
        segmentedImage, vectorRadius=(4, 4, 3)
    )
    sitk.WriteImage(
        muscleMask,
        os.path.join(outputPathThisSubject, f"{subject}_muscleMask{extensionImages}"),
        True
    )

    # Generate GIF
    image_path = os.path.join("/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifti_pelvis/", f"{subject}", f"{subject}_I.nii.gz")
    mask_path = os.path.join(outputPathThisSubject, f"{subject}_pelvis_skinFat{extensionImages}")
    gif_output = os.path.join(outputPathThisSubject, f"{subject}_pelvis_skinFat_overlay.gif")
    # Load image and mask
    sitkImage = sitk.ReadImage(image_path)
    sitkMask = sitk.ReadImage(mask_path)
    # Make sure mask has same metadata as image
    sitkMask.CopyInformation(sitkImage)
    create_segmentation_overlay_animated_gif(sitkImage, sitkMask, gif_output)

    # Generate GIF

    # Paths dinámicos para este voluntario
    image_path = os.path.join(input_folder, f"{subject}_in_dixon_concatenated.nii.gz")
    mask_path = os.path.join(outputPathThisSubject, f"{subject}_skinFat{extensionImages}")
    gif_output = os.path.join(outputPathThisSubject, f"{subject}_skinFat_overlay.gif")
    # Load image and mask
    sitkImage = sitk.ReadImage(image_path)
    sitkMask = sitk.ReadImage(mask_path)
    # Make sure mask has same metadata as image
    sitkMask.CopyInformation(sitkImage)
    create_segmentation_overlay_animated_gif(sitkImage, sitkMask, gif_output)

    # === VOLUMES OF SUBCUTANEOUS FAT ===

    # Volumen de skinFat total
    skinFat_array = sitk.GetArrayFromImage(skinFat)
    voxel_volume = np.prod(skinFat.GetSpacing())
    skinFat_total_vol = np.sum(skinFat_array > 0) * voxel_volume

    # Volumen de skinFat en pelvis
    skinFat_pelvis_array = sitk.GetArrayFromImage(skinFatPelvis)
    voxel_volume_pelvis = np.prod(skinFatPelvis.GetSpacing())
    skinFat_pelvis_vol = np.sum(skinFat_pelvis_array > 0) * voxel_volume_pelvis

    # -------------------- FAT FRACTION CALCULATION --------------------

    # LUMBAR
    if (fatImageLumbar is not None) and (waterImageLumbar is not None):
        fatImageLumbar = sitk.Cast(fatImageLumbar, sitk.sitkFloat32)
        waterImageLumbar = sitk.Cast(waterImageLumbar, sitk.sitkFloat32)
        waterfatLumbar = sitk.Add(fatImageLumbar, waterImageLumbar)
        ffLumbar = sitk.Divide(fatImageLumbar, waterfatLumbar)
        ffLumbar = sitk.Cast(
            sitk.Mask(ffLumbar, waterfatLumbar > 0, outsideValue=0, maskingValue=0),
            sitk.sitkFloat32
        )
        sitk.WriteImage(ffLumbar, os.path.join(outputPathThisSubject, f"{subject}_lumbar_ff{extensionImages}"), True)
    else:
        print(f"[WARN] Missing W and/or F lumbar images for {subject} — skipping lumbar FF.")

    # PELVIS
    if (fatImagePelvis is not None) and (waterImagePelvis is not None):
        fatImagePelvis = sitk.Cast(fatImagePelvis, sitk.sitkFloat32)
        waterImagePelvis = sitk.Cast(waterImagePelvis, sitk.sitkFloat32)
        waterfatPelvis = sitk.Add(fatImagePelvis, waterImagePelvis)
        ffPelvis = sitk.Divide(fatImagePelvis, waterfatPelvis)
        ffPelvis = sitk.Cast(
            sitk.Mask(ffPelvis, waterfatPelvis > 0, outsideValue=0, maskingValue=0),
            sitk.sitkFloat32
        )
        sitk.WriteImage(ffPelvis, os.path.join(outputPathThisSubject, f"{subject}_pelvis_ff{extensionImages}"), True)
    else:
        print(f"[WARN] Missing W and/or F pelvis images for {subject} — skipping pelvis FF.")

    print(f"Lumbar segmentation")

    # Input image for the segmentation:
    if inPhaseImageLumbar != 0:
        sitkImage = inPhaseImageLumbar
    else:
        # use fat image that is similar:
        sitkImage = fatImageLumbar

    # Get the spacial dimensions
    spacing = sitkImage.GetSpacing()  # Tuple (spacing_x, spacing_y, spacing_z)
    print(spacing)
    # Apply Bias Field Correction
    shrinkFactor = (4, 4, 2)
    sitkImage = ApplyBiasCorrection(sitkImage, shrinkFactor=shrinkFactor)
    # Obtains the name of the file (without the complete path and divide name and extension)
    filename_no_ext = subject
    file_extension = ".nii.gz"
    new_filename = f"{filename_no_ext}_biasFieldCorrection{file_extension}"
    outputBiasFilename = os.path.join(outputPathThisSubject, new_filename)
    sitk.WriteImage(sitkImage, outputBiasFilename, True)

    if preRegistration:
        # elastixImageFilter filter
        elastixImageFilter = sitk.ElastixImageFilter() #Create the object
        # Register image to reference data
        elastixImageFilter.SetFixedImage(referenceImage_lumbar) #Defines reference image
        elastixImageFilter.SetMovingImage(sitkImage) #Defines moving image
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.SetLogToConsole(False)
        elastixImageFilter.Execute()
        transform = elastixImageFilter.GetParameterMap()
        sitkImageResampled = elastixImageFilter.GetResultImage() #Result image from the register
        # Write transformed image:
        elastixImageFilter.WriteParameterFile(transform[0], 'transform.txt')
    else:
        sitkImageResampled = sitkImage
    # Convert to float and register it:
    image = sitk.GetArrayFromImage(sitkImageResampled).astype(np.float32)
    image = np.expand_dims(image, axis=0)

    # Run the segmentation through the model:
    torch.cuda.empty_cache()
    with torch.no_grad(): #SEGMENTATION:
        input = torch.from_numpy(image).to(device)
        output = lumbarModel(input.unsqueeze(0))
        output = torch.sigmoid(output.cpu().to(torch.float32))
        outputs = maxProb(output, multilabelNum)
        output = ((output > 0.5) * 1)
        output = multilabel(output.detach().numpy())
    output = FilterUnconnectedRegions(output.squeeze(0), multilabelNum, sitkImageResampled)

    if preRegistration:
        # Resample to original space:
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetInitialTransformParameterFileName('TransformParameters.0.txt')
        elastixImageFilter.SetFixedImage(sitkImageResampled) # sitkImage
        elastixImageFilter.SetMovingImage(sitkImageResampled) # sitkImage
        elastixImageFilter.LogToConsoleOff()
        # rigid_pm = affine_parameter_map()
        rigid_pm = sitk.GetDefaultParameterMap("affine")
        rigid_pm['MaximumNumberOfIterations'] = ("1000",) # By default 256, but it's not enough
        # rigid_pm["AutomaticTransformInitialization"] = "true"
        # rigid_pm["AutomaticTransformInitializationMethod"] = ["Origins"]
        elastixImageFilter.SetParameterMap(rigid_pm)
        elastixImageFilter.SetParameter('HowToCombineTransforms', 'Compose')
        elastixImageFilter.SetParameter('Metric', 'DisplacementMagnitudePenalty')

        elastixImageFilter.Execute()

        Tx = elastixImageFilter.GetTransformParameterMap()
        Tx[0]['InitialTransformParametersFileName'] = ('NoInitialTransform',)
        Tx[0]['Origin'] = tuple(map(str, sitkImage.GetOrigin()))
        Tx[0]['Spacing'] = tuple(map(str, sitkImage.GetSpacing()))
        Tx[0]['Size'] = tuple(map(str, sitkImage.GetSize()))
        Tx[0]['Direction'] = tuple(map(str, sitkImage.GetDirection()))

        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(Tx)
        transformixImageFilter.SetMovingImage(output)
        transformixImageFilter.SetLogToConsole(False)
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        output = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)

    # Enforce the same space in the raw image (there was a bug before, without this they match in geometrical space but not in voxel space):
    output = sitk.Resample(output, sitkImage, sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
    output_single_mask = output > 0 #Binary segmentation
    sitk.WriteImage(output, os.path.join(outputPathThisSubject, subject + '_lumbar_segmentation' + extensionImages), True)
    sitk.WriteImage(output_single_mask, os.path.join(outputPathThisSubject, subject + '_lumbar_spine_mask' + extensionImages), True) #The binary segmentation will be called 'subject_lumbar_spine_mask.mhd'

    #VOLUME CALCULATION:
    #Segmentation to array
    segmentation_array = sitk.GetArrayFromImage(output)
    num_labels = multilabelNum
    voxel_volume = np.prod(spacing) #volume (X.Y.Z)

        # Volume dictionary to save the label volumes
    volumes = {}

    # Iterate over labels
    for label in range(1, num_labels+1):
        label_mask = (segmentation_array == label).astype(np.uint8)
        label_voxels = np.sum(label_mask)
        label_volume = label_voxels * voxel_volume
        volumes[label] = label_volume       # Save the data on the dictionary
    #Add them to the list
    volume_all_subjects.append(volumes)
    # Print the volume of all the labels:
    print("\nVolumes:")
    for label, volume in volumes.items():
        print(f"Muscle {label}: {volume} mm³")

    # WRITE AN ANIMATED GIF WITH THE SEGMENTATION
    create_segmentation_overlay_animated_gif(sitkImage, output, os.path.join(outputPathThisSubject, f"{subject}_lumbar_segmentation_check.gif"))

    #FF CALCULATION:

    # Images to numpy arrays
    fatfraction_array = sitk.GetArrayFromImage(ffLumbar)
    fat_fraction_means = {}
    fat_fraction_means_pelvis = {}

    for label in range(1, multilabelNum+1):
        # Mask for actual label
        label_mask = (segmentation_array == label)

        # Check for true values
        if np.any(label_mask):
            # Get the FF values linked to the mask
            fat_values = fatfraction_array[label_mask]

            # Calculate mean value for the label
            fat_fraction_means[label] = np.mean(fat_values)
        else:
            fat_fraction_means[label] = None  # No values for that label

    # Add them to the list:
    fat_fraction_all_subjects.append(fat_fraction_means)

    # Write on the csv file
    #write_ff_to_csv(os.path.join(outputPathThisSubject, 'ffs_lumbar.csv'), fat_fraction_means, subject)
    #write_ff_to_csv(os.path.join(outputPath, ff_lumbar_all), fat_fraction_means, subject)

    # Print the results
    print("\nFFs:")
    #for label in range(0,9):
    for label, fat_mean in fat_fraction_means.items():
        if fat_mean is not None:
            print(f"Muscle {label}: {fat_mean:.4f}")
        else:
            print(f"Muscle {label}: Sin valores válidos")

    ############     PELVIS   #######################################################

    print(f"Pelvis segmentation")
    # 1) Imagen de entrada (pelvis)
    if inPhaseImagePelvis is not None:
        sitkImagePelvis = inPhaseImagePelvis
    else:
        sitkImagePelvis = fatImagePelvis  # fallback

    # 2) Espaciado y bias field correction
    spacingPelvis = sitkImagePelvis.GetSpacing()
    print("Pelvis spacing:", spacingPelvis)
    sitkImagePelvis = ApplyBiasCorrection(sitkImagePelvis, shrinkFactor=(4, 4, 2))

    # Guardar la pelvis corregida (opcional, mismo formato que arriba)
    pelvis_bias_fname = os.path.join(outputPathThisSubject, f"{subject}_pelvis_biasFieldCorrection.nii.gz")
    sitk.WriteImage(sitkImagePelvis, pelvis_bias_fname, True)

    # 3) Registro (usar referencia de glúteo)
    if preRegistration:
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(referenceImage_gluteus)  # <--- referencia GLÚTEO
        elastixImageFilter.SetMovingImage(sitkImagePelvis)  # <--- imagen PELVIS
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.SetLogToConsole(False)
        elastixImageFilter.Execute()
        transform = elastixImageFilter.GetParameterMap()
        sitkImageResampledPelvis = elastixImageFilter.GetResultImage()
        elastixImageFilter.WriteParameterFile(transform[0], 'transform.txt')
    else:
        sitkImageResampledPelvis = sitkImagePelvis

    # 4) Preparar tensor y segmentar con el MODELO DE GLÚTEO
    imagePelvis = sitk.GetArrayFromImage(sitkImageResampledPelvis).astype(np.float32)
    imagePelvis = np.expand_dims(imagePelvis, axis=0)

    torch.cuda.empty_cache()
    with torch.no_grad():
        input_t = torch.from_numpy(imagePelvis).to(device)
        outputPelvis = glutealModel(input_t.unsqueeze(0))  # <--- modelo GLÚTEO
        outputPelvis = torch.sigmoid(outputPelvis.cpu().to(torch.float32))
        _ = maxProb(outputPelvis, multilabelNum)
        outputPelvis = ((outputPelvis > 0.5) * 1)
        outputPelvis = multilabel(outputPelvis.detach().numpy())

    outputPelvis = FilterUnconnectedRegions(outputPelvis.squeeze(0), multilabelNum, sitkImageResampledPelvis)

    # 5) Volver al espacio original de la pelvis (si hubo registro)
    if preRegistration:
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetInitialTransformParameterFileName('TransformParameters.0.txt')
        elastixImageFilter.SetFixedImage(sitkImageResampledPelvis)
        elastixImageFilter.SetMovingImage(sitkImageResampledPelvis)
        elastixImageFilter.LogToConsoleOff()
        rigid_pm = sitk.GetDefaultParameterMap("affine")
        rigid_pm['MaximumNumberOfIterations'] = ("1000",)
        elastixImageFilter.SetParameterMap(rigid_pm)
        elastixImageFilter.SetParameter('HowToCombineTransforms', 'Compose')
        elastixImageFilter.SetParameter('Metric', 'DisplacementMagnitudePenalty')
        elastixImageFilter.Execute()

        Tx = elastixImageFilter.GetTransformParameterMap()
        Tx[0]['InitialTransformParametersFileName'] = ('NoInitialTransform',)
        Tx[0]['Origin'] = tuple(map(str, sitkImagePelvis.GetOrigin()))
        Tx[0]['Spacing'] = tuple(map(str, sitkImagePelvis.GetSpacing()))
        Tx[0]['Size'] = tuple(map(str, sitkImagePelvis.GetSize()))
        Tx[0]['Direction'] = tuple(map(str, sitkImagePelvis.GetDirection()))

        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(Tx)
        transformixImageFilter.SetMovingImage(outputPelvis)
        transformixImageFilter.SetLogToConsole(False)
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        outputPelvis = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)

    # Asegurar mismo voxel grid que la pelvis original
    outputPelvis = sitk.Resample(outputPelvis, sitkImagePelvis, sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)
    pelvis_single_mask = outputPelvis > 0

    # 6) Guardados con nombres de PELVIS
    sitk.WriteImage(outputPelvis,
                    os.path.join(outputPathThisSubject, subject + '_pelvis_segmentation' + extensionImages), True)
    sitk.WriteImage(pelvis_single_mask, os.path.join(outputPathThisSubject, subject + '_pelvis_mask' + extensionImages),
                    True)

    # 7) Volúmenes (pelvis)
    segmentation_array_pelvis = sitk.GetArrayFromImage(outputPelvis)
    voxel_volume_pelvis = np.prod(spacingPelvis)
    volumes_pelvis = {}
    for label in range(1, multilabelNum + 1):
        label_vox = np.sum((segmentation_array_pelvis == label).astype(np.uint8))
        volumes_pelvis[label] = label_vox * voxel_volume_pelvis

    #write_volumes_to_csv(os.path.join(outputPathThisSubject, 'volumes_pelvis.csv'), volumes_pelvis, subject)
    #write_volumes_to_csv(os.path.join(outputPath, vol_pelvis_all), volumes_pelvis, subject)

    print("\nVolumes (pelvis):")
    for label, volume in volumes_pelvis.items():
        print(f"Pelvis label {label}: {volume} mm³")

    # 8) GIF de control (pelvis)
    create_segmentation_overlay_animated_gif(
        sitkImagePelvis,
        outputPelvis,
        os.path.join(outputPathThisSubject, f"{subject}_pelvis_segmentation_check.gif")
    )

    # 9) FF por etiqueta (pelvis) — usa la FF calculada arriba para pelvis (ffPelvis)
    if ffPelvis is not None:
        # --- FF por etiqueta ---
        fatfraction_array_pelvis = sitk.GetArrayFromImage(ffPelvis)
        fat_fraction_means_pelvis = {}
        for label in range(1, multilabelNum + 1):
            mask_l = (segmentation_array_pelvis == label)
            fat_fraction_means_pelvis[label] = float(np.mean(fatfraction_array_pelvis[mask_l])) if np.any(
                mask_l) else None

        # ---- AGREGAR DATOS DE PELVIS A LAS LISTAS GLOBALES (ahora que ya hay FF) ----

        volume_all_subjects.append(volumes_pelvis)

        # FF media y volumen total de la máscara pelvis
        pelvis_mask_array = sitk.GetArrayFromImage(pelvis_single_mask)
        total_volume_pelvis = np.sum(pelvis_mask_array) * voxel_volume_pelvis
        totalvolume_all_subjects.append(total_volume_pelvis)
        masked_values_pelvis = fatfraction_array_pelvis[pelvis_mask_array > 0]
        mean_ff_pelvis = np.mean(masked_values_pelvis) if masked_values_pelvis.size > 0 else None
        meanff_all_subjects.append(mean_ff_pelvis)
        fat_fraction_all_subjects.append(fat_fraction_means_pelvis)

        # nombre diferenciado para el CSV grande
        names_subjects.append(subject + "_pelvis")
    else:
        print("[WARN] No se encontró ffPelvis para calcular FF por etiqueta en pelvis.")

    #TOTAL VOLUME:
    single_array = sitk.GetArrayFromImage(output_single_mask)
    num_segmented_voxels = np.sum(single_array)
    total_volume = num_segmented_voxels * voxel_volume #TOTAL VOLUME FOR THAT SUBJECT
    totalvolume_all_subjects.append(total_volume)

    #MEAN FF:
    # Apply the mask
    masked_values = fatfraction_array[single_array > 0]
    # Calculate the mean ff
    mean_ff = np.mean(masked_values) #MEAN FF FOR THAT SUBJECT
    meanff_all_subjects.append(mean_ff)

    #Name
    names_subjects.append(subject + "_lumbar")


    # --- CSV this suject ---
    write_vol_ff_simple_csv(
        os.path.join(outputPathThisSubject, "volumes_and_ffs.csv"),
        volumes, fat_fraction_means,  # lumbar
        volumes_pelvis, fat_fraction_means_pelvis,  # pelvis
        skinfat_total=skinFat_total_vol,
        skinfat_pelvis=skinFat_pelvis_vol,
        subject_name=subject)

    # --- CSV global  ---

    # Y lo mismo para el CSV global
    write_vol_ff_simple_csv(
        os.path.join(outputPath, "all_subjects_volumes_and_ffs.csv"),
        volumes, fat_fraction_means,
        volumes_pelvis, fat_fraction_means_pelvis,
        skinfat_total=skinFat_total_vol,
        skinfat_pelvis=skinFat_pelvis_vol,
        subject_name=subject)
