import os, sys, csv
import numpy as np, matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk, torch, imageio
from PIL import Image
from skimage.morphology import convex_hull_image
from unet_3d import Unet
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
from utils import apply_bias_correction, multilabel, maxProb, FilterUnconnectedRegions

torch.cuda.empty_cache()
dixon_types = ['in', 'opp', 'f', 'w']
dixon_output_tag = ['I', 'O', 'F', 'W']

# --------------------------- FUNCTIONS  ---------------------------

# BIAS FIELD CORRECTION
def ApplyBiasCorrection(inputImage, shrinkFactor = (1,1,1)):
    # Bias correction filter:
    biasFieldCorrFilter = sitk.N4BiasFieldCorrectionImageFilter()
    mask = sitk.OtsuThreshold( inputImage, 0, 1, 100)
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)

    # Parameter for the bias corredtion filter:
    biasFieldCorrFilter.SetSplineOrder(3)
    biasFieldCorrFilter.SetConvergenceThreshold(0.0001)
    biasFieldCorrFilter.SetMaximumNumberOfIterations((50, 40, 30))

    if shrinkFactor != (1,1,1):
        # Shrink image and mask to accelerate:
        shrinkedInput = sitk.Shrink(inputImage, shrinkFactor)
        mask = sitk.Shrink(mask, shrinkFactor)


        #biasFieldCorrFilter.SetNumberOfThreads()
        #biasFieldCorrFilter.UseMaskLabelOff() # Because I'm having problems with the mask.
        # Run the filter:
        output = biasFieldCorrFilter.Execute(shrinkedInput, mask)
        # Get the field by dividing the output by the input:
        outputArray = sitk.GetArrayFromImage(output)
        shrinkedInputArray = sitk.GetArrayFromImage(shrinkedInput)
        biasFieldArray = np.ones(np.shape(outputArray), 'float32')
        biasFieldArray[shrinkedInputArray != 0] = outputArray[shrinkedInputArray != 0]/shrinkedInputArray[shrinkedInputArray != 0]
        biasFieldArray[shrinkedInputArray == 0] = 0
        # Generate bias field image:
        biasField = sitk.GetImageFromArray(biasFieldArray)
        biasField.SetSpacing(shrinkedInput.GetSpacing())
        biasField.SetOrigin(shrinkedInput.GetOrigin())
        biasField.SetDirection(shrinkedInput.GetDirection())

        # Now expand
        biasField = sitk.Resample(biasField, inputImage)

        # Apply to the image:
        output = sitk.Multiply(inputImage, biasField)
    else:
        #output = biasFieldCorrFilter.Execute(inputImage, mask)
        output = biasFieldCorrFilter.Execute(inputImage)
    # return the output:
    return output

#VOLUMES TO .CSV
def write_volumes_to_csv(output_csv_path, volumes, subject_name):

    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #if not file_exists:
            # Write headers if the file is new
            #headers = ['Subject'] + [f'Vol {label}' for label in sorted(volumes.keys())]
            #writer.writerow(headers)
        # Write info in a row (1st element: subject name, then: all the labels)
        row = [subject_name] + [volumes[label] for label in sorted(volumes.keys())]
        writer.writerow(row)

#FFS TO .CSV
def write_ff_to_csv(output_csv_path, ffs, subject_name):

    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #if not file_exists:
            # Write headers if the file is new
            #headers = ['Subject'] + [f'FF {label}' for label in sorted(ffs.keys())]
            #writer.writerow(headers)
        # Write info in a row (1st element: subject name, then: all the labels)
        row = [subject_name] + [ffs[label] for label in sorted(ffs.keys())]
        writer.writerow(row)

def write_vol_ff_to_csv(output_csv_path, volumes, ffs, subject_name):

    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #if not file_exists:
            # Write headers if the file is new
            #headers = ['Subject'] + [f'FF {label}' for label in sorted(ffs.keys())]
            #writer.writerow(headers)
        # Write info in a row (1st element: subject name, then: all the labels)
        row = [subject_name] + [volumes[label] for label in sorted(volumes.keys())] + [ffs[label] for label in sorted(ffs.keys())]
        writer.writerow(row)

#BINARY SEGMENTATIONS:
# Function to load, binarize and save segmentations
#Also calculates the total volume of the binary segmentation
def process_and_save_segmentations(folder, output_folder):
    volume_results = {}
    # CCreate output directory
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(folder):
        if file.endswith(".mhd") and "_segmentation" in file:
            filepath = os.path.join(folder, file)
            key = file.replace("_segmentation.mhd", "")

            segmentation_image = sitk.ReadImage(filepath)
            array = sitk.GetArrayFromImage(segmentation_image)  # Array 3D: (Depth, Height, Width)

            # Binarize
            binary_array = (array > 0).astype("uint8")

            # Convert again to image
            binary_image = sitk.GetImageFromArray(binary_array)

            # Copy spatial information
            binary_image.CopyInformation(segmentation_image)

            # Save the binary image in the output directory
            output_path = os.path.join(output_folder, f"{key}_seg_binary.mhd")
            sitk.WriteImage(binary_image, output_path, True)

            print(f"Binary segmentation saved in: {output_path}")

            # Volume calculation:
            spacing = segmentation_image.GetSpacing()  # (z_spacing, y_spacing, x_spacing)

            voxel_volume = spacing[0] * spacing[1] * spacing[2]  # Volumen de un voxel

            num_segmented_voxels = np.sum(binary_array)

            total_volume = num_segmented_voxels * voxel_volume

            volume_results[key] = total_volume

            print(f"Total volume for {key}: {total_volume} mm^3")

    return volume_results

#APPLY MASK AND CALCULATE MEAN FF:
def apply_mask_and_calculate_ff(folder):
    ff_results = {}
    for file in os.listdir(folder):
        if file.endswith("_seg_binary.mhd"):
            # Load the binary mask
            mask_path = os.path.join(folder, file)
            mask_image = sitk.ReadImage(mask_path)
            mask_array = sitk.GetArrayFromImage(mask_image)  # Array 3D: (Depth, Height, Width)

            # Find the '_ff' image for every subject
            key = file.replace("_seg_binary.mhd", "")
            ff_file = f"{key}_ff.mhd"
            ff_path = os.path.join(folder, ff_file)

            if os.path.exists(ff_path):
                # Load the '_ff' image
                ff_image = sitk.ReadImage(ff_path)
                ff_array = sitk.GetArrayFromImage(ff_image)  # Array 3D: (Depth, Height, Width)

                # Check the dimensions
                if mask_array.shape != ff_array.shape:
                    print(f"Dimensiones no coinciden entre máscara y '_ff' para: {key}")
                    continue

                # Apply the mask
                masked_values = ff_array[mask_array > 0]

                # Calculate the mean ff
                mean_ff = np.mean(masked_values)
                ff_results[key] = mean_ff

                print(f"Mean FF for {key}: {mean_ff}")

    return ff_results

#SAVE TOTAL VOLUME AND FF ON THE SAME .CSV FILE
def save_results_to_csv(volume_results, ff_results, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        #writer.writerow(["Sujeto", "Volumen (mm^3)", "FF Promedio"])

        for key in volume_results:
            volume = volume_results.get(key, "N/A")
            ff = ff_results.get(key, "N/A")
            writer.writerow([key, volume, ff])

    print(f"Information saved in: {output_csv}")

def subject_csv(name, all_volumes, all_ffs, total_vol, mean_ff, subject_path):

    # Validar que la carpeta exista
    if not os.path.exists(subject_path):
        print(f"Error: La carpeta {subject_path} no existe.")
        return

    # Crear la ruta del archivo CSV
    csv_file = os.path.join(subject_path, f"{name}.csv")

    # Preparar los datos
    row = [name] + list(all_volumes.values()) + list(all_ffs.values()) + [total_vol, mean_ff]

    # Escribir el archivo CSV
    try:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
        print(f"Archivo CSV generado exitosamente en {subject_path}")
    except Exception as e:
        print(f"Error al generar el archivo CSV: {e}")

def all_subjects_csv(names, list_all_volumes, list_all_ffs, list_total_vol, list_mean_ff, general_path):
    # Validar que las listas tengan la misma longitud
    if not all(len(lista) == len(names) for lista in [list_all_volumes, list_all_ffs, list_total_vol, list_mean_ff]):
        print("Error: Todas las listas deben tener la misma longitud.")
        return

    # Determinar el máximo número de columnas requerido para los vectores
    max_volumenes = max(len(vol) for vol in list_all_volumes)
    max_ffs = max(len(ff) for ff in list_all_ffs)

    # Preparar las filas
    rows = []
    for i in range(len(names)):
        row = [names[i]]
        # Agregar volúmenes y rellenar con ceros si faltan valores
        row += list(list_all_volumes[i].values()) + [0] * (max_volumenes - len(list_all_volumes[i]))
        # Agregar FFs y rellenar con ceros si faltan valores
        row += list(list_all_ffs[i].values()) + [0] * (max_ffs - len(list_all_ffs[i]))
        # Agregar vol_total y ff_medio
        row += [list_total_vol[i], list_mean_ff[i]]
        rows.append(row)

    # Escribir el archivo CSV
    try:
        with open(general_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)  # Escribir las filas
        print(f"CSV succesfully created in {general_path}")
    except Exception as e:
        print(f"Failure to create the csv file: {e}")


def create_mri_segmentation_gif(mri, segmentation, output_path, colormap='tab10'):
    """
    Creates an animated GIF overlaying segmentation masks on MRI slices.

    Parameters:
        mri (numpy.ndarray): 3D array of MRI data.
        segmentation (numpy.ndarray): 3D array of segmentation labels (same size as mri).
        output_path (str): Path to save the output GIF.
        colormap (str): Matplotlib colormap for the segmentation labels.
    """
    # Check input dimensions
    assert mri.shape == segmentation.shape, "MRI and segmentation must have the same shape."

    # Normalize MRI for better visualization
    #mri_normalized = (mri - np.min(mri)) / (np.max(mri) - np.min(mri))
    mri_normalized = mri / np.max(mri)

    # Create a colormap
    cmap = plt.get_cmap(colormap)

    # Collect frames for the GIF
    frames = []
    for i in range(mri.shape[2]):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(mri_normalized[:, :, i], cmap='gray', interpolation='none')
        ax.imshow(segmentation[:, :, i], cmap=cmap, alpha=0.4, interpolation='none')
        ax.axis('off')

        # Save the frame to a temporary buffer
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    # Write frames to an animated GIF
    imageio.mimsave(output_path, frames, duration=0.1)  # duration = time per frame in seconds
    print(f"GIF saved to {output_path}")

def create_segmentation_overlay_animated_gif(sitkImage, sitkLabels, output_path):
    frames = []
    imageSize = sitkImage.GetSize()
    sitkImage = sitk.RescaleIntensity(sitkImage, 0, 1)

    for i in range(imageSize[2]):
        fig, ax = plt.subplots(figsize=(6, 6))
        contour_overlaid_image = sitk.LabelMapContourOverlay(
            sitk.Cast(sitkLabels[:, :, i], sitk.sitkLabelUInt8),
            sitkImage[:, :, i],
            opacity=1,
            contourThickness=[4, 4],
            dilationRadius=[3, 3]
        )

        arr = sitk.GetArrayFromImage(contour_overlaid_image)
        ax.imshow(arr, interpolation='none', origin='lower')
        ax.axis('off')

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(output_path, frames, duration=0.1)
    print(f"GIF saved to {output_path}")


# Función para cargar archivos .mhd
def load_segmentations_mhd(folder):
    segmentations = {}
    for root, _, files in os.walk(folder):  # ahora recorre TODO
        for file in files:
            # busca archivos de segmentación con la extensión que estés usando
            if file.endswith(extensionImages) and "_segmentation" in file:
                filepath = os.path.join(root, file)
                try:
                    img = sitk.ReadImage(filepath)
                    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
                    subject_name = os.path.basename(root)
                    key = f"{subject_name}_{os.path.splitext(os.path.splitext(file)[0])[0]}"
                    segmentations[key] = arr
                except Exception as e:
                    print(f"Error al leer {filepath}: {e}")
    print(f"Total de segmentaciones encontradas: {len(segmentations)}")
    return segmentations


# Crear GIFs
def create_gifs(segmentations, output_folder):
    for key, segmentation in segmentations.items():
        frames = []
        for i in range(segmentation.shape[0]):  # Recorrer el eje de profundidad (Depth)
            frame = segmentation[i]
            # Escalar los valores al rango [0, 255] para convertir en imagen
            normalized = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255
            img = Image.fromarray(normalized.astype('uint8'))
            frames.append(img)

        # Guardar como GIF
        output_path = os.path.join(output_folder, f"{key}_animation.gif")
        imageio.mimsave(output_path, frames, fps=5)  # Ajustar `fps` según la velocidad deseada
        print(f"GIF guardado en: {output_path}")

#FUNCIONES PARA IMAGENES DE TEJIDOS Y DE GRASA SUBCUTANEA:
#FUNCION BODY MASK DESDE INPHASE
# Function that creates a mask for the body from an in-phase dixon image. It uses an Otsu thresholding and morphological operations
# to create a mask where the background is 0 and the body is 1. Can be used for masking image registration.
def GetBodyMaskFromInPhaseDixon(inPhaseImage, vectorRadius = (2,2,2)):
    kernel = sitk.sitkBall
    otsuImage = sitk.OtsuMultipleThresholds(inPhaseImage, 4, 0, 128, # 4 classes and 128 bins
                                            False)  # 5 Classes, itk, doesn't coun't the background as a class, so we use 4 in the input parameters.
    # Open the mask to remove connected regions
    background = sitk.BinaryMorphologicalOpening(sitk.Equal(otsuImage, 0), vectorRadius, kernel)
    background = sitk.BinaryDilate(background, vectorRadius, kernel)
    bodyMask = sitk.Not(background)
    bodyMask.CopyInformation(inPhaseImage)
    # Fill holes:
    #bodyMask = sitk.BinaryFillhole(bodyMask, False)
    # Fill holes in 2D (to avoid holes coming from bottom and going up):
    bodyMask = BinaryFillHolePerSlice(bodyMask)
    return bodyMask


#FUNCION BODY MASK DESDE FAT
# Function that creates a mask for the body from an fat dixon image. It uses an Otsu thresholding and morphological operations
# to create a mask where the background is 0 and the body is 1. Can be used for masking image registration. Assumes that skin fat
# surround the patient body.
def GetBodyMaskFromFatDixonImage(fatImage, vectorRadius = (2,2,2), minObjectSizeInSkin = 500):
    kernel = sitk.sitkBall
    otsuImage = sitk.OtsuMultipleThresholds(fatImage, 1, 0, 128, # 1 classes and 128 bins
                                            False)  # 2 Classes, itk, doesn't coun't the background as a class, so we use 1 in the input parameters.
    # Open the mask to remove connected regions, mianly motion artefacts outside the body
    fatMask = sitk.BinaryMorphologicalOpening(sitk.Equal(otsuImage, 1), vectorRadius, kernel)
    # Remove small objects:
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOff()
    relabelComponentFilter = sitk.RelabelComponentImageFilter()
    relabelComponentFilter.SetMinimumObjectSize(minObjectSizeInSkin)
    sliceFatObjects = relabelComponentFilter.Execute(
        connectedFilter.Execute(fatMask))  # RelabelComponent sort its by size.
    fatMask = sliceFatObjects > 0  # Assumes that can be two large objetcts at most (for each leg)

    fatMask = sitk.BinaryDilate(fatMask, vectorRadius)
    bodyMask = fatMask
    # Go through all the slices:
    for j in range(0, fatMask.GetSize()[2]):
        sliceFat = fatMask[:, :, j]
        ndaSliceFatMask = sitk.GetArrayFromImage(sliceFat)
        ndaSliceFatMask = convex_hull_image(ndaSliceFatMask)
        sliceFatConvexHull = sitk.GetImageFromArray(ndaSliceFatMask.astype('uint8'))
        sliceFatConvexHull.CopyInformation(sliceFat)
        # Now paste the slice in the output:
        sliceBody = sitk.JoinSeries(sliceFatConvexHull)  # Needs to be a 3D image
        bodyMask = sitk.Paste(bodyMask, sliceBody, sliceBody.GetSize(), destinationIndex=[0, 0, j])
    bodyMask = sitk.BinaryDilate(bodyMask, vectorRadius)
    return bodyMask


#FUNCION CALCULO TEJIDO ADIPOSO SUBCUTANEO (Toma la imagen de la función anterior porque tiene que usar las etiquetas 3)
# gets the skin fat from a dixon segmented image, which consists of dixonSegmentedImage (0=air, 1=muscle, 2=muscle/fat,
# 3=fat)
def GetSkinFatFromTissueSegmentedImageUsingConvexHullPerSlice(dixonSegmentedImage, minObjectSizeInMuscle = 500, minObjectSizeInSkin = 500):
    # Inital skin image:
    skinFat = dixonSegmentedImage == 3
    # Body image:
    bodyMask = dixonSegmentedImage > 0
    # Create a mask for other tissue:
    notFatMask = sitk.And(bodyMask, (dixonSegmentedImage < 3))
    notFatMask = sitk.BinaryMorphologicalOpening(notFatMask, 3)
    #Filter to process the slices:
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOff()
    relabelComponentFilterMuscle = sitk.RelabelComponentImageFilter()
    relabelComponentFilterMuscle.SetMinimumObjectSize(minObjectSizeInMuscle)
    relabelComponentFilterSkin = sitk.RelabelComponentImageFilter()
    relabelComponentFilterSkin.SetMinimumObjectSize(minObjectSizeInSkin)
    # Go through all the slices:
    for j in range(0, skinFat.GetSize()[2]):
        sliceFat = skinFat[:, :, j]
        sliceNotFat = notFatMask[:, :, j]
        # Remove external objects:
        sliceFatEroded = sitk.BinaryMorphologicalOpening(sliceFat, 5)
        ndaSliceFatMask = sitk.GetArrayFromImage(sliceFatEroded)
        ndaSliceFatMask = convex_hull_image(ndaSliceFatMask)
        sliceFatConvexHull = sitk.GetImageFromArray(ndaSliceFatMask.astype('uint8'))
        sliceFatConvexHull.CopyInformation(sliceFat)
        #sliceNotFat = sitk.BinaryErode(sliceNotFat, 3)

        # Get the largest connected component:
        sliceNotFat = sitk.And(sliceNotFat, sliceFatConvexHull) # To remove fake object in the outer region of the body due to coil artefacts.
        sliceNotFatObjects = relabelComponentFilterMuscle.Execute(
            connectedFilter.Execute(sliceNotFat))  # RelabelComponent sort its by size.
        sliceNotFat = sliceNotFatObjects > 0 # sitk.And(sliceNotFatObjects > 0, sliceNotFatObjects < 3) # Assumes that can be two large objetcts at most (for each leg)
        # Dilate to return to the original size:
        sliceNotFat = sitk.BinaryDilate(sliceNotFat, 3)  # dilate to recover original size

        # Now apply the convex hull:
        ndaNotFatMask = sitk.GetArrayFromImage(sliceNotFat)
        ndaNotFatMask = convex_hull_image(ndaNotFatMask)
        sliceNotFat = sitk.GetImageFromArray(ndaNotFatMask.astype('uint8'))
        sliceNotFat.CopyInformation(sliceFat)
        sliceFat = sitk.And(sliceFat, sitk.Not(sliceNotFat))
        # Leave the objects larger than minSize for the skin fat:
        sliceFat = relabelComponentFilterSkin.Execute(
            connectedFilter.Execute(sliceFat))
        #sliceFat = sitk.Cast(sliceFat, sitk.sitkUInt8)
        sliceFat = sliceFat > 0
        # Now paste the slice in the output:
        sliceFat = sitk.JoinSeries(sliceFat)  # Needs to be a 3D image
        skinFat = sitk.Paste(skinFat, sliceFat, sliceFat.GetSize(), destinationIndex=[0, 0, j])
    skinFat = sitk.BinaryDilate(skinFat, 3)
    return skinFat


#FUNCION RELLENAR AGUJEROS POR CORTE
# Auxiliary function that fill hole in an image but per each slice:
def BinaryFillHolePerSlice(input):
    output = input
    for j in range(0, input.GetSize()[2]):
        slice = input[:,:,j]
        slice = sitk.BinaryFillhole(slice, False)
        # Now paste the slice in the output:
        slice = sitk.JoinSeries(slice) # Needs tobe a 3D image
        output = sitk.Paste(output, slice, slice.GetSize(), destinationIndex=[0, 0, j])
    return output


# FUNCION SEGMENTAR TODOS LOS TEJIDOS
# DixonTissueSegmentation received the four dixon images in the following order: in-phase, out-of-phase, water, fat.
# Returns a labelled image into 4 tissue types: air-background (0), soft-tissue (1), soft-tissue/fat (2), fat (3)
def DixonTissueSegmentation(dixonImages):
    labelAir = 0
    labelFat = 3
    labelSoftTissue = 1
    labelFatWater = 2
    labelBone = 4
    labelUnknown = 5

    # Threshold for background:
    backgroundThreshold = 80
    # Threshold for water fat ratio:
    waterFatThreshold = 2
    # Generate a new image:
    segmentedImage = sitk.Image(dixonImages[0].GetSize(), sitk.sitkUInt8)
    segmentedImage.SetSpacing(dixonImages[0].GetSpacing())
    segmentedImage.SetOrigin(dixonImages[0].GetOrigin())
    segmentedImage.SetDirection(dixonImages[0].GetDirection())

    # otsuOtuput = sitk.OtsuMultipleThresholds(dixonImages[0], 4, 0, 128, False)
    # voxelsAir = sitk.Equal(otsuOtuput, 0)
    # Faster and simpler version but will depend on intensities:
    voxelsAir = sitk.Less(dixonImages[0], backgroundThreshold)

    # Set air tags for lower values:
    # segmentedImage = sitk.Mask(segmentedImage, voxelsAir, labelUnknown, labelAir)
    ndaSegmented = sitk.GetArrayFromImage(segmentedImage)
    ndaInPhase = sitk.GetArrayFromImage(dixonImages[0])
    ndaSegmented.fill(labelUnknown)
    ndaSegmented[ndaInPhase < backgroundThreshold] = labelAir

    # Get arrays for the images:
    ndaInPhase = sitk.GetArrayFromImage(dixonImages[0])
    ndaOutOfPhase = sitk.GetArrayFromImage(dixonImages[1])
    ndaWater = sitk.GetArrayFromImage(dixonImages[2])
    ndaFat = sitk.GetArrayFromImage(dixonImages[3])

    # SoftTisue:
    WFratio = np.zeros(ndaWater.shape)
    WFratio[(ndaFat != 0)] = ndaWater[(ndaFat != 0)] / ndaFat[(ndaFat != 0)]
    # ndaSegmented[np.isnan(WFratio)] = labelUnknown
    ndaSegmented[np.logical_and(WFratio >= waterFatThreshold, (ndaSegmented == labelUnknown))] = labelSoftTissue
    # Also include when fat is zero and water is different to zero:
    ndaSegmented[np.logical_and((ndaWater != 0) & (ndaFat == 0), (ndaSegmented == labelUnknown))] = labelSoftTissue

    # For fat use the FW ratio:
    WFratio = np.zeros(ndaWater.shape)
    WFratio[(ndaWater != 0)] = ndaFat[(ndaWater != 0)] / ndaWater[(ndaWater != 0)]

    # Fat:
    ndaSegmented[np.logical_and(WFratio >= waterFatThreshold, ndaSegmented == labelUnknown)] = labelFat
    ndaSegmented[np.logical_and((ndaWater != 0) & (ndaFat == 0), (ndaSegmented == labelUnknown))] = labelFat

    # SoftTissue/Fat:
    ndaSegmented[np.logical_and(WFratio < waterFatThreshold, ndaSegmented == labelUnknown)] = labelFatWater

    # Set the array:
    segmentedImage = sitk.GetImageFromArray(ndaSegmented)
    segmentedImage.SetSpacing(dixonImages[0].GetSpacing())
    segmentedImage.SetOrigin(dixonImages[0].GetOrigin())
    segmentedImage.SetDirection(dixonImages[0].GetDirection())

    # The fat fraction image can have issues in the edge, for that reason we apply a body mask from the inphase image
    maskBody = GetBodyMaskFromInPhaseDixon(dixonImages[0], vectorRadius=(2, 2, 2))

    # Apply mask:
    maskFilter = sitk.MaskImageFilter()
    maskFilter.SetMaskingValue(1)
    maskFilter.SetOutsideValue(0)
    segmentedImage = maskFilter.Execute(segmentedImage, sitk.Not(maskBody))
    return segmentedImage

# Función para encontrar y cargar segmentaciones de tejidos
def load_tissue_segmentations(folder):
    segmentations = {}
    for root, _, files in os.walk(folder):  # Recorrer todas las subcarpetas
        print(f"Explorando carpeta: {root}")  # Para depuración
        for file in files:
            if file.endswith("_tissue_segmented.mhd"):  # Corrige el sufijo según los archivos
                filepath = os.path.join(root, file)
                print(f"Archivo encontrado: {filepath}")  # Para depuración
                try:
                    # Leer la imagen con SimpleITK
                    image = sitk.ReadImage(filepath)
                    array = sitk.GetArrayFromImage(image)  # Array 3D
                    # Usar la carpeta del sujeto como parte del identificador
                    subject_name = os.path.basename(root)
                    key = f"{subject_name}_{file.replace('_tissue_segmented.mhd', '')}"
                    key = key.lstrip('_')  # Elimina el guion bajo al principio si existe
                    segmentations[key] = array
                except Exception as e:
                    print(f"Error al leer {filepath}: {e}")
    print(f"Total de archivos encontrados: {len(segmentations)}")
    return segmentations

# Crear GIFs
def create_gifs(segmentations, output_folder):
    for key, segmentation in segmentations.items():
        frames = []
        for i in range(segmentation.shape[0]):  # Recorrer el eje de profundidad (Depth)
            frame = segmentation[i]
            # Escalar los valores al rango [0, 255] para convertir en imagen
            normalized = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255
            img = Image.fromarray(normalized.astype('uint8'))
            frames.append(img)

        # Guardar como GIF
        output_path = os.path.join(output_folder, f"{key}_animationTS.gif")
        imageio.mimsave(output_path, frames, fps=5)  # Ajustar `fps` según la velocidad deseada
        print(f"GIF guardado en: {output_path}")

# --------------------------- CONFIG PATHS  ---------------------------
input_root = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifty_output/'
outputPath = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/segmentations/' #PATH DE SALIDA (Donde se guardan los resultados)
output_pelvis_path = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifti_pelvis/'
output_lumbar_path = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifti_lumbar/'
#outputBiasCorrectedPath = outputPath + '/BiasFieldCorrection/'
os.makedirs(outputPath, exist_ok=True)
os.makedirs(output_pelvis_path, exist_ok=True)
os.makedirs(output_lumbar_path, exist_ok=True)
coord_csv = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/slices_cortes_anatomicos.csv'

# REFERENCE IMAGE FOR THE PRE PROCESSING REGISTRATION:
coords_df = pd.read_csv(coord_csv)

#referenceGluteusImageFilename = '/home/martin/data_imaging/Muscle/GlutealSegmentations/PelvisFOV/ManualSegmentations/MhdRegisteredDownsampled/ID00002.mhd'

# Modelos
lumbar_model_path  = "/home/german/lower-body-muscle-qMRI-pipeline/models/lumbarspine_unet3d_20230626_191618_173_best_fit.pt"
gluteal_model_path = "/home/german/lower-body-muscle-qMRI-pipeline/models/gluteal_unet3d_20250807_110716_123_best_fit.pt"

# Imágenes de referencia
lumbar_reference_path  = "/data/MuscleSegmentation/Data/LumbarSpine3D/ResampledData/C00001.mhd"
#lumbar_reference_path  = '/home/german/lower-body-muscle-qMRI-pipeline/data/reference_images/lumbar_spine_reference.nii.gz'
gluteus_reference_path = "/home/german/lower-body-muscle-qMRI-pipeline/data/reference_images/pelvis_reference.nii.gz"

# CONFIGURATION:
device_to_use = 'cuda' #'cpu'
# Needs registration
preRegistration = True #TRUE: Pre-register using the next image
dataInSubdirPerSubject = True
registrationReferenceFilename = '/data/MuscleSegmentation/Data/LumbarSpine3D/ResampledData/C00001.mhd'

imageNames = []
imageFilenames = []
i = 0
fat_fraction_all_subjects = list() #List to then write the .csv file
volume_all_subjects = list() #List to then write the .csv file
totalvolume_all_subjects = list()
meanff_all_subjects = list()
names_subjects = list()

#PATHS FOR THE BINARY SEGMENTATION: In that case input and output paths are the same
inputSeg = outputPath # Where the original segmentations are saved
outputBinSeg = outputPath # Where will be the binary segmentations
binarySegAndFFPath = outputPath #Folder that contains the binary segmentation and 'ff' images

# REGISTRATION PARAMETER FILES:
similarityMetricForReg = 'NMI' #NMI Metric
parameterFilesPath = '/data/MuscleSegmentation/Data/Elastix/' #Parameters path
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg

#IMAGE FORMATS AND EXTENSION:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.mhd' #'.nii.gz'
tagInPhase = '_I'#
# outOfPhaseSuffix
waterSuffix = '_W'
fatSuffix = '_F'
tagAutLabels = '_aut'
tagManLabels = '_labels'

# OUTPUT PATHS:
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

vol_csv_name = 'volumes.csv'
ff_csv_name = 'ffs.csv'
all_csv_name = 'all_volumes_and_ffs.csv'
vol_and_ff_name = 'volumes_and_ffs.csv'
totalvol_and_meanff_name = 'TotalVol_MeanFF.csv'

# ---- nombres globales  ----
vol_lumbar_all      = 'volumes_lumbar_all.csv'
ff_lumbar_all       = 'ffs_lumbar_all.csv'
volff_lumbar_all    = 'volumes_and_ffs_lumbar.csv'

vol_pelvis_all      = 'volumes_pelvis_all.csv'
ff_pelvis_all       = 'ffs_pelvis_all.csv'
volff_pelvis_all    = 'volumes_and_ffs_pelvis.csv'

#Clear the .csv files if they've got information
file_path_v = os.path.join(outputPath, vol_csv_name)
if os.path.exists(file_path_v):
    with open(file_path_v, 'w') as file:
        pass  # Clear

for fname in [vol_lumbar_all, ff_lumbar_all, volff_lumbar_all,
              vol_pelvis_all, ff_pelvis_all, volff_pelvis_all]:
    fpath = os.path.join(outputPath, fname)
    if os.path.exists(fpath):
        open(fpath, 'w').close()

file_path_ff = os.path.join(outputPath, ff_csv_name)
if os.path.exists(file_path_ff):
    with open(file_path_ff, 'w') as file:
        pass  # Clear

file_path_vol_ff = os.path.join(outputPath, vol_and_ff_name)
if os.path.exists(file_path_vol_ff):
    with open(file_path_vol_ff, 'w') as file:
        pass  # Clear

general_path = os.path.join(outputPath, all_csv_name)
if os.path.exists(general_path):
    with open(general_path, 'w') as file:
        pass  # Clear

# CSV THAT CONTAINS TOTAL VOLUME AND MEAN FF PATH
TotalVol_MeanFF_csv = os.path.join(outputPath, totalvol_and_meanff_name)
if os.path.exists(TotalVol_MeanFF_csv):
    with open(TotalVol_MeanFF_csv, 'w') as file:
        pass  # Clear

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


# MODEL INIT:
multilabelNum = 8
torch.cuda.empty_cache()
model = Unet(1, multilabelNum)
model.load_state_dict(torch.load(lumbar_model_path, map_location=device))
lumbarModel = model.to(device)

#READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS:
# Parameters for image registration:
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.ElastixImageFilter().ReadParameterFile(parameterFilesPath + paramFileRigid + '.txt'))
print('Reference image voxel size: {0}'.format(referenceImage_lumbar.GetSize()))

# MODEL INIT:
multilabelNum = 8
torch.cuda.empty_cache()
model = Unet(1, multilabelNum)
model.load_state_dict(torch.load(gluteal_model_path, map_location=device))
glutealModel = model.to(device)

#READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS:
# Parameters for image registration:
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.ElastixImageFilter().ReadParameterFile(parameterFilesPath + paramFileRigid + '.txt'))
print('Reference image voxel size: {0}'.format(referenceImage_gluteus.GetSize()))


# --------------------------- PROCESS EACH VOLUNTEER ---------------------------

#for idx, row in coords_df.iterrows():
for idx, row in coords_df.iloc[0:5].iterrows():
    inPhaseImageLumbar, fatImageLumbar, waterImageLumbar = None, None, None
    inPhaseImagePelvis, fatImagePelvis, waterImagePelvis = None, None, None
    ffLumbar, ffPelvis = None, None
    volunteer_id = row['ID']
    subject = volunteer_id
    outputPathThisSubject = os.path.join(outputPath, volunteer_id ) #antes era outputroot
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
        if dixon_tag == 'in':
            inPhaseImagePelvis = sitk_pelvis_image
        elif dixon_tag == 'f':
            fatImagePelvis = sitk_pelvis_image
        elif dixon_tag == 'w':
            waterImagePelvis = sitk_pelvis_image
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

    # Output path for this subject:
    outputPathThisSubject = os.path.join(outputPath, subject) #Generate the path of that subject to save his/her info
    if not os.path.exists(outputPathThisSubject):
        os.makedirs(outputPathThisSubject)

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

    # PELVIS (GLÚTEO)
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

    # Number of labels of the segmentation
    num_labels = multilabelNum

    # Voxel volume
    voxel_volume = np.prod(spacing) #volume (X.Y.Z)

        # Volume dictionary to save the label volumes
    volumes = {}

    # Iterate over labels
    for label in range(1, num_labels+1):
        # Creates a mask for the actual label (1 or 0)
        label_mask = (segmentation_array == label).astype(np.uint8)

        # Counts the number of voxels
        label_voxels = np.sum(label_mask)

        # Calculates the volume of that class (quantity of voxels * individual voxel volume)
        label_volume = label_voxels * voxel_volume

        # Save the data on the dictionary
        volumes[label] = label_volume

    #Add them to the list
    volume_all_subjects.append(volumes)

    #Write on the .csv
    write_volumes_to_csv(os.path.join(outputPathThisSubject, 'volumes_lumbar.csv'), volumes, subject)
    write_volumes_to_csv(os.path.join(outputPath, vol_lumbar_all), volumes, subject)

    # Print the volume of all the labels:
    print("\nVolumes:")
    for label, volume in volumes.items():
        print(f"Muscle {label}: {volume} mm³")

    # WRITE AN ANIMATED GIF WITH THE SEGMENTATION
    create_segmentation_overlay_animated_gif(sitkImage, output, os.path.join(outputPathThisSubject, f"{subject}_lumbar_segmentation_check.gif"))

    #FF CALCULATION:

    # Images to numpy arrays
    fatfraction_array = sitk.GetArrayFromImage(ffLumbar)

    #FF .CSV PATH:
    output_csv_path = os.path.join(outputPathThisSubject, 'ffs.csv')

    # Mean FF for every single label
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
    write_ff_to_csv(os.path.join(outputPathThisSubject, 'ffs_lumbar.csv'), fat_fraction_means, subject)
    write_ff_to_csv(os.path.join(outputPath, ff_lumbar_all), fat_fraction_means, subject)

    # Print the results
    print("\nFFs:")
    #for label in range(0,9):
    for label, fat_mean in fat_fraction_means.items():
        if fat_mean is not None:
            print(f"Muscle {label}: {fat_mean:.4f}")
        else:
            print(f"Muscle {label}: Sin valores válidos")

    ############PELVIS!!! #######################################################

    ############ PELVIS (GLÚTEO) ############
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

    write_volumes_to_csv(os.path.join(outputPathThisSubject, 'volumes_pelvis.csv'), volumes_pelvis, subject)
    write_volumes_to_csv(os.path.join(outputPath, vol_pelvis_all), volumes_pelvis, subject)
    # FF pelvis:
    write_ff_to_csv(os.path.join(outputPathThisSubject, 'ffs_pelvis.csv'), fat_fraction_means_pelvis, subject)
    write_ff_to_csv(os.path.join(outputPath, ff_pelvis_all), fat_fraction_means_pelvis, subject)

    # combinado pelvis:
    write_vol_ff_to_csv(os.path.join(outputPathThisSubject, 'volumes_and_ffs_pelvis.csv'),
                        volumes_pelvis, fat_fraction_means_pelvis, subject)
    write_vol_ff_to_csv(os.path.join(outputPath, volff_pelvis_all),
                        volumes_pelvis, fat_fraction_means_pelvis, subject)

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
    if 'ffPelvis' in locals():
        fatfraction_array_pelvis = sitk.GetArrayFromImage(ffPelvis)
        fat_fraction_means_pelvis = {}
        for label in range(1, multilabelNum + 1):
            mask_l = (segmentation_array_pelvis == label)
            fat_fraction_means_pelvis[label] = float(np.mean(fatfraction_array_pelvis[mask_l])) if np.any(
                mask_l) else None

        write_ff_to_csv(os.path.join(outputPathThisSubject, 'ffs_pelvis.csv'), fat_fraction_means_pelvis, subject)
        write_vol_ff_to_csv(os.path.join(outputPathThisSubject, 'volumes_and_ffs_pelvis.csv'),
                            volumes_pelvis, fat_fraction_means_pelvis, subject)
    else:
        print("[WARN] No se encontró ffPelvis para calcular FF por etiqueta en pelvis.")

    #SAVE THE VOLUME AND FF VALUES FOR EVERY SUBJECT ON THE SAME .CSV FILE:

    # VOL & FF .CSV PATH:
    output_csv_path = os.path.join(outputPathThisSubject, 'volumes_and_ffs.csv')

    #CSV FOR ALL THE VOLUME AND FF LABELS:
    write_vol_ff_to_csv(os.path.join(outputPathThisSubject, 'volumes_and_ffs_lumbar.csv'),
                        volumes, fat_fraction_means, subject)
    write_vol_ff_to_csv(os.path.join(outputPath, volff_lumbar_all),
                        volumes, fat_fraction_means, subject)

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
    names_subjects.append(subject)
    subject_csv(subject,volumes,fat_fraction_means,total_volume,mean_ff,outputPathThisSubject)

all_subjects_csv(names_subjects, volume_all_subjects, fat_fraction_all_subjects, totalvolume_all_subjects, meanff_all_subjects, general_path)

#BINARY SEGMENTATIONS:
#Run the methods that calculate the binary segmentation, calculate the total volume and the mean FF
volume_results = process_and_save_segmentations(inputSeg, outputBinSeg)
ff_results = apply_mask_and_calculate_ff(binarySegAndFFPath)

# Save the results
save_results_to_csv(volume_results, ff_results, TotalVol_MeanFF_csv)

