import SimpleITK as sitk
import SimpleITK as sitk, torch, imageio
import numpy as np
import multiprocessing


def apply_bias_correction_2(image: np.ndarray, shrink_factor=4) -> np.ndarray:
    """
    Applies an N4 bias correction filter to an image in a numpy array.

    Parameters:
        image (np.ndarray): Input image as a numpy array.

    Returns:
        np.ndarray: Bias-corrected image.
    """
    
    # Convert the numpy array to a SimpleITK image
    sitk_image = sitk.GetImageFromArray(image)
    sitk_image_lowres = sitk_image #In cas shrinking is appleid
    # ---- (optional but recommended) build a mask ----
    mask = sitk.OtsuThreshold(sitk_image, 0, 1, 200)

    # Shrink the image:
    if shrink_factor > 1:
        sitk_image_lowres = sitk.Shrink(sitk_image, [shrink_factor] * sitk_image.GetDimension())
        mask = sitk.Shrink(
            mask, [shrink_factor] * mask.GetDimension()
        )  

    # Initialize the N4 bias field correction filter
    n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
    n4_filter.SetNumberOfThreads(multiprocessing.cpu_count())
    # Apply the filter
    corrected_image = n4_filter.Execute(sitk_image_lowres, mask)

    log_bias_field = n4_filter.GetLogBiasFieldAsImage(sitk_image)

    corrected_image_full_resolution = sitk.Cast(sitk_image, sitk.sitkFloat32) / sitk.Exp(log_bias_field)

    # Convert the corrected image back to a numpy array
    corrected_array = sitk.GetArrayFromImage(corrected_image_full_resolution)

    return corrected_array

# BIAS FIELD CORRECTION
def apply_bias_correction(inputImage, shrinkFactor = (1,1,1)):
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
    
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
#from MultiAtlasSegmenter.MultiAtlasSegmentation.EvaluateSegmentation import hausdorff_distance_filter


def imshow_from_torch(img, imin=0, imax=1, ialpha = 1, icmap='gray'):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),vmin = imin, vmax = imax, cmap=icmap, alpha = ialpha)


def swap_labels(img, label1=0, label2=1):
    #img = img / 2 + 0.5     #unnormalize
    mask1 = img == label1
    mask1not = img != label1
    mask2 = img == label2
    mask2not = img != label2
    img = (img * mask1not) + mask1 * label2
    img = (img * mask2not) + mask2 * label1
    return img


def flip_image(image, axis, spacing):
    image = sitk.GetArrayFromImage(image)
    image = np.flip(image, axis)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def create_csv(vector, outpath):
    data = []
    for i in range(len(vector)):
        data.append([i, vector[i]])
    if "Epoch" in outpath:
        header = ['Epoch']
    else:
        header = ['Iteration']
    if "Dice" in outpath:
        header.append("Dice")
    else:
        header.append("Score")
    with open(outpath, 'w', newline='', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
        file.close()


def dice2d(reference, segmented):
    if reference.shape != segmented.shape:
        print('Error: shape')
        return 0
    reference = (reference > 0) * 1
    segmented = (segmented > 0) * 1
    tp = reference * segmented
    if tp.max() != 0:
        score = (2 * tp.sum())/(reference.sum() + segmented.sum())
    else:
        score = 0
    return score


def dice(reference, segmented):
    if reference.shape != segmented.shape:
        print('Error: shape')
        return 0
    reference = reference > 0
    segmented = segmented > 0
    tp = (reference * segmented) * 1
    fn = (~segmented * reference) * 1
    fp = (~reference * segmented) * 1
    score = (2 * tp.sum())/(2 * tp.sum() + fn.sum() + fp.sum())
    if tp.sum() == 0:
        score = 0
    return score


def sensitivity(label, segmented):
    if label.shape != segmented.shape:
        print('Error: shape')
        return 0
    label = label > 0
    segmented = segmented > 0
    tp = (label * segmented) * 1
    fn = (~segmented * label) * 1
    score = tp.sum()/(tp.sum() + fn.sum())
    if tp.sum() == 0:
        score = 0
    return score


def precision(label, segmented):
    if label.shape != segmented.shape:
        print('Error: shape')
        return 0
    label = label > 0
    segmented = segmented > 0
    tp = (label * segmented) * 1
    fp = (segmented * ~label) * 1
    score = tp.sum()/(tp.sum() + fp.sum())
    if tp.sum() == 0:
        score = 0
    return score

def specificity(label, segmented):
    if label.shape != segmented.shape:
        print('Error: shape')
        return 0
    label = label > 0
    segmented = segmented > 0
    tn = (~label * ~segmented) * 1
    fp = (segmented * ~label) * 1
    score = tn.sum() / (tn.sum() + fp.sum())
    return score

def hausdorff_distance(A, B):
    def point_to_set_distance(point, set_points):
        return np.min(np.linalg.norm(set_points - point, axis=1))

    def set_to_set_distance(set1, set2):
        return np.max([point_to_set_distance(point, set2) for point in set1])

    return max(set_to_set_distance(A, B), set_to_set_distance(B, A))

def hausdorff(label, segmented):
    if label.shape != segmented.shape:
        print('Error: shape')
        return 0
    label = sitk.Cast(sitk.GetImageFromArray(label),sitk.sitkFloat32)
    label_edge = sitk.GetArrayFromImage(sitk.SobelEdgeDetection(label))
    segmented = sitk.Cast(sitk.GetImageFromArray(segmented),sitk.sitkFloat32)
    segmented_edge = sitk.GetArrayFromImage(sitk.SobelEdgeDetection(segmented))
    distance = hausdorff_distance(segmented_edge,label_edge)
    return distance

def haus_distance(label,segmented):
    if label.shape != segmented.shape:
        print('Error: shape')
        return 0
    label = sitk.Cast(sitk.GetImageFromArray(label),sitk.sitkFloat32)
    segmented = sitk.Cast(sitk.GetImageFromArray(segmented),sitk.sitkFloat32)
    haus_filter = sitk.HausdorffDistanceImageFilter()
    haus_filter.Execute(label, segmented)
    return haus_filter.GetHausdorffDistance()

def rvd(label,segmented):
    if label.shape != segmented.shape:
        print('Error: shape')
        return 0
    label = (label > 0) * 1
    segmented = (segmented > 0) * 1
    resta= np.abs(label-segmented)
    score = resta.sum()/label.sum()
    return score


def maxProb(image, numlabels):
    outImage = np.zeros(image.shape)
    indexImage = np.argmax(image, axis=1)
    for k in range(numlabels):
        outImage[:, k, :, :] = image[:, k, :, :] * (indexImage == k)
    return outImage


def multilabel(image,Background = False):
    numLabels = image.shape[1]
    shape = image.shape
    shape = list(shape)
    shape.remove(numLabels)
    outImage = np.zeros(shape)
    for k in range(numLabels):
        if Background:
            outImage = outImage + image[:, k, :, :] * k
        else:
            outImage = outImage + image[:, k, :, :] * (k + 1)
    return outImage

def maskSize(image):
    numLabels = np.max(image).astype(np.uint8)
    outArray = []
    for k in range(numLabels):
        auximage = (image == k+1)
        outArray.append(np.sum(auximage))
    outArray = np.array(outArray, dtype=np.float64)
    return outArray


def labelfilter(image):
    filteredimage = sitk.GetImageFromArray(image)   # imagen binaria
    cc = sitk.ConnectedComponent(filteredimage)     # detecta cada uno de los connected components y le asigna un valor
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    largest_label = max(stats.GetLabels(), key=lambda x: stats.GetPhysicalSize(x)) # busco el valor del cc mas grande
    filteredimage = sitk.BinaryThreshold(cc, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1,
                                      outsideValue=0) # filtro unicamente el cc mas grande. Descarto sobresegmentaciones
    filteredimage = sitk.Not(filteredimage) # ~ mi imagen filtrada. El fondo y los huecos dentro de mi cc serán 1
    cc = sitk.ConnectedComponent(filteredimage)
    stats.Execute(cc)
    largest_label = max(stats.GetLabels(), key=lambda x: stats.GetPhysicalSize(x)) #busco el valor del label del fondo
    filtered_image = sitk.BinaryThreshold(cc, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1,
                                      outsideValue=0) #filtro unicamente el fondo eliminando los huecos de mi CC
    filteredimage = sitk.GetArrayFromImage(sitk.Not(filtered_image))
    return filteredimage

def FilterUnconnectedRegions(image, numLabels, ref , radiusErodeDilate=0):
    segmentedImage = sitk.GetImageFromArray(image)
    segmentedImage.CopyInformation(ref)
    # Output image:
    outSegmentedImage = sitk.Image(segmentedImage.GetSize(), sitk.sitkUInt8)
    outSegmentedImage.CopyInformation(ref)
    # Go through each label:
    relabelFilter = sitk.RelabelComponentImageFilter() # I use the relabel filter to get largest region for each label.
    relabelFilter.SortByObjectSizeOn()
    maskFilter = sitk.MaskImageFilter()
    maskFilter.SetMaskingValue(1)
    for i in range(0, numLabels):
        maskThisLabel = segmentedImage == (i+1)
        # Erode the mask:
        maskThisLabel = sitk.BinaryErode(maskThisLabel, radiusErodeDilate)
        # Now resegment to get labels for each segmented object:
        maskThisLabel = sitk.ConnectedComponent(maskThisLabel)
        # Relabel by object size:
        maskThisLabel = relabelFilter.Execute(maskThisLabel)
        # get the largest object:
        maskThisLabel = (maskThisLabel==1)
        # dilate the mask:
        maskThisLabel = sitk.BinaryDilate(maskThisLabel, radiusErodeDilate)
        # Assign to the output:
        maskFilter.SetOutsideValue(i+1)
        outSegmentedImage = maskFilter.Execute(outSegmentedImage, maskThisLabel)

    return outSegmentedImage


def filtered_multilabel(image, Background = False): #usar cuando se segmentan imagenes nuevas
    numLabels = image.shape[1]
    shape = image.shape
    shape = list(shape)
    shape.remove(numLabels)
    outImage = np.zeros(shape)
    for k in range(numLabels):
        filteredImage = labelfilter(image[:, k, :, :])
        if Background:
            outImage = outImage + filteredImage * k
        else:
            outImage = outImage + filteredImage * (k + 1)
    return outImage


def writeMhd(image, outpath, ref=0):
    img = sitk.GetImageFromArray(image)
    if ref != 0:
        img.CopyInformation(ref)
    sitk.WriteImage(img, outpath)



def pn_weights(trainingset, numlabels, background):         # positive to negative ratio
    weights = np.zeros(numlabels)
    for k in range(numlabels):
        if background:
            weights[k] = (trainingset.size - np.sum(trainingset == k))/np.sum(trainingset == k)
        else:
            weights[k] = (trainingset.size - np.sum(trainingset == (k+1))) / np.sum(trainingset == (k+1))
    weights = weights/np.sum(weights)
    weights.resize((1, weights.size, 1, 1))
    return torch.tensor(weights)


def rel_weights(trainingset, numlabels, background):        #positive to size ratio
    weights = np.zeros(numlabels)
    for k in range(numlabels):
        if background:
            weights[k] = trainingset.size / np.sum(trainingset == k)
        else:
            weights[k] = trainingset.size/np.sum(trainingset == (k+1))
    weights = weights / np.sum(weights)
    weights.resize((1, weights.size, 1, 1))
    return torch.tensor(weights)


def boxplot(data, xlabel, outpath, yscale, title):
    plt.figure()
    plt.boxplot(data, labels=xlabel)
    plt.title(title)
    plt.ylim(yscale)
    plt.savefig(outpath)
    plt.close()

# PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

def csv_creator(metric,subjects, colNames, datapath):
    datos = [subjects] + metric
    dictionary = dict(zip(colNames, datos))
    df = pd.DataFrame(dictionary)
    df.to_csv(datapath, index=False)
    

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

        # --- evitar advertencia de matplotlib ---
        if arr.dtype == np.uint8:
            arr = np.clip(arr, 0, 255)
            ax.imshow(arr, interpolation='none', origin='lower', vmin=0, vmax=255)
        else:
            arr = np.clip(arr, 0, 1)
            ax.imshow(arr, interpolation='none', origin='lower', vmin=0, vmax=1)
        # -----------------------------------------

        ax.axis('off')

        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
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


#FUNCIONES PARA IMAGENES DE TEJIDOS Y DE GRASA SUBCUTANEA:
#FUNCION BODY MASK DESDE INPHASE
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

def write_vol_ff_simple_csv(output_csv_path, volumes_lumbar, ffs_lumbar,
                            volumes_pelvis, ffs_pelvis,
                            skinfat_total=None, skinfat_pelvis=None,
                            subject_name="NA",
                            volumes_short=None, ffs_short=None):

    import os
    import pandas as pd

    new_row = {}
    new_row["Subject"] = subject_name

    # ----- Volúmenes -----
    for i in range(1, 9):
        new_row[f"Vol Lumbar_{i}"] = volumes_lumbar.get(i, "")
        new_row[f"Vol Pelvis_{i}"] = volumes_pelvis.get(i, "")

    new_row["Vol skinFat_total"] = skinfat_total if skinfat_total is not None else ""
    new_row["Vol skinFat_pelvis"] = skinfat_pelvis if skinfat_pelvis is not None else ""

    for i in range(1, 9):
        new_row[f"Vol ShortFOV_{i}"] = volumes_short.get(i, "") if volumes_short else ""

    # ----- FF -----
    for i in range(1, 9):
        new_row[f"FF Lumbar_{i}"] = ffs_lumbar.get(i, "")
        new_row[f"FF Pelvis_{i}"] = ffs_pelvis.get(i, "")

    for i in range(1, 9):
        new_row[f"FF ShortFOV_{i}"] = ffs_short.get(i, "") if ffs_short else ""

    new_row_df = pd.DataFrame([new_row])

    # Si no existe → crear
    if not os.path.exists(output_csv_path):
        new_row_df.to_csv(output_csv_path, index=False)
        return

    df = pd.read_csv(output_csv_path)

    # Agregar columnas faltantes al CSV
    for col in new_row_df.columns:
        if col not in df.columns:
            df[col] = ""

    # Asegurar que la nueva fila tenga todas las columnas
    for col in df.columns:
        if col not in new_row_df.columns:
            new_row_df[col] = ""

    new_row_df = new_row_df[df.columns]

    # 🔹 ACTUALIZAR SIN BORRAR OTRAS COLUMNAS
    if subject_name in df["Subject"].values:
        idx = df.index[df["Subject"] == subject_name][0]
        for col in new_row_df.columns:
            df.at[idx, col] = new_row_df.iloc[0][col]
    else:
        df = pd.concat([df, new_row_df], ignore_index=True)

    df.to_csv(output_csv_path, index=False)
