import os
import re
import nibabel as nib
import numpy as np
#import dcm2niix
import pydicom
from pathlib import Path
import subprocess, shutil, tempfile
import dicom2nifti
from nibabel.orientations import aff2axcodes
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
from utils import apply_bias_correction_2
import SimpleITK as sitk
# Configuration
convert_dicom = False  # Set to False if you want to skip DICOM conversion
concatenate_niftis = True  # Set to False if you want to skip concatenation
calculate_fat_fraction = False  # Set to False if you want to skip FF calculation
rewrite_converted = False  # If True, will re-convert DICOMs even if output exists
rewrite_concatenated = True  # If True, will re-concatenate even if output exists
# Datapath
#dicomDataPath = "/home/martin/data_imaging/Muscle/data_sarcopenia_tx/dicom/"
#niftiOtuputPath = "/home/martin/data_imaging/Muscle/data_sarcopenia_tx/nifti_output/"
dicomDataPath = "/home/martin/data_imaging/Muscle/data_sarcopenia_tx/dicom/"
niftiOtuputPath = "/home/martin/data_imaging/Muscle/data_sarcopenia_tx/nifti_output/"

if not os.path.exists(niftiOtuputPath):
    os.makedirs(niftiOtuputPath, exist_ok=True)

print("Script started")

def cargar_slices_dicom(dicom_folder):
    print("def cargar_slices_dicom(dicom_folder)")
    slices = []
    for filename in sorted(os.listdir(dicom_folder)):
        filepath = os.path.join(dicom_folder, filename)
        try:
            dcm = pydicom.dcmread(filepath)
            slices.append(dcm)
        except:
            continue
    return slices

def convertir_a_nifti(slices, output_file):
    print("def convertir_a_nifti(slices, output_file):")
    # Sort by Z position
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Build 3D volume
    image = np.stack([s.pixel_array for s in slices], axis=-1)
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    image = image * slope + intercept

    # Orientation and scale matrix
    pixel_spacing = slices[0].PixelSpacing
    slice_thickness = float(slices[0].SliceThickness)
    affine = np.diag([pixel_spacing[0], pixel_spacing[1], slice_thickness, 1])

    nifti_img = nib.Nifti1Image(image, affine)
    nib.save(nifti_img, output_file)
    print(f"🧠 NIfTI saved to: {output_file}")

def convertir_todos_los_voluntarios(dicom_base_folder, carpeta_salida_base):
    print("convertir_todos_los_voluntarios")
    for nombre_voluntario in sorted(os.listdir(dicom_base_folder)):
        print(f"⚠️ {nombre_voluntario}")
        ruta_voluntario = os.path.join(dicom_base_folder, nombre_voluntario)
        if not os.path.isdir(ruta_voluntario):
            continue

        carpetas_dentro = os.listdir(ruta_voluntario)
        if len(carpetas_dentro) != 1:
            print(f"⚠️ {nombre_voluntario} skipped: no single inner folder found.")
            continue

        carpeta_dicom = os.path.join(ruta_voluntario, carpetas_dentro[0])
        codigo_simple = nombre_voluntario.split("_")[0]
        carpeta_salida = os.path.join(carpeta_salida_base, codigo_simple)

        # Create output dir and convert only if it doesn't exist, or if rewrite_converted is True
        if not os.path.exists(carpeta_salida): 
            os.makedirs(carpeta_salida, exist_ok=True)
            dicom2nifti.convert_directory(carpeta_dicom, carpeta_salida,
                          compression=True, reorient=True)
        else:
            if rewrite_converted:
                print(f"Rewriting existing output for {codigo_simple}")
                dicom2nifti.convert_directory(carpeta_dicom, carpeta_salida,
                                compression=True, reorient=True)
            else:
                print(f"Skipping conversion for {codigo_simple} (output exists)")


def concatenar_niftis_en_grupos(carpeta_salida_base, apply_bias_correction=False, voluntarios_seleccionados=None):

    print("concatenar_niftis_en_grupos")

    for nombre_voluntario in sorted(os.listdir(carpeta_salida_base)):
        ruta_voluntario = os.path.join(carpeta_salida_base, nombre_voluntario)
        if not os.path.isdir(ruta_voluntario):
            continue

        # 🔹 Nuevo: si hay lista y no está este voluntario → saltearlo
        if voluntarios_seleccionados and nombre_voluntario not in voluntarios_seleccionados:
            continue

        archivos_nii = [f for f in os.listdir(ruta_voluntario) if f.endswith(".nii.gz")]

        grupos = {}
        archivos_unicos = []

        pattern = re.compile(r"^(\d+)(?!.*_nd).*_(f|in|opp|w)\.nii\.gz$")

        for archivo in archivos_nii:
            m = pattern.match(archivo)
            if not m:
                archivos_unicos.append(archivo)
                continue
            grupo = m.group(2)
            numero = int(m.group(1))
            key = f"{nombre_voluntario}_{grupo}"
            grupos.setdefault(key, []).append((archivo, numero))

        for nombre_grupo, lista_archivos in grupos.items():
            output_file = os.path.join(ruta_voluntario, f"{nombre_grupo}_dixon_concatenated.nii.gz")
            if os.path.exists(output_file) and not rewrite_concatenated:
                print(f"Skipping concatenation for {nombre_grupo} (output exists)")
                continue
            if len(lista_archivos) < 2:
                archivos_unicos.extend([x[0] for x in lista_archivos])
                continue

            lista_archivos_tra = [x for x in lista_archivos if "tra" in x[0]]
            if len(lista_archivos_tra) < 2:
                archivos_unicos.extend([x[0] for x in lista_archivos_tra])
                continue

            lista_archivos_tra.sort(key=lambda x: x[1])
            archivos_ordenados = [os.path.join(ruta_voluntario, x[0]) for x in lista_archivos_tra]

            print(f"\n📁 Group {nombre_grupo} → the following files will be concatenated:")
            for ruta in archivos_ordenados:
                print(f"   - {os.path.basename(ruta)}")

            imgs = [nib.load(archivo) for archivo in archivos_ordenados]
            datos = [img.get_fdata() for img in imgs]
            for i, data in enumerate(datos):
                mean_value = np.mean(data)
                max_value = np.max(data)
                print(f"   - Max value of {os.path.basename(archivos_ordenados[i])}: {max_value:.4f}")
                print(f"   - Mean value of {os.path.basename(archivos_ordenados[i])}: {mean_value:.4f}")

            if apply_bias_correction:
                datos = [apply_bias_correction_2(data, (3, 8, 8)) for data in datos]

            afines = [img.affine for img in imgs]
            datos = datos[::-1]

            concatenado = np.concatenate(datos, axis=2)
            print(f"Concatenated shape: {concatenado.shape}")

            nifti_concat = nib.Nifti1Image(concatenado, afines[-1])            
            nib.save(nifti_concat, output_file)
            print(f"✅ Concatenated saved: {output_file}")

        if archivos_unicos:
            print(f"🟨 Unique (not concatenated) files in {nombre_voluntario}:")
            for archivo in archivos_unicos:
                print(f"  {archivo}")


def calcular_fat_fraction_voluntarios(carpeta_salida_base, extensionImages=".nii.gz"):

    print("\n=== Calculating Fat Fraction (FF) for each volunteer ===")

    for nombre_voluntario in sorted(os.listdir(carpeta_salida_base)):
        ruta_voluntario = os.path.join(carpeta_salida_base, nombre_voluntario)
        if not os.path.isdir(ruta_voluntario):
            continue

        fat_file = os.path.join(ruta_voluntario, f"{nombre_voluntario}_f_dixon_concatenated{extensionImages}")
        water_file = os.path.join(ruta_voluntario, f"{nombre_voluntario}_w_dixon_concatenated{extensionImages}")

        if os.path.exists(fat_file) and os.path.exists(water_file):
            print(f"\n🧠 Calculating FF for {nombre_voluntario}")

            fatImage = sitk.Cast(sitk.ReadImage(fat_file), sitk.sitkFloat32)
            waterImage = sitk.Cast(sitk.ReadImage(water_file), sitk.sitkFloat32)

            # Calculate FF and apply mask
            waterfatImage = sitk.Add(fatImage, waterImage)
            fatfractionImage = sitk.Divide(fatImage, waterfatImage)
            fatfractionImage = sitk.Cast(
                sitk.Mask(fatfractionImage, waterfatImage > 0, outsideValue=0, maskingValue=0),
                sitk.sitkFloat32)

            # Save FF image
            output_filename = os.path.join(ruta_voluntario, f"{nombre_voluntario}_ff{extensionImages}")
            sitk.WriteImage(fatfractionImage, output_filename, True)
            print(f"✅ FF Image saved: {output_filename}")

        else:
            print(f"⚠️ Missing F and/or W concatenated images for {nombre_voluntario}.")

#################### Script entry point ##################################################

# Data conversion to nifty
if convert_dicom:
    convertir_todos_los_voluntarios(dicomDataPath,niftiOtuputPath)

# Group-wise concatenation
if concatenate_niftis:
    concatenar_niftis_en_grupos(
        niftiOtuputPath,
        apply_bias_correction=False,
        voluntarios_seleccionados=["S0032", "S0037", "S0041"]  # Example: ["001", "002"] to process only those volunteers
    )

# Fat Fraction calculation
if calculate_fat_fraction:
    calcular_fat_fraction_voluntarios(niftiOtuputPath)

