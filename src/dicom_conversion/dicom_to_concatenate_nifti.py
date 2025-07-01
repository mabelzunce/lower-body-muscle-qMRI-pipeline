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

        os.makedirs(carpeta_salida, exist_ok=True)

        dicom2nifti.convert_directory(carpeta_dicom, carpeta_salida,
                              compression=True, reorient=True)
        """ for nombre_subcarpeta in sorted(os.listdir(carpeta_dicom)):
            ruta_subcarpeta = os.path.join(carpeta_dicom, nombre_subcarpeta)
            if not os.path.isdir(ruta_subcarpeta):
                continue

            slices = cargar_slices_dicom(ruta_subcarpeta)
            print(f"📂 {nombre_voluntario}/{nombre_subcarpeta} → {len(slices)} slices")

            if len(slices) < 25:
                print(f"⚠️ Skipped: less than 25 slices.")
                continue

            output_file = os.path.join(carpeta_salida, f"{nombre_subcarpeta}.nii.gz")
            convertir_a_nifti(slices, output_file) """

def concatenar_niftis_en_grupos(carpeta_salida_base):
    print("concatenar_niftis_en_grupos")
    for nombre_voluntario in sorted(os.listdir(carpeta_salida_base)):
        ruta_voluntario = os.path.join(carpeta_salida_base, nombre_voluntario)
        if not os.path.isdir(ruta_voluntario):
            continue

        archivos_nii = [f for f in os.listdir(ruta_voluntario) if f.endswith(".nii.gz")]

        grupos = {}
        archivos_unicos = []

        pattern = re.compile(r".*_(F|IN|OPP|W)_(\d+)\.nii\.gz$")

        for archivo in archivos_nii:
            m = pattern.match(archivo)
            if not m:
                archivos_unicos.append(archivo)
                continue
            grupo = m.group(1)
            numero = int(m.group(2))
            key = f"{nombre_voluntario}_{grupo}"
            grupos.setdefault(key, []).append((archivo, numero))

        for nombre_grupo, lista_archivos in grupos.items():
            if len(lista_archivos) < 2:
                archivos_unicos.extend([x[0] for x in lista_archivos])
                continue

            # Filter only files containing "TRA" in the name
            lista_archivos_tra = [x for x in lista_archivos if "TRA" in x[0]]

            if len(lista_archivos_tra) < 2:
                archivos_unicos.extend([x[0] for x in lista_archivos_tra])
                continue

            # Sort by number
            lista_archivos_tra.sort(key=lambda x: x[1])
            archivos_ordenados = [os.path.join(ruta_voluntario, x[0]) for x in lista_archivos_tra]

            print(f"\n📁 Group {nombre_grupo} → the following files will be concatenated:")
            for ruta in archivos_ordenados:
                print(f"   - {os.path.basename(ruta)}")

            # Load and prepare data
            imgs = [nib.load(archivo) for archivo in archivos_ordenados]
            datos = [img.get_fdata() for img in imgs]
            afines = [img.affine for img in imgs]

            # Reverse order so the image with the lowest number is on top
            datos = datos[::-1]

            # Check that all affines are similar
            if not all(np.allclose(afines[0], af, atol=1e-5) for af in afines):
                print(f"⚠️ Different affines in group {nombre_grupo}, skipping")
                continue

            # Concatenate volumes
            concatenado = np.concatenate(datos, axis=2)
            print(f"Concatenated shape: {concatenado.shape}")

            nifti_concat = nib.Nifti1Image(concatenado, afines[0])
            output_file = os.path.join(ruta_voluntario, f"{nombre_grupo}_concatenated.nii.gz")
            nib.save(nifti_concat, output_file)
            print(f"✅ Concatenated saved: {output_file}")

        if archivos_unicos:
            print(f"🟨 Unique (not concatenated) files in {nombre_voluntario}:")
            for archivo in archivos_unicos:
                print(f"  {archivo}")

# Script entry point
convertir_todos_los_voluntarios(
    dicomDataPath,
    niftiOtuputPath)

# Group-wise concatenation
concatenar_niftis_en_grupos(niftiOtuputPath)
