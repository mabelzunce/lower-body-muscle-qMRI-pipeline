import SimpleITK as sitk
import numpy as np
import os
import csv

import SimpleITK as sitk
import numpy as np
import os

def _same_geom(a, b):
    return (a.GetSize()==b.GetSize() and a.GetSpacing()==b.GetSpacing() and
            a.GetOrigin()==b.GetOrigin() and a.GetDirection()==b.GetDirection())

def _resample_to(ref_img, mov_img, is_label=False, default_value=0.0):
    return sitk.Resample(
        mov_img, ref_img, sitk.Transform(),
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear,
        default_value,
        sitk.sitkUInt16 if is_label else sitk.sitkFloat32
    )

def generar_imagen_ff_robusta(carpeta, seg_path=None, normalize=True, p=95, write_nans=False):
    # 1) Detectar archivos
    fat_file = next((f for f in os.listdir(carpeta) if f.endswith("_f_dixon_concatenated.nii.gz")), None)
    if fat_file is None:
        fat_file = next((f for f in os.listdir(carpeta) if f.endswith("_f.nii.gz")), None)
    if fat_file is None:
        raise FileNotFoundError("❌ No se encontró archivo de grasa (_f o _f_dixon_concatenated).")

    water_file = next((f for f in os.listdir(carpeta) if f.endswith("_w_dixon_concatenated.nii.gz")), None)
    if water_file is None:
        water_file = next((f for f in os.listdir(carpeta) if f.endswith("_w.nii.gz")), None)
    if water_file is None:
        raise FileNotFoundError("❌ No se encontró archivo de agua (_w o _w_dixon_concatenated).")

    fat_path   = os.path.join(carpeta, fat_file)
    water_path = os.path.join(carpeta, water_file)
    print(f"Calculando Fat Fraction (robusta) para:\n  FAT: {fat_file}\n  WATER: {water_file}")

    # 2) Leer como float32
    fat   = sitk.Cast(sitk.ReadImage(fat_path),   sitk.sitkFloat32)
    water = sitk.Cast(sitk.ReadImage(water_path), sitk.sitkFloat32)

    # 3) Alinear geometría: water → espacio de fat (evita pequeñas descoincidencias)
    if not _same_geom(fat, water):
        print("⚠️ Geometría fat/water distinta. Resampleando water → fat …")
        water = _resample_to(fat, water, is_label=False, default_value=0.0)

    # 4) Máscara de soporte (cuerpo/ROI). Si hay segmentación, la usamos.
    if seg_path is not None:
        seg = sitk.ReadImage(seg_path)
        if not _same_geom(fat, seg):
            seg = _resample_to(fat, seg, is_label=True, default_value=0)
        support = sitk.Cast(seg>0, sitk.sitkUInt8)
    else:
        # soporte simple: cualquier señal en fat o water (evita aire)
        support = sitk.Cast((fat + water) > 0, sitk.sitkUInt8)

    fat_arr   = sitk.GetArrayFromImage(fat)
    water_arr = sitk.GetArrayFromImage(water)
    sup_arr   = sitk.GetArrayFromImage(support).astype(bool)

    # 5) (Opcional) Normalización por percentil p dentro del soporte
    if normalize:
        if sup_arr.any():
            pf = np.percentile(fat_arr[sup_arr],   p)
            pw = np.percentile(water_arr[sup_arr], p)
            pf = max(pf, 1e-6); pw = max(pw, 1e-6)
            fat_arr_n   = fat_arr   / pf
            water_arr_n = water_arr / pw
        else:
            fat_arr_n, water_arr_n = fat_arr, water_arr
    else:
        fat_arr_n, water_arr_n = fat_arr, water_arr

    # 6) FF = F/(F+W) con eps y máscara de soporte
    eps = 1e-6
    denom = fat_arr_n + water_arr_n + eps
    ff_arr = fat_arr_n / denom

    if write_nans:
        # NaN fuera de soporte (mejor para evitar sesgo si luego usás np.nanmean)
        ff_arr[~sup_arr] = np.nan
    else:
        # Cero fuera de soporte (comportamiento igual a tu función original)
        ff_arr[~sup_arr] = 0.0

    ff_img = sitk.GetImageFromArray(ff_arr)
    ff_img.CopyInformation(fat)  # hereda origen/spacing/dirección de fat (nuestro ref)

    out_path = os.path.join(carpeta, "fat_fraction_robust.nii.gz")
    sitk.WriteImage(ff_img, out_path)

    # 7) Log rápido
    if sup_arr.any():
        m_all   = np.nanmean(ff_arr) if write_nans else float(ff_arr[sup_arr].mean())
        m_mus   = None
        if seg_path is not None:
            seg_arr = sitk.GetArrayFromImage(seg)
            m = (seg_arr>0) & sup_arr
            if m.any():
                m_mus = float(np.nanmean(ff_arr[m]) if write_nans else ff_arr[m].mean())
        print(f"✅ FF guardada en: {out_path} | FF media(support): {m_all:.4f}" + (f" | FF media(ROI): {m_mus:.4f}" if m_mus is not None else ""))
    else:
        print(f"✅ FF guardada en: {out_path} | (sin soporte válido)")

def generar_imagen_ff(carpeta):
    fat_file = None
    if any(f.endswith("_f_dixon_concatenated.nii.gz") for f in os.listdir(carpeta)):
        fat_file = [f for f in os.listdir(carpeta) if f.endswith("_f_dixon_concatenated.nii.gz")][0]
    elif any(f.endswith("_f.nii.gz") for f in os.listdir(carpeta)):
        fat_file = [f for f in os.listdir(carpeta) if f.endswith("_f.nii.gz")][0]
    else:
        raise FileNotFoundError("❌ No se encontró archivo de grasa (_f o _f_dixon_concatenated) en la carpeta.")

    # Buscar archivo de agua (prioriza _w_dixon_concatenated.nii.gz)
    water_file = None
    if any(f.endswith("_w_dixon_concatenated.nii.gz") for f in os.listdir(carpeta)):
        water_file = [f for f in os.listdir(carpeta) if f.endswith("_w_dixon_concatenated.nii.gz")][0]
    elif any(f.endswith("_w.nii.gz") for f in os.listdir(carpeta)):
        water_file = [f for f in os.listdir(carpeta) if f.endswith("_w.nii.gz")][0]
    else:
        raise FileNotFoundError("❌ No se encontró archivo de agua (_w o _w_dixon_concatenated) en la carpeta.")

    # Leer imágenes
    fat_path = os.path.join(carpeta, fat_file)
    water_path = os.path.join(carpeta, water_file)
    print(f"🧠 Calculando Fat Fraction para:\n  FAT: {fat_file}\n  WATER: {water_file}")

    fatImage = sitk.Cast(sitk.ReadImage(fat_path), sitk.sitkFloat32)
    waterImage = sitk.Cast(sitk.ReadImage(water_path), sitk.sitkFloat32)

    waterfatImage = sitk.Add(fatImage, waterImage)
    fatfractionImage = sitk.Divide(fatImage, waterfatImage)
    fatfractionImage = sitk.Mask(fatfractionImage, waterfatImage > 0, outsideValue=0, maskingValue=0)

    output_path = os.path.join(carpeta, "fat_fraction.nii.gz")
    sitk.WriteImage(fatfractionImage, output_path)

    print(f"✅ Fat Fraction Image guardada en: {output_path}")


def calcular_volumen_y_ff_por_etiqueta(segmentation_path, fat_fraction_path, multilabelNum=1):
    segmentation = sitk.ReadImage(segmentation_path)
    fat_fraction = sitk.ReadImage(fat_fraction_path)

    # Si los tamaños no coinciden, resamplear FF al espacio de la segmentación
    if segmentation.GetSize() != fat_fraction.GetSize():
        print("Geometría distinta. Resampleando FF al espacio de la segmentación")
        fat_fraction = sitk.Resample(
            fat_fraction,
            segmentation,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            fat_fraction.GetPixelID()
        )

    seg_arr = sitk.GetArrayFromImage(segmentation)
    ff_arr = sitk.GetArrayFromImage(fat_fraction)

    spacing = segmentation.GetSpacing()
    print(f"spacing: {spacing}")
    voxel_volume = np.prod(spacing)
    print(f"voxel_volume: {voxel_volume}")

    vols = []
    ffs = []
    total_vol = 0
    total_ff_pesado = 0

    for label in range(1, multilabelNum + 1):
        print(f"label:{label}")
        mask = seg_arr == label
        n_voxels = np.sum(mask)
        print(f"Volumen (en voxeles): {n_voxels}")
        vol = n_voxels * voxel_volume
        ff = np.mean(ff_arr[mask]) if n_voxels > 0 else 0
        vols.append(vol)
        ffs.append(ff)
        total_vol += vol
        total_ff_pesado += ff * vol

    mean_ff = total_ff_pesado / total_vol if total_vol > 0 else 0
    print(f"\nVolumen total (en unidades físicas): {total_vol}")
    print(f"Fat Fraction media ponderada: {mean_ff:.4f}")
    return vols, ffs, total_vol, mean_ff



def guardar_en_csv(path_csv, sujeto, vols, ffs, total_vol, mean_ff, multilabelNum):
    if not os.path.isfile(path_csv):
        with open(path_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Subject"]
            header += [f"Vol {i}" for i in range(1, multilabelNum + 1)]
            header += [f"FF {i}" for i in range(1, multilabelNum + 1)]
            header += ["Total Volume", "Mean FF"]
            writer.writerow(header)

    with open(path_csv, "a", newline="") as f:
        writer = csv.writer(f)
        row = [sujeto] + vols + ffs + [total_vol, mean_ff]
        writer.writerow(row)

def resamplear_a_espacio_de(imagen_a_resamplear, imagen_referencia):
    return sitk.Resample(
        imagen_a_resamplear,
        imagen_referencia,
        sitk.Transform(),  # Identidad
        sitk.sitkLinear,   # Interpolación lineal (para FF)
        0.0,               # Valor fuera del campo
        imagen_a_resamplear.GetPixelID()
    )

def registrar_y_comparar_ff(fixed_path, moving_path, custom_param_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    transform_output_path = os.path.join(output_folder, "Transform.txt")
    registered_path = os.path.join(output_folder, "fat_fraction_registered.nii.gz")
    diff_path = os.path.join(output_folder, "difference_fixed_vs_registered.nii.gz")

    # ==== CARGA IMÁGENES ====
    fixed_image = sitk.ReadImage(fixed_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_path, sitk.sitkFloat32)

    # ==== REGISTRACIÓN CON ELASTIX ====
    parameterMapVector = sitk.VectorOfParameterMap()
    custom_param_map = sitk.ReadParameterFile(custom_param_path)
    parameterMapVector.append(custom_param_map)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.SetLogToConsole(True)
    elastixImageFilter.Execute()

    # ==== GUARDAR RESULTADOS ====
    registered_image = elastixImageFilter.GetResultImage()
    sitk.WriteImage(registered_image, registered_path)

    diff_image = sitk.Subtract(fixed_image, registered_image)
    sitk.WriteImage(diff_image, diff_path)

    elastixImageFilter.WriteParameterFile(elastixImageFilter.GetParameterMap()[0], transform_output_path)

    print("✅Registración completada.")
    print(f"Imagen registrada guardada en: {registered_path}")
    print(f"Imagen diferencia guardada en: {diff_path}")

def audit_overlap(seg_path, ff_path):
    seg = sitk.ReadImage(seg_path)
    ff  = sitk.ReadImage(ff_path)

    # Resamplear FF al espacio de la segmentación si hace falta
    if (seg.GetSize()!=ff.GetSize() or seg.GetSpacing()!=ff.GetSpacing() or
        seg.GetOrigin()!=ff.GetOrigin() or seg.GetDirection()!=ff.GetDirection()):
        ff = sitk.Resample(ff, seg, sitk.Transform(), sitk.sitkLinear, 0.0, sitk.sitkFloat32)

    seg_arr = sitk.GetArrayFromImage(seg)
    ff_arr  = sitk.GetArrayFromImage(ff)

    mask = seg_arr > 0
    n_mask = int(mask.sum())
    if n_mask == 0:
        print("⚠️ La máscara no tiene voxeles > 0")
        return

    ff_mask = ff_arr[mask]
    n_zeros = int(np.sum(ff_mask == 0))
    frac_zeros = n_zeros / n_mask

    print(f"[AUDIT] Size seg {seg.GetSize()} | Size ff {ff.GetSize()}")
    print(f"[AUDIT] Zero-voxels dentro de máscara: {n_zeros}/{n_mask} ({100*frac_zeros:.2f}%)")
    print(f"[AUDIT] FF mean (incluye ceros): {float(ff_mask.mean()):.6f}")
    if n_zeros < n_mask:
        print(f"[AUDIT] FF mean (excluyendo ceros): {float(ff_mask[ff_mask>0].mean()):.6f}")

def audit_raw_intensity(f_path, w_path, seg_path=None):
    f = sitk.Cast(sitk.ReadImage(f_path), sitk.sitkFloat32)
    w = sitk.Cast(sitk.ReadImage(w_path), sitk.sitkFloat32)

    arr_f = sitk.GetArrayFromImage(f)
    arr_w = sitk.GetArrayFromImage(w)

    # Body mask simple para evitar aire (opcionalmente usa tu segmentación si querés)
    if seg_path:
        seg = sitk.ReadImage(seg_path)
        # Traer seg al espacio de F/W si hiciera falta
        if seg.GetSize()!=f.GetSize() or seg.GetSpacing()!=f.GetSpacing() or seg.GetOrigin()!=f.GetOrigin() or seg.GetDirection()!=f.GetDirection():
            seg = sitk.Resample(seg, f, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
        m = sitk.GetArrayFromImage(seg) > 0
    else:
        # Heurística: voxeles con señal (fuera de aire)
        m = (arr_f>0) | (arr_w>0)

    f_m = arr_f[m]
    w_m = arr_w[m]

    print(f"[RAW] fat min/max: {float(f_m.min())} / {float(f_m.max())}")
    print(f"[RAW] water min/max: {float(w_m.min())} / {float(w_m.max())}")
    print(f"[RAW] mean fat: {float(f_m.mean()):.3f} | mean water: {float(w_m.mean()):.3f}")
    print(f"[RAW] mean(F/(F+W)): {float((f_m/(f_m+w_m+1e-6)).mean()):.6f}")

# =================== USO ===================

carpeta_base = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/comparacionFatFraction/VersionConcateneted/"

# 1. Generar fat fraction
#generar_imagen_ff_robusta(carpeta_base)
#generar_imagen_ff(carpeta_base)

# output_root = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/comparacionFatFraction/"
# csv_path = os.path.join(output_root, "vol_ff_segmentation.csv")

# 2. Calcular volumen y FF
subject_id = "Sin concatenar"
print(subject_id)
seg_path = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/comparacionFatFraction/Version SinConcatenar/Segmentation preview-Segment_1-label.mhd"
ff_path = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/comparacionFatFraction/Version SinConcatenar/fat_fraction_robust.nii.gz"
vols, ffs, total_vol, mean_ff = calcular_volumen_y_ff_por_etiqueta(seg_path, ff_path, multilabelNum=1)
# guardar_en_csv(csv_path, subject_id, vols, ffs, total_vol, mean_ff, multilabelNum=1)

subject_id = "Concatenadas"
print(subject_id)
seg_path = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/comparacionFatFraction/VersionConcateneted/BELZUNCE_in_dixon_concatenated_Segmentation-3d-label.mhd"
ff_path = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/comparacionFatFraction/VersionConcateneted/fat_fraction_robust.nii.gz"
vols, ffs, total_vol, mean_ff = calcular_volumen_y_ff_por_etiqueta(seg_path, ff_path, multilabelNum=1)
# guardar_en_csv(csv_path, subject_id, vols, ffs, total_vol, mean_ff, multilabelNum=1)

# 3. Guardar en CSV
#output_root = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/comparacionFatFraction/"
#csv_path = os.path.join(output_root, "vol_ff_segmentation.csv")
#guardar_en_csv(csv_path, subject_id, vols, ffs, total_vol, mean_ff, multilabelNum)
#print(f"Resultados guardados en: {csv_path}")


fixed_path= "/data/MuscleSegmentation/Data/Gluteus&Lumbar/comparacionFatFraction/VersionConcateneted/fat_fraction_robust.nii.gz"
moving_path= "/data/MuscleSegmentation/Data/Gluteus&Lumbar/comparacionFatFraction/Version SinConcatenar/fat_fraction_robust.nii.gz"
custom_param_path= "/home/german/lower-body-muscle-qMRI-pipeline/src/segmentation/Parameters_Rigid_NCC.txt"
output_folder= "/data/MuscleSegmentation/Data/Gluteus&Lumbar/comparacionFatFraction/output_registro_robust"
registrar_y_comparar_ff(fixed_path,moving_path,custom_param_path,output_folder)
