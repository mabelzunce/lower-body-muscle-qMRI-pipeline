import SimpleITK as sitk
import numpy as np
import pandas as pd
import os, sys, csv
import re
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
from utils.utils import ApplyBiasCorrection, create_segmentation_overlay_animated_gif

# --- Configuración ---
base_dir_masks = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/segmentations'
base_dir_maps = '/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifty_output'
output_csv = os.path.join(base_dir_masks, 'T2star_means_pelvis.csv')

resultados = []

subjects = sorted([d for d in os.listdir(base_dir_masks) if d.startswith('S')])

for subj in subjects:
    mask_dir = os.path.join(base_dir_masks, subj)
    map_dir = os.path.join(base_dir_maps, subj)

    if not os.path.isdir(map_dir):
        print(f"⚠️ No existe carpeta de mapas para {subj}")
        continue

    # --- Buscar mapa T2* ---
    maps = [f for f in os.listdir(map_dir)
        if f.lower().endswith('images.nii.gz')
           and 't2' in f.lower()]
    if not maps:
        print(f"⚠️ No se encontró mapa T2* en {subj}")
        continue

    def get_number(name):
        m = re.match(r'(\d+)_', name)
        return int(m.group(1)) if m else 999
    maps.sort(key=get_number)
    t2_path = os.path.join(map_dir, maps[0])

    # --- Buscar máscara de pelvis ---
    pelvis_mask = [f for f in os.listdir(mask_dir) if f.endswith('pelvis_segmentation.mhd')]
    if not pelvis_mask:
        print(f"⚠️ No se encontró máscara de pelvis en {subj}")
        continue

    mask_path = os.path.join(mask_dir, pelvis_mask[0])

    print(f"\n🧠 Procesando {subj}")
    print(f"Mapa T2*: {maps[0]}")
    print(f"Máscara: {pelvis_mask[0]}")

    # --- Leer imágenes ---
    t2_img = sitk.ReadImage(t2_path, sitk.sitkFloat32)
    mask_img = sitk.ReadImage(mask_path, sitk.sitkUInt16)

    # --- Resamplear máscara al espacio del mapa ---
    if t2_img.GetSize() != mask_img.GetSize() or t2_img.GetSpacing() != mask_img.GetSpacing():
        print(f"↪️ Resampleando máscara de {subj} al espacio del mapa T2*...")
        mask_img = sitk.Resample(
            mask_img,
            t2_img,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,  # mantiene labels enteros
            0,
            mask_img.GetPixelID(),
        )

    # --- Generar GIF de overlay para control visual ---
    # Buscar imagen anatómica (de la misma secuencia o sujeto)
    anat_candidates = [f for f in os.listdir(map_dir) if 'artcadera' in f and f.endswith('.nii.gz')]

    if anat_candidates:
        anat_path = os.path.join(map_dir, anat_candidates[0])
        print(f"🖼️ Usando imagen anatómica para el GIF: {anat_candidates[0]}")
        anat_img = sitk.ReadImage(anat_path, sitk.sitkFloat32)
        # Siempre es 4D → tomar solo el primer volumen (eco 0)
        size = list(anat_img.GetSize())
        size[3] = 0  # dejar la última dimensión en 0 (extrae 1 volumen)
        anat_img = sitk.Extract(anat_img, size=size, index=[0, 0, 0, 0])
        # Resamplear la imagen anatómica al espacio del mapa (si difiere)
        if anat_img.GetSize() != t2_img.GetSize():
            anat_img = sitk.Resample(
                anat_img,
                t2_img,
                sitk.Transform(),
                sitk.sitkLinear,
                0.0,
                anat_img.GetPixelID(),
            )
        gif_input_img = anat_img
    else:
        print("⚠️ No se encontró imagen anatómica, usando el mapa T2* como fondo.")
        gif_input_img = t2_img

    gif_path = os.path.join(mask_dir, f"{subj}_T2_overlay.gif")
    create_segmentation_overlay_animated_gif(gif_input_img, mask_img, gif_path)

    # --- Calcular T2* medio por label ---
    t2_arr = sitk.GetArrayFromImage(t2_img)
    mask_arr = sitk.GetArrayFromImage(mask_img)
    unique_labels = [int(x) for x in np.unique(mask_arr) if x > 0]

    for label in unique_labels:
        vals = t2_arr[mask_arr == label]
        if len(vals) == 0:
            continue

        mean_t2 = np.mean(vals)
        std_t2 = np.std(vals)
        n_vox = len(vals)

        resultados.append([subj, 'pelvis', label, mean_t2, std_t2, n_vox])
        print(f"  Label {label}: {mean_t2:.2f} ± {std_t2:.2f} ms ({n_vox} voxels)")

# Convertir a DataFrame
df = pd.DataFrame(resultados, columns=['Subject', 'Region', 'Label', 'Mean_T2', 'Std_T2', 'N_voxels'])

# Pivotear para tener una columna por label
df_pivot = df.pivot_table(
    index='Subject',
    columns='Label',
    values='Mean_T2'
)

# Renombrar columnas para que queden bonitas: T2_Pelvis_1, T2_Pelvis_2...
df_pivot.columns = [f"T2_Pelvis_{int(c)}" for c in df_pivot.columns]

# Reset index para que Subject sea una columna más
df_pivot = df_pivot.reset_index()

# Guardar CSV final (una fila por voluntario)
df_pivot.to_csv(output_csv, index=False)

print(f"\n✅ CSV guardado con una fila por voluntario y una columna por label en {output_csv}")

# --------------------------------------------------------
# --- ACTUALIZAR CSV MAESTRO CON LOS VALORES DE T2* ------
# --------------------------------------------------------

csv_master_path = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/segmentations/all_subjects_volumes_and_ffs.csv"

print(f"\n📘 Actualizando CSV maestro: {csv_master_path}")

# Leer CSV existente
df_master = pd.read_csv(csv_master_path)

# Asegurarse de que la columna Subject exista
if "Subject" not in df_master.columns:
    raise ValueError("❌ ERROR: El CSV maestro no contiene columna 'Subject'")

# Para cada columna T2_Pelvis_X producida por df_pivot
for col in df_pivot.columns:
    if col == "Subject":
        continue

    # Si la columna no existe en el maestro → crearla
    if col not in df_master.columns:
        print(f"🆕 Creando columna nueva: {col}")
        df_master[col] = np.nan

    # Actualizar valores por sujeto
    for idx, row in df_pivot.iterrows():
        subject = row["Subject"]
        value = row[col]

        # Filtrar filas donde Subject coincida
        mask = df_master["Subject"] == subject
        if mask.any():
            df_master.loc[mask, col] = value
        else:
            print(f"⚠️ Voluntario {subject} no está en el CSV maestro")

# Guardar cambios (SOBREESCRIBIENDO el archivo CSV maestro)
df_master.to_csv(csv_master_path, index=False)

print(f"✅ CSV maestro actualizado exitosamente en {csv_master_path}")

