import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# 0. COLORES POR GRUPO
# ============================================================

group_colors = {
    "Control": "#1f77b4",      # azul
    "Transplante": "#ff7f0e",  # naranja
    "Dialisis": "#2ca02c"      # verde
}

# ============================================================
# 0.1 DICCIONARIO DE NOMBRES ANATÓMICOS (BASE: Vol...)
# ============================================================

muscle_labels = {
    # LUMBAR
    "Vol Lumbar_1": "Psoas R",
    "Vol Lumbar_2": "Muscle_2_R",
    "Vol Lumbar_3": "Quadratus R",
    "Vol Lumbar_4": "Multifidus R",

    "Vol Lumbar_5": "Psoas L",
    "Vol Lumbar_6": "Muscle_6_L",
    "Vol Lumbar_7": "Quadratus L",
    "Vol Lumbar_8": "Multifidus L",

    # PELVIS
    "Vol Pelvis_1": "Gluteus Max R",
    "Vol Pelvis_2": "Gluteus Med R",
    "Vol Pelvis_3": "Gluteus Min R",
    "Vol Pelvis_4": "TFL R",

    "Vol Pelvis_5": "Gluteus Max L",
    "Vol Pelvis_6": "Gluteus Med L",
    "Vol Pelvis_7": "Gluteus Min L",
    "Vol Pelvis_8": "TFL L",
}

# ============================================================
# 0.2 FUNCIÓN PARA OBTENER NOMBRE AMIGABLE DE CUALQUIER COLUMNA
#       Vol → usa muscle_labels directo
#       FF  → reemplaza 'FF ' por 'Vol '
#       T2_ → reemplaza 'T2_' por 'Vol '
#       *_norm → saca el '_norm' y luego aplica lo anterior
# ============================================================

def get_muscle_label(col: str) -> str:
    base = col

    # Sacar sufijo de normalización si existe
    if base.endswith("_norm"):
        base = base[:-5]  # quita "_norm"

    # Mapear FF → Vol
    if base.startswith("FF "):
        base = base.replace("FF ", "Vol ", 1)

    # Mapear T2_ → Vol
    elif base.startswith("T2_"):
        base = base.replace("T2_", "Vol ", 1)

    # Buscar en diccionario base
    return muscle_labels.get(base, col)


# ============================================================
# 1. CARGA Y LIMPIEZA DE DATOS
# ============================================================

ruta_csv = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/segmentations/all_subjects_volumes_and_ffs.csv"
df = pd.read_csv(ruta_csv)

print(f"\nCSV CARGADO CORRECTAMENTE\nFilas: {df.shape[0]}\nColumnas: {df.shape[1]}")
print("Grupos detectados:", df["Group"].unique())
print("\nPrimeras 5 columnas:", df.columns[:5], "\n")

for col in df.columns:
    if col not in ["Subject", "Group"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

print("Conversión a float completada.\n")

output_dir = os.path.join(os.path.dirname(ruta_csv), "boxplot")
os.makedirs(output_dir, exist_ok=True)
print(f"Los gráficos se guardarán en:\n{output_dir}\n")


# ============================================================
# 2. IDENTIFICACIÓN AUTOMÁTICA DE COLUMNAS
# ============================================================

def is_valid(col):
    forbidden = ["short", "shortfov", "skin", "skinfat"]
    return not any(f in col.lower() for f in forbidden)

vol_columns = [c for c in df.columns if c.startswith("Vol ") and is_valid(c)]
ff_columns = [c for c in df.columns if c.startswith("FF ") and is_valid(c)]
t2_columns = [c for c in df.columns if c.startswith("T2_")]

print("COLUMNAS USADAS:\nVolumen:", vol_columns,
      "\nFF:", ff_columns,
      "\nT2:", t2_columns, "\n")


# ============================================================
# FUNCIÓN: LEYENDA PARA MULTI
# ============================================================

def agregar_leyenda(grupos):
    handles = [
        plt.Rectangle((0,0), 1, 1, color=group_colors[g], alpha=0.7)
        for g in grupos
    ]
    plt.legend(
        handles, grupos,
        loc="upper right",
        bbox_to_anchor=(1.18, 1),   # fuera del gráfico
        borderaxespad=0.
    )


# ============================================================
# 3. BOXPLOT INDIVIDUAL
# ============================================================

def boxplot_por_grupo(data, variable, ylabel, output_path):

    grupos = data["Group"].unique()
    plot_data = data.copy()

    # ---------- FAT FRACTION EN PORCENTAJE ----------
    if variable.startswith("FF "):
        plot_data[variable] = plot_data[variable] * 100
        ylabel = "Fat Fraction (%)"

    datos = [plot_data.loc[plot_data["Group"] == g, variable].dropna() for g in grupos]
    ns = [len(v) for v in datos]

    friendly = get_muscle_label(variable)
    title = f"{friendly} – {ylabel}"

    plt.figure()

    bp = plt.boxplot(
        datos,
        labels=[f"{g}\n(n={n})" for g, n in zip(grupos, ns)],
        patch_artist=True
    )

    for patch, g in zip(bp["boxes"], grupos):
        patch.set_facecolor(group_colors[g])
        patch.set_alpha(0.7)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.ticklabel_format(style="plain", axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# 4. BOXPLOTS INDIVIDUALES
# ============================================================

def generar_boxplots(data, columns, ylabel, prefix):
    for col in columns:
        output = os.path.join(output_dir, f"{col}_{prefix}.png")
        boxplot_por_grupo(
            data=data,
            variable=col,
            ylabel=ylabel,
            output_path=output
        )


print("\nGenerando BOXPLOTS individuales...\n")

generar_boxplots(df, vol_columns, "Volumen (mm³)", "volumen_por_grupo")
generar_boxplots(df, ff_columns, "Fat Fraction (%)", "ff_por_grupo")

df_t2_valid = df.dropna(how="all", subset=t2_columns)
generar_boxplots(df_t2_valid, t2_columns, "T2 (ms)", "t2_por_grupo")

print("Boxplots individuales generados.\n")


# ============================================================
# 5. SEPARACIÓN ANATÓMICA
# ============================================================

def separar(anatomy, cols):
    return [c for c in cols if anatomy in c.lower()]

vol_lumbar = separar("lumbar", vol_columns)
vol_pelvis = separar("pelvis", vol_columns)
ff_lumbar = separar("lumbar", ff_columns)
ff_pelvis = separar("pelvis", ff_columns)


# ============================================================
# 6. BOXPLOT GLOBAL
# ============================================================

def boxplot_global(data, cols, title, ylabel, filename):

    grupos = data["Group"].unique()

    plot_data = data.copy()
    if any(c.startswith("FF ") for c in cols):
        plot_data[cols] = plot_data[cols] * 100
        ylabel = "Fat Fraction (%)"

    datos = []
    ns = []

    for g in grupos:
        vals = plot_data.loc[plot_data["Group"] == g, cols].values.flatten()
        vals = vals[~np.isnan(vals)]
        datos.append(vals)
        ns.append(len(vals))

    plt.figure()

    bp = plt.boxplot(
        datos,
        labels=[f"{g}\n(n={n})" for g, n in zip(grupos, ns)],
        patch_artist=True
    )

    for patch, g in zip(bp["boxes"], grupos):
        patch.set_facecolor(group_colors[g])
        patch.set_alpha(0.7)

    plt.title(f"{title} – Global comparison")
    plt.ylabel(ylabel)
    plt.ticklabel_format(style="plain", axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


boxplot_global(df, vol_lumbar, "Lumbar muscles volume", "Volumen (mm³)", "vol_lumbar_global.png")
boxplot_global(df, vol_pelvis, "Pelvis muscles volume", "Volumen (mm³)", "vol_pelvis_global.png")
boxplot_global(df, ff_lumbar, "Lumbar FF", "Fat Fraction (%)", "ff_lumbar_global.png")
boxplot_global(df, ff_pelvis, "Pelvis FF", "Fat Fraction (%)", "ff_pelvis_global.png")
boxplot_global(df_t2_valid, t2_columns, "Pelvis T2", "T2 (ms)", "t2_pelvis_global.png")


# ============================================================
# 7. BOXPLOT MULTI-MÚSCULO
# ============================================================

def boxplot_multi(data, cols, title, ylabel, filename):

    grupos = list(data["Group"].unique())
    plot_data = data.copy()

    # FF en porcentaje si corresponde
    if any(c.startswith("FF ") for c in cols):
        plot_data[cols] = plot_data[cols] * 100
        ylabel = "Fat Fraction (%)"

    G = len(grupos)
    M = len(cols)

    plt.figure(figsize=(max(12, M * 1.2), 7))

    positions = []
    box_data = []
    box_colors = []

    offset = 0.25   # separación entre grupos dentro del mismo músculo

    for i, col in enumerate(cols):
        for j, g in enumerate(grupos):
            vals = plot_data.loc[plot_data["Group"] == g, col].dropna()

            pos = i + (j - 1) * offset  # centro en i, desplazamiento por grupo

            positions.append(pos)
            box_data.append(vals)
            box_colors.append(group_colors[g])

    # Dibujar boxplots
    bp = plt.boxplot(
        box_data,
        positions=positions,
        widths=0.2,
        patch_artist=True,
        showfliers=True
    )

    # Colores por grupo
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Etiquetas SOLO por músculo (horizontales)
    plt.xticks(
        ticks=range(M),
        labels=[get_muscle_label(c) for c in cols],
        rotation=0
    )

    plt.subplots_adjust(bottom=0.22)  # margen inferior extra

    agregar_leyenda(grupos)

    plt.title(title, fontsize=20)
    plt.ylabel(ylabel)
    plt.ticklabel_format(style="plain", axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


# Volumen y FF multi
boxplot_multi(df, vol_lumbar, "Lumbar muscles volume", "Volumen (mm³)", "multi_vol_lumbar.png")
boxplot_multi(df, vol_pelvis, "Pelvis muscles volume", "Volumen (mm³)", "multi_vol_pelvis.png")
boxplot_multi(df, ff_lumbar, "Lumbar FF", "Fat Fraction (%)", "multi_ff_lumbar.png")
boxplot_multi(df, ff_pelvis, "Pelvis FF", "Fat Fraction (%)", "multi_ff_pelvis.png")
boxplot_multi(df_t2_valid, t2_columns, "Pelvis T2", "T2 (ms)", "multi_t2_pelvis.png")


# ============================================================
# 8. NORMALIZACIÓN POR PESO
# ============================================================

peso_col = "Weight (kg)"
vol_norm_cols = []

for col in vol_columns:
    new = col + "_norm"
    df[new] = df[col] / df[peso_col]
    vol_norm_cols.append(new)

vol_lumbar_norm = separar("lumbar", vol_norm_cols)
vol_pelvis_norm = separar("pelvis", vol_norm_cols)

generar_boxplots(df, vol_norm_cols, "Volumen normalizado (mm³/kg)", "vol_norm_por_grupo")

boxplot_global(df, vol_lumbar_norm, "Normalized lumbar volume", "Volumen normalizado", "vol_lumbar_norm_global.png")
boxplot_global(df, vol_pelvis_norm, "Normalized pelvis volume", "Volumen normalizado", "vol_pelvis_norm_global.png")

boxplot_multi(df, vol_lumbar_norm, "Normalized lumbar muscles volume", "Volumen normalizado", "multi_vol_lumbar_norm.png")
boxplot_multi(df, vol_pelvis_norm, "Normalized pelvis muscles volume", "Volumen normalizado", "multi_vol_pelvis_norm.png")
