import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# 0. COLORES POR GRUPO
# ============================================================

group_colors = {
    "Control": "#1f77b4",
    "Transplante": "#ff7f0e",
    "Dialisis": "#2ca02c"
}

# ============================================================
# 0.1 DICCIONARIO DE NOMBRES ANATÓMICOS
# ============================================================

muscle_labels = {

    # LUMBAR
    "Vol Lumbar_1": "Psoas R",
    "Vol Lumbar_2": "MuscleX_R",
    "Vol Lumbar_3": "Quadratus R",
    "Vol Lumbar_4": "Multifidus R",

    "Vol Lumbar_5": "Psoas L",
    "Vol Lumbar_6": "Illiacus_L",
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


def get_muscle_label(col):

    base = col

    if base.endswith("_norm"):
        base = base[:-5]

    if base.startswith("FF "):
        base = base.replace("FF ", "Vol ", 1)

    elif base.startswith("T2_"):
        base = base.replace("T2_", "Vol ", 1)

    return muscle_labels.get(base, col)


# ============================================================
# 1. CARGA DE DATOS
# ============================================================

ruta_csv = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/segmentations/all_subjects_volumes_and_ffs.csv"

df = pd.read_csv(ruta_csv)

print(f"\nCSV cargado\nFilas: {df.shape[0]}  Columnas: {df.shape[1]}")
print("Grupos:", df["Group"].unique())

for col in df.columns:
    if col not in ["Subject", "Group"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")


# ============================================================
# 2. CREACIÓN DE CARPETAS
# ============================================================

base_output = os.path.join(os.path.dirname(ruta_csv), "boxplot")

dirs = {

    "multi": os.path.join(base_output, "multi"),
    "global": os.path.join(base_output, "global"),

    "vol_ind": os.path.join(base_output, "individual_volumen"),
    "ff_ind": os.path.join(base_output, "individual_ff"),
    "t2_ind": os.path.join(base_output, "individual_t2"),
    "volnorm_ind": os.path.join(base_output, "individual_volumen_normalizado")

}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

print("Carpetas creadas en:", base_output)


# ============================================================
# 3. FILTRO VOLUNTARIOS
# ============================================================

df["Subject_num"] = df["Subject"].str.extract(r'(\d+)').astype(float)
df_40 = df[df["Subject_num"] >= 40].copy()


def get_group_counts(data):

    counts = data.groupby("Group")["Subject"].nunique()

    return " | ".join([f"{g}: n={counts[g]}" for g in counts.index])


# ============================================================
# 4. IDENTIFICACIÓN DE COLUMNAS
# ============================================================

def is_valid(col):
    forbidden = ["short", "shortfov", "skin", "skinfat"]
    return not any(f in col.lower() for f in forbidden)


vol_columns = [c for c in df.columns if c.startswith("Vol ") and is_valid(c)]
ff_columns = [c for c in df.columns if c.startswith("FF ") and is_valid(c)]
t2_columns = [c for c in df.columns if c.startswith("T2_")]

df_t2_valid = df.dropna(how="all", subset=t2_columns)
df_t2_40 = df_40.dropna(how="all", subset=t2_columns)


# ============================================================
# 5. LEYENDA MULTI
# ============================================================

def agregar_leyenda(grupos):

    handles = [plt.Rectangle((0,0),1,1,color=group_colors[g],alpha=0.7) for g in grupos]

    plt.legend(handles, grupos, loc="upper right", bbox_to_anchor=(1.18,1))


# ============================================================
# 6. BOXPLOT INDIVIDUAL
# ============================================================

def boxplot_por_grupo(data, variable, ylabel, output_path):

    grupos = data["Group"].unique()
    plot_data = data.copy()

    if variable.startswith("FF "):
        plot_data[variable] = plot_data[variable] * 100
        ylabel = "Fat Fraction (%)"

    datos = [plot_data.loc[plot_data["Group"] == g, variable].dropna() for g in grupos]

    ns = [len(v) for v in datos]

    friendly = get_muscle_label(variable)

    plt.figure()

    bp = plt.boxplot(datos,
                     labels=[f"{g}\n(n={n})" for g, n in zip(grupos, ns)],
                     patch_artist=True)

    for patch, g in zip(bp["boxes"], grupos):
        patch.set_facecolor(group_colors[g])
        patch.set_alpha(0.7)

    plt.title(f"{friendly} – {ylabel}")
    plt.ylabel(ylabel)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)

    plt.close()


def generar_boxplots(data, columns, ylabel, output_folder):

    for col in columns:

        output = os.path.join(output_folder, f"{col}.png")

        boxplot_por_grupo(data, col, ylabel, output)


print("\nGenerando boxplots individuales")

generar_boxplots(df, vol_columns, "Volumen (mm³)", dirs["vol_ind"])
generar_boxplots(df, ff_columns, "Fat Fraction (%)", dirs["ff_ind"])
generar_boxplots(df_t2_valid, t2_columns, "T2 (ms)", dirs["t2_ind"])


# ============================================================
# 7. SEPARACIÓN ANATÓMICA
# ============================================================

def separar(anatomy, cols):

    return [c for c in cols if anatomy in c.lower()]


vol_lumbar = separar("lumbar", vol_columns)
vol_pelvis = separar("pelvis", vol_columns)

ff_lumbar = separar("lumbar", ff_columns)
ff_pelvis = separar("pelvis", ff_columns)


# ============================================================
# 8. BOXPLOT GLOBAL
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

    bp = plt.boxplot(datos,
                     labels=[f"{g}\n(n={n})" for g, n in zip(grupos, ns)],
                     patch_artist=True)

    for patch, g in zip(bp["boxes"], grupos):
        patch.set_facecolor(group_colors[g])
        patch.set_alpha(0.7)

    plt.title(f"{title} – Global comparison")

    plt.ylabel(ylabel)

    plt.tight_layout()

    plt.savefig(os.path.join(dirs["global"], filename), dpi=300)

    plt.close()


boxplot_global(df, vol_lumbar, "Lumbar muscles volume", "Volumen (mm³)", "vol_lumbar.png")
boxplot_global(df, vol_pelvis, "Pelvis muscles volume", "Volumen (mm³)", "vol_pelvis.png")

boxplot_global(df, ff_lumbar, "Lumbar FF", "Fat Fraction (%)", "ff_lumbar.png")
boxplot_global(df, ff_pelvis, "Pelvis FF", "Fat Fraction (%)", "ff_pelvis.png")

boxplot_global(df_t2_valid, t2_columns, "Pelvis T2", "T2 (ms)", "t2_pelvis.png")


# ============================================================
# 9. BOXPLOT MULTI
# ============================================================

def boxplot_multi(data, cols, title, ylabel, filename):

    grupos = list(data["Group"].unique())

    plot_data = data.copy()

    if any(c.startswith("FF ") for c in cols):

        plot_data[cols] = plot_data[cols] * 100

        ylabel = "Fat Fraction (%)"

    plt.figure(figsize=(max(12, len(cols)*1.2),7))

    positions = []
    box_data = []
    box_colors = []

    offset = 0.25

    for i, col in enumerate(cols):

        for j, g in enumerate(grupos):

            vals = plot_data.loc[plot_data["Group"] == g, col].dropna()

            pos = i + (j-1)*offset

            positions.append(pos)

            box_data.append(vals)

            box_colors.append(group_colors[g])

    bp = plt.boxplot(box_data,
                     positions=positions,
                     widths=0.2,
                     patch_artist=True,
                     showfliers=True)

    for patch, color in zip(bp["boxes"], box_colors):

        patch.set_facecolor(color)

        patch.set_alpha(0.7)

    plt.xticks(range(len(cols)),
               [get_muscle_label(c) for c in cols],
               rotation=0)

    plt.subplots_adjust(bottom=0.22)

    agregar_leyenda(grupos)

    counts_text = get_group_counts(data)

    plt.title(f"{title}\n{counts_text}")

    plt.ylabel(ylabel)

    plt.tight_layout()

    plt.savefig(os.path.join(dirs["multi"], filename), dpi=300)

    plt.close()


boxplot_multi(df, vol_lumbar, "Lumbar muscles volume", "Volumen (mm³)", "multi_vol_lumbar.png")
boxplot_multi(df, vol_pelvis, "Pelvis muscles volume", "Volumen (mm³)", "multi_vol_pelvis.png")

boxplot_multi(df, ff_lumbar, "Lumbar FF", "Fat Fraction (%)", "multi_ff_lumbar.png")
boxplot_multi(df, ff_pelvis, "Pelvis FF", "Fat Fraction (%)", "multi_ff_pelvis.png")

boxplot_multi(df_t2_valid, t2_columns, "Pelvis T2", "T2 (ms)", "multi_t2_pelvis.png")
boxplot_multi(df_t2_40, t2_columns, "Pelvis T2 ≥40", "T2 (ms)", "multi_t2_pelvis_40.png")


# ============================================================
# 10. NORMALIZACIÓN POR PESO
# ============================================================

peso_col = "Weight (kg)"

vol_norm_cols = []

for col in vol_columns:

    new = col + "_norm"

    df[new] = df[col] / df[peso_col]

    vol_norm_cols.append(new)

vol_lumbar_norm = separar("lumbar", vol_norm_cols)
vol_pelvis_norm = separar("pelvis", vol_norm_cols)

generar_boxplots(df, vol_norm_cols,
                 "Volumen normalizado (mm³/kg)",
                 dirs["volnorm_ind"])

boxplot_global(df, vol_lumbar_norm,
               "Normalized lumbar volume",
               "Volumen normalizado",
               "vol_lumbar_norm.png")

boxplot_global(df, vol_pelvis_norm,
               "Normalized pelvis volume",
               "Volumen normalizado",
               "vol_pelvis_norm.png")

boxplot_multi(df, vol_lumbar_norm,
              "Normalized lumbar muscles volume",
              "Volumen normalizado",
              "multi_vol_lumbar_norm.png")

boxplot_multi(df, vol_pelvis_norm,
              "Normalized pelvis muscles volume",
              "Volumen normalizado",
              "multi_vol_pelvis_norm.png")