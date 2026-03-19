import SimpleITK as sitk

fixed = sitk.ReadImage("/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifty_output/S0014/pelvis_crop.nii.gz")
moving = sitk.ReadImage("/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifty_output/S0014/29_t2starmap_fl2d_tra_mbh.nii.gz")

# Convertir a mismo tipo
fixed = sitk.Cast(fixed, sitk.sitkFloat32)
moving = sitk.Cast(moving, sitk.sitkFloat32)

# Si la moving es 4D → convertir a 3D
if moving.GetDimension() == 4:
    print("⚠️ moving es 4D, extrayendo el primer volumen...")
    size = list(moving.GetSize())
    size[3] = 0
    moving = sitk.Extract(moving, size=size, index=[0,0,0,0])

label = sitk.ReadImage("/data/MuscleSegmentation/Data/Gluteus&Lumbar/segmentations/S0014/S0014_pelvis_segmentation.mhd")

# --------------------------- REGISTRO ---------------------------
registration = sitk.ImageRegistrationMethod()
registration.SetMetricAsMeanSquares()
registration.SetOptimizerAsRegularStepGradientDescent(
    learningRate=2.0, minStep=1e-4, numberOfIterations=100
)
registration.SetInterpolator(sitk.sitkLinear)

# TRANSFORMACIÓN INICIAL
initial_transform = sitk.TranslationTransform(fixed.GetDimension())
registration.SetInitialTransform(initial_transform, inPlace=False)

print("🔄 Ejecutando registro...")
transform = registration.Execute(fixed, moving)

# Registrar la imagen moving
moving_reg = sitk.Resample(
    moving,
    fixed,
    transform,
    sitk.sitkLinear,
    0.0,
    moving.GetPixelID()
)
sitk.WriteImage(
    moving_reg,
    "/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifty_output/S0014/t2star_registered.nii.gz"
)
print("✔ moving registrada guardada.")

# --------------------------- REGISTRAR MÁSCARA ---------------------------
label_reg = sitk.Resample(
    label,
    fixed,
    transform,
    sitk.sitkNearestNeighbor,
    0,
    label.GetPixelID()
)
sitk.WriteImage(
    label_reg,
    "/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifty_output/S0014/label_registered.nii.gz"
)
print("✔ máscara registrada guardada.")
