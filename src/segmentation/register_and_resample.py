import SimpleITK as sitk

#fixed_path = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifti_pelvis/S0043/S0043_I.nii.gz"
#moving_path = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifty_output/S0043/31_t2_images.nii.gz"
moving_path = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifti_pelvis/S0043/S0043_I.nii.gz"
fixed_path = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifty_output/S0043/31_t2_images.nii.gz"
moving_out_path = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/nifty_output/S0043/3S0043_I_registered.nii.gz"

REGISTER_LABEL = True
label_path = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/segmentations/S0043/S0043_pelvis_segmentation.mhd"
label_out_path = "/data/MuscleSegmentation/Data/Gluteus&Lumbar/segmentations/S0043/S0043_pelvis_segmentation_registered.mhd"

# --------------------------- LOAD ---------------------------
fixed = sitk.Cast(sitk.ReadImage(fixed_path), sitk.sitkFloat32)
moving = sitk.Cast(sitk.ReadImage(moving_path), sitk.sitkFloat32)

print("FIXED origin:", fixed.GetOrigin())
print("MOVING origin:", moving.GetOrigin())

print("FIXED spacing:", fixed.GetSpacing())
print("MOVING spacing:", moving.GetSpacing())

print("FIXED direction:", fixed.GetDirection())
print("MOVING direction:", moving.GetDirection())

if moving.GetDimension() == 4:
    size = list(moving.GetSize())
    size[3] = 0
    moving = sitk.Extract(moving, size=size, index=[0, 0, 0, 0])

if REGISTER_LABEL:
    label = sitk.ReadImage(label_path)

# --------------------------- REGISTRATION ---------------------------
registration = sitk.ImageRegistrationMethod()
#registration.SetMetricAsMeanSquares()
registration.SetOptimizerAsRegularStepGradientDescent(
    learningRate=2.0,
    minStep=1e-4,
    numberOfIterations=100
)
registration.SetInterpolator(sitk.sitkLinear)

initial_transform = sitk.CenteredTransformInitializer(
    fixed,
    moving,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

registration.SetInitialTransform(initial_transform, inPlace=False)

registration.SetMetricAsMattesMutualInformation(50)
registration.SetMetricSamplingStrategy(registration.RANDOM)
registration.SetMetricSamplingPercentage(0.2)
registration.SetOptimizerScalesFromPhysicalShift()

transform = registration.Execute(fixed, moving)

# --------------------------- RESAMPLE IMAGE ---------------------------
moving_reg = sitk.Resample(
    moving,
    fixed,
    transform,
    sitk.sitkLinear,
    0.0,
    moving.GetPixelID()
)

sitk.WriteImage(moving_reg, moving_out_path)

# --------------------------- RESAMPLE LABEL ---------------------------
if REGISTER_LABEL:
    label_reg = sitk.Resample(
        label,
        fixed,
        transform,
        sitk.sitkNearestNeighbor,
        0,
        label.GetPixelID()
    )
    sitk.WriteImage(label_reg, label_out_path)