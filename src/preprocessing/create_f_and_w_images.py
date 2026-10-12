import os
import nibabel as nib
import numpy as np


def create_f_and_w_images(target_image_name, input_dir, output_dir, suffix_in_phase="_in", suffix_out_phase="_opp"):
    """
    Create F (fat), W (water), and FF (fat fraction) images from Dixon in-phase and out-phase images.

    Parameters:
    -----------
    target_image_name : str
        Base name of the image (without extension or suffixes)
    input_dir : str
        Directory containing the input Dixon images
    output_dir : str
        Directory to save the F, W and FF images
    suffix_in_phase : str
        Suffix for in-phase image (default: "_in")
    suffix_out_phase : str
        Suffix for out-phase image (default: "_opp")
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct filenames for Dixon images
    in_phase_file = os.path.join(input_dir, f"{target_image_name}{suffix_in_phase}.nii.gz")
    out_phase_file = os.path.join(input_dir, f"{target_image_name}{suffix_out_phase}.nii.gz")

    # Load Dixon images
    in_phase_img = nib.load(in_phase_file)
    out_phase_img = nib.load(out_phase_file)

    in_phase_data = in_phase_img.get_fdata()
    out_phase_data = out_phase_img.get_fdata()

    # Calculate F (fat) and W (water) images
    # F = (In - Out) / 2
    # W = (In + Out) / 2
    f_data = (in_phase_data - out_phase_data) / 2.0
    w_data = (in_phase_data + out_phase_data) / 2.0

    # Calculate fat fraction image: FF = F / (F + W)
    ff_data = np.zeros_like(f_data, dtype=np.float32)
    denominator = f_data + w_data
    valid = np.abs(denominator) > 0
    ff_data[valid] = f_data[valid] / denominator[valid]
    ff_data = np.clip(ff_data, 0.0, 1.0)

    # Create NIfTI images with the same affine and header as input
    f_img = nib.Nifti1Image(f_data, in_phase_img.affine, in_phase_img.header)
    w_img = nib.Nifti1Image(w_data, in_phase_img.affine, in_phase_img.header)
    ff_img = nib.Nifti1Image(ff_data, in_phase_img.affine, in_phase_img.header)

    # Save F, W and FF images
    f_output_file = os.path.join(output_dir, f"{target_image_name}_F.nii.gz")
    w_output_file = os.path.join(output_dir, f"{target_image_name}_W.nii.gz")
    ff_output_file = os.path.join(output_dir, f"{target_image_name}_FF.nii.gz")

    nib.save(f_img, f_output_file)
    nib.save(w_img, w_output_file)
    nib.save(ff_img, ff_output_file)

    print(f"F image saved to: {f_output_file}")
    print(f"W image saved to: {w_output_file}")
    print(f"FF image saved to: {ff_output_file}")
    return f_output_file, w_output_file, ff_output_file


if __name__ == "__main__":
    # Example usage
    target_name = "pd_tse_dixon_tra_muscle_atrophy"
    input_directory = "C:/Users/mabel/OneDrive - NHS/Cases/Images for Martin - 22092026/Nifti/"
    output_directory = "C:/Users/mabel/OneDrive - NHS/Cases/Images for Martin - 22092026/Nifti/"
    
    create_f_and_w_images(target_name, input_directory, output_directory)
