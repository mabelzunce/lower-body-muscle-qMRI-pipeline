"""
Extract patient demographics (height and weight) from DICOM files.

This module processes DICOM files organized in subfolders by patient and extracts
height and weight information from any series of each patient.

Author: Medical Imaging Pipeline
Date: October 2025
"""

import os
import pandas as pd
import pydicom
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_patient_demographics_from_dicom(dicom_path: str, output_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Extract height and weight from DICOM files organized in patient subfolders.
    
    Args:
        dicom_path (str): Path to directory containing patient subfolders with DICOM files
        output_csv (str, optional): Path to save the results as CSV file
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Patient_ID', 'Height_cm', 'Weight_kg', 'Age', 'Sex', 'Series_Found']
    """
    
    if not os.path.exists(dicom_path):
        raise ValueError(f"Directory does not exist: {dicom_path}")
    
    results = []
    
    # Get all subdirectories (patient folders)
    patient_folders = [f for f in os.listdir(dicom_path) 
                      if os.path.isdir(os.path.join(dicom_path, f))]
    
    logger.info(f"Found {len(patient_folders)} patient folders in {dicom_path}")
    
    for patient_id in patient_folders:
        patient_folder = os.path.join(dicom_path, patient_id)
        logger.info(f"Processing patient: {patient_id}")
        
        # Extract demographics for this patient
        demographics = extract_demographics_from_patient_folder(patient_folder, patient_id)
        if demographics:
            results.append(demographics)
        else:
            logger.warning(f"No valid DICOM files found for patient: {patient_id}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to: {output_csv}")
    
    return df


def extract_demographics_from_patient_folder(patient_folder: str, patient_id: str) -> Optional[Dict]:
    """
    Extract demographics from any DICOM file in a patient folder.
    
    Args:
        patient_folder (str): Path to patient folder containing DICOM files
        patient_id (str): Patient identifier
        
    Returns:
        Dict: Dictionary containing patient demographics or None if no valid DICOM found
    """
    
    demographics = {
        'Patient_ID': patient_id,
        'Height_cm': None,
        'Weight_kg': None,
        'Age': None,
        'Sex': None,
        'Series_Found': 0,
        'Patient_Name': None,
        'Study_Date': None,
        'Modality': None
    }
    
    # Find all DICOM files recursively in the patient folder
    dicom_files = find_dicom_files(patient_folder)
    
    if not dicom_files:
        return None
    
    demographics['Series_Found'] = len(dicom_files)
    
    # Try to extract demographics from the first valid DICOM file
    for dicom_file in dicom_files:
        try:
            dcm = pydicom.dcmread(dicom_file, force=True)
            
            # Extract height (in meters, convert to cm)
            if hasattr(dcm, 'PatientSize') and dcm.PatientSize:
                height_m = float(dcm.PatientSize)
                demographics['Height_cm'] = height_m * 100  # Convert to cm
            
            # Extract weight (in kg)
            if hasattr(dcm, 'PatientWeight') and dcm.PatientWeight:
                demographics['Weight_kg'] = float(dcm.PatientWeight)
            
            # Extract age
            if hasattr(dcm, 'PatientAge') and dcm.PatientAge:
                # PatientAge is usually in format like '025Y' (25 years)
                age_str = dcm.PatientAge
                if 'Y' in age_str:
                    demographics['Age'] = int(age_str.replace('Y', ''))
            
            # Extract sex
            if hasattr(dcm, 'PatientSex') and dcm.PatientSex:
                demographics['Sex'] = dcm.PatientSex
            
            # Extract patient name
            if hasattr(dcm, 'PatientName') and dcm.PatientName:
                demographics['Patient_Name'] = str(dcm.PatientName)
            
            # Extract study date
            if hasattr(dcm, 'StudyDate') and dcm.StudyDate:
                demographics['Study_Date'] = dcm.StudyDate
            
            # Extract modality
            if hasattr(dcm, 'Modality') and dcm.Modality:
                demographics['Modality'] = dcm.Modality
            
            # If we found height and weight, we can stop
            if demographics['Height_cm'] and demographics['Weight_kg']:
                break
                
        except Exception as e:
            logger.warning(f"Error reading DICOM file {dicom_file}: {str(e)}")
            continue
    
    return demographics


def find_dicom_files(directory: str) -> List[str]:
    """
    Find all DICOM files in a directory recursively.
    
    Args:
        directory (str): Directory to search for DICOM files
        
    Returns:
        List[str]: List of paths to DICOM files
    """
    
    dicom_files = []
    dicom_extensions = ['.dcm', '.dicom', '.dic']
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check by extension first
            if any(file.lower().endswith(ext) for ext in dicom_extensions):
                dicom_files.append(file_path)
            else:
                # Check if file might be DICOM without extension
                try:
                    # Try to read as DICOM (quick check)
                    pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                    dicom_files.append(file_path)
                except:
                    continue
    
    return dicom_files


def print_demographics_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the extracted demographics.
    
    Args:
        df (pd.DataFrame): DataFrame containing patient demographics
    """
    
    print(f"\n=== DEMOGRAPHICS EXTRACTION SUMMARY ===")
    print(f"Total patients processed: {len(df)}")
    print(f"Patients with height data: {df['Height_cm'].notna().sum()}")
    print(f"Patients with weight data: {df['Weight_kg'].notna().sum()}")
    print(f"Patients with both height and weight: {(df['Height_cm'].notna() & df['Weight_kg'].notna()).sum()}")
    print(f"Patients with age data: {df['Age'].notna().sum()}")
    print(f"Patients with sex data: {df['Sex'].notna().sum()}")
    
    if df['Height_cm'].notna().any():
        print(f"\nHeight statistics (cm):")
        print(f"  Mean: {df['Height_cm'].mean():.1f}")
        print(f"  Std: {df['Height_cm'].std():.1f}")
        print(f"  Range: {df['Height_cm'].min():.1f} - {df['Height_cm'].max():.1f}")
    
    if df['Weight_kg'].notna().any():
        print(f"\nWeight statistics (kg):")
        print(f"  Mean: {df['Weight_kg'].mean():.1f}")
        print(f"  Std: {df['Weight_kg'].std():.1f}")
        print(f"  Range: {df['Weight_kg'].min():.1f} - {df['Weight_kg'].max():.1f}")
    
    print(f"\nSex distribution:")
    print(df['Sex'].value_counts())
    
    print(f"\nModality distribution:")
    print(df['Modality'].value_counts())


def main():
    """
    Example usage of the demographics extraction functionality.
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract patient demographics from DICOM files')
    parser.add_argument('dicom_path', help='Path to directory containing patient subfolders with DICOM files')
    parser.add_argument('--output', '-o', help='Output CSV file path', default=None)
    parser.add_argument('--summary', '-s', action='store_true', help='Print summary statistics')
    
    args = parser.parse_args()
    
    # Extract demographics
    df = extract_patient_demographics_from_dicom(args.dicom_path, args.output)
    
    # Print summary if requested
    if args.summary:
        print_demographics_summary(df)
    
    print(f"\nExtracted demographics for {len(df)} patients")
    print("\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()
