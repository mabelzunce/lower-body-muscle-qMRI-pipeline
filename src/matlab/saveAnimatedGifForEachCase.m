clear all
close all
addpath('./encoder/')
addpath('./ReadData3D_version1k/mha/')
%% RESULTS TO PROCESS
enablePostprocessing = 0;
%% DATA PATH
dixonSubFolders = ''; 
% Segmented images:
automatedSegmentionPath = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx/nifti_pelvis/';
outputPath = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx/nifti_pelvis/';

automatedSegmentionPath = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx/nifti_lumbar/';
outputPath = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx/nifti_lumbar/';
%automatedSegmentionPath = '/home/martin/data_imaging/Muscle/data_sherpas/ProcessedCropped/';
%outputPath = '/home/martin/data_imaging/Muscle/data_sherpas/ProcessedCropped/';
%% PARAMETERS
labelsImageSuffix = ['_segmentation'];

numLabels = 8;
inPhaseSuffix = '_I_biasFieldCorrection';
fatFractionPhaseSuffix = '_ff';
tissueSegmentedSuffix = '_tissue_segmented';
skinFatSuffix = '_skin_fat_2';
muscleMaskSuffix = '_muscle';
outputFilenameSuffix = 'segmentation_check';
extension = '.nii.gz';% '.nii.gz';%'.mhd';
%% PARAMETERS TO PROCESS THE DIXON IMAGES
tags = {'I','O','W','F'};
%labels = 0:4; % 
label_air=0; label_fat=3; label_water=1; label_fat_water=2; label_bone = 4; label_unknown = 5; 
threshold = 2;
background = 80;
%% GET THE FILENAMES FOR EACH SEGMENTED IMAGE FOR THE AUTOMATED SEGMENTED CASES AND THE IN PHASE DIXON
j = 1;
automatedFF_percentaje = [];
dirNames = dir(automatedSegmentionPath);
for i = 3 : numel(dirNames)
    subjectName = dirNames(i).name;
    % Segmented image:
    filename = fullfile(automatedSegmentionPath, subjectName, [subjectName labelsImageSuffix  extension]);
    % In phase, tissue segmented and fat fraction:
    inPhaseFilename = fullfile(automatedSegmentionPath, subjectName, [subjectName inPhaseSuffix  extension]);
    fatFractionFilename = fullfile(automatedSegmentionPath, subjectName, [subjectName fatFractionPhaseSuffix  extension]);
    %tissueSegmentedFilename = [dixonDataPath '\' dirNames(i).name '\' dixonSubFolders dirNames(i).name tissueSegmentedSuffix '.mhd'];
    %muscleMaskFilename = [dixonDataPath '\' dirNames(i).name '\' dixonSubFolders dirNames(i).name muscleMaskSuffix '.mhd'];
      
    if exist(filename) &&  exist(inPhaseFilename)
        caseNames{j} = dirNames(i).name;   
        segmentedImageFilenames{j} = filename;
        inPhaseFilenames{j} = inPhaseFilename;
        fatFractionFilenames{j} = fatFractionFilename;
        %tissueSegmentedFilenames{j} = tissueSegmentedFilename;
        %muscleMaskFilenames{j} = muscleMaskFilename;
        j = j + 1;  
    end  
end

%% GO THROUGH EACH ELEMENT IN THE LIBRARY AND GENERATE AN ANIMATED GIF
colorsToMatchOtherFigures = [ 51 136 93; 22 64 149;  149 22 22;  107 150 65; 148 148 20; 22 149 149; 21 21 106; 14 14 141; 255, 229, 204; 255, 229, 204; 0 0 0; 0 0 0; 0 0 0; 0 0 0; 0 0 0]./255;
strelCube = strel('cube',3);
strelSphere = strel('sphere',1);
strelBall = offsetstrel('ball',1,1,0);
fwhm_voxels = 3;
stdDevGauss_voxels = fwhm_voxels ./ 2.355;
for i = 1 : numel(segmentedImageFilenames)
    % Read dixon images:        
    if strcmp(extension, '.mhd')
        try
            [dixonImages, info] = ReadMhdImage(inPhaseFilenames{i});
        catch
            info = mha_read_header(inPhaseFilenames{i});
            dixonImages = single(mha_read_volume(inPhaseFilenames{i}));
        end
        
        % Read segmented image:
        [segmentedImage, infoSeg] = ReadMhdImage(segmentedImageFilenames{i});
    else
        info = niftiinfo(inPhaseFilenames{i});
        dixonImages = niftiread(inPhaseFilenames{i});
        % Read segmented image:
        segmentedImage = niftiread(segmentedImageFilenames{i});
    end
    
    % Read muscle mask image amd apply it if said so:
    %[muscleMask, referenceInfoMask] = ReadMhdImage(muscleMaskFilenames{i});
    segmentedImage = single(segmentedImage);
    %if applyMuscleMask
    %    segmentedImage = segmentedImage .* single(muscleMask);
    %end
    if enablePostprocessing
        for k = 1 : numLabels
            % Process the image as in the plugin:
            maskLabel = segmentedImage == k;
            maskLabel = imgaussfilt3(single(maskLabel), stdDevGauss_voxels) > 0.5;
            %maskLabel = imerode(maskLabel, strelSphere);
            %maskLabel = bwareafilt(maskLabel,1); % this is only 2d
            CC = bwconncomp(maskLabel, 6);
            if CC.NumObjects > 0
                numPixels = cellfun(@numel,CC.PixelIdxList);
                [~,idx] = max(numPixels);
                maskLabel = false(size(maskLabel));
                maskLabel(CC.PixelIdxList{idx}) = true;
                maskLabel = imdilate(maskLabel, strelSphere);
                % Fill cavities:
                maskLabel = imfill(maskLabel,'holes');
                % Get the fat fr
            end
            maskLabel = imgaussfilt3(single(maskLabel), stdDevGauss_voxels) > 0.5;
            segmentedImage(maskLabel) = k;  
        end
    end
    % reshape both images:
    dixonImages = permute(dixonImages, [2 1 3]); % Rows=y
    segmentedImage = permute(segmentedImage, [2 1 3]); % Rows=y
    if ~strcmp(extension, '.mhd')
        dixonImages = flip(dixonImages, 1);
        segmentedImage = flip(segmentedImage, 1);
    end
    % Rescale the image:
    dixonImages = dixonImages./max(max(max(dixonImages)))/0.7;
    sizeThisImage = size(dixonImages);
    % Output filename:
    filename = fullfile(outputPath, caseNames{i}, [caseNames{i} outputFilenameSuffix '.gif']);

    figure;
    k = 1;
    for j = 1 : size(dixonImages, 3)
        % Go through each slice and generate a fused image:
        inPhaseSlice = dixonImages(:,:,j);
        segmentedCaseSlice = segmentedImage(:,:,j);
        maskSlices = single((segmentedCaseSlice >0) & (segmentedCaseSlice<(numLabels+1)));
        outlineMuscleLabelsForCase = segmentedCaseSlice.*maskSlices-imerode(segmentedCaseSlice.*maskSlices, strel('disk',5));
        if max(max(outlineMuscleLabelsForCase)) > 0
            imageWithMask = labeloverlay(inPhaseSlice, outlineMuscleLabelsForCase, 'Transparency',0, 'IncludedLabels', [1:max(outlineMuscleLabelsForCase(:))], 'Colormap', colorsToMatchOtherFigures);
            imshow(imageWithMask)
            [imind,cm] = rgb2ind(imageWithMask,256); 
            % Write to the GIF File 
            if k == 1 
              imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
            else 
              imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',0.1);
            end
            k = k + 1;
        end
    end
    close all
end
