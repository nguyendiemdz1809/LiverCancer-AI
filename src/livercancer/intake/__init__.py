"""Input intake for the website (separate from the locked AI pipeline):
image2d   - 2D images / screenshots: find the medical slice, remove viewer UI and burned-in text, frame like training
dicom_io  - DICOM files / series -> uint8 display slices (CT window from the source paper, MRI 99.5th percentile)
modality  - CT vs MRI recognition: DICOM tag > NIfTI header geometry > image classifier (separate, Val-evaluated model)
synthetic - synthetic screenshots built from dataset slices, ONLY for training / evaluating the modality classifier
            and testing the screenshot pre-processing (never shown as patient data, never used to score the AI pipeline)
"""
