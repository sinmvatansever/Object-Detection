==============================================================================
Engineering Programming Project 2: Object Detection with Homography & Affine
==============================================================================

GROUP MEMBERS:
1. Nandani Ajay Gupta
2. Samra Iqbal
3. Sinem Vatansever
4. Yeaba Shamsher Astha

------------------------------------------------------------------------------
1. PROJECT OVERVIEW
------------------------------------------------------------------------------
This project implements a feature-based object detection system to locate 
four specific objects (Vitamin C, Controller, Artclass Palette, Airpods) 
within cluttered query images. 

We have provided two implementations to demonstrate robust analysis:
  1. Homography Code: Strictly follows the project requirements 
     using the Homography assumption (8 Degrees of Freedom).
  2. Affine Code: An alternative implementation using 
     Affine Transformation (6 Degrees of Freedom) which proved more robust 
     for the 3D-curved Controller in Query 3.

------------------------------------------------------------------------------
2. DIRECTORY STRUCTURE
------------------------------------------------------------------------------
Ensure your project folder looks like this:

Project_Folder/
│
├── dataset/                   # Folder containing all images
│   ├── img_airpod.jpg
│   ├── img_pallette.jpg
│   ├── img_controller.jpg
│   ├── img_vitaminc.jpg
│   ├── img_query_1.jpg
│   ├── img_query_2.jpg
│   └── img_query_3.jpg
│
├── detection_results/         # Output folder (auto-created by the script)
│   ├── match_subplots_filtered
│   ├── results_affine        # results from affine parameter
│   ├── results_homography    # results from homography parameter
│
├── environment.yml            # Environment configuration file
├── detector_homography_code.py # Main submission code
├── detector_affine_code.py     # Comparative analysis code
└── README.txt                 # This file

------------------------------------------------------------------------------
3. INSTALLATION INSTRUCTIONS
------------------------------------------------------------------------------
To run the code, please create a virtual environment using the provided
YAML file. Open your terminal or Anaconda Prompt and run:

1. Navigate to the project folder:
   cd "path\to\your\folder"

2. Create the environment:
   conda env create -f environment.yml

3. Activate the environment:
   conda activate cv_project

------------------------------------------------------------------------------
4. HOW TO RUN THE CODE
------------------------------------------------------------------------------

--- Option A: Homography ---
This script uses cv2.findHomography() with RANSAC. It is the primary 
deliverable for the assignment.

   Run command:
   python detector_homography_code.py

   Expected Output:
   - Processes all 3 query images.
   - Saves result images with bounding boxes to 'detection_results/'.
   - Saves feature match details to 'detection_results/match_subplots_filtered/'.


--- Option B: Affine ---
This script uses cv2.estimateAffinePartial2D(). We included this to show
how a stricter geometric model improves stability for 3D objects like
the Controller.

   Run command:
   python detector_affine_code.py

------------------------------------------------------------------------------
5. KEY LIBRARIES USED
------------------------------------------------------------------------------
* OpenCV (cv2): SIFT feature extraction, FLANN matching, Homography/Affine.
* NumPy: Array manipulations and coordinate calculations.
* Matplotlib: Visualizing the bounding boxes and matches.