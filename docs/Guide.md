Current pipeline:
Calibration -> Speckle -> Terrain Correction

Output:
New vector around the datasets trees -> Mask -> Output subsets

Auth GCS: \
User needs to create a service account with `Object Viewer` permission under that project. Then, generate a JSON key from it.

User need to manually upload the SA auth file to auth for both GEE and GCS download. 
