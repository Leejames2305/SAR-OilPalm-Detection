Current pipeline:
Calibration -> ML -> Speckle -> Terrain Correction

Output:
New vector around the datasets trees -> Mask -> Output subsets

Auth GCS: \
User needs to create a service account with `Object Viewer` permission under that project. Then, generate a JSON key from it.

Before running the GCS cells, user need to manually add the auth file by doing commands below:
> cat >/content/SABucketRead.json << 'EOF'\
> {paste the key JSON here}\
> EOF