## Project Introduction
This postgraduate project is **Oil Palm Disease Classification project using SAR (Synthetic Aperture Radar) imagery.**

The goal is to classify disease status in oil palm trees - distinguishing `Healthy` from `Unhealthy` - using values derived ALOS PALSAR-2 L-band data. This includes (not limited to) 4-Bands Backscatter intensity (VV, VH, HH, HV), Yamaguchi | H-Alpha decomposition, Vegetation Index, and much more. 

Key challenges include small-to-medium dataset size (samples in the thousands) and expected severe class imbalance between infected and non-infected trees.

The target is not to get perfect accuracy (or positive rate) in detecting Unhealthy oil palm tree, instead it is to detect majority of the Unhealthy tree/area, while keeping false positive rate (Healthy  being labeled as Unhealthy) reasonable. The results from this project serves as a guide for the downstream (An Polarized Imagery-based, on-site automated detection system ; Out of current scope) that does the final classification.


## Project Structure
Due to limitation in local compute power, compute-intensive notebooks (Extracting SAR data, ML modeling, ...) are all designed to run on Google Colab.  

Key things about current project structure:
- `./data`: Contains clean datasets that is either freshly extracted, or filtered
- `./docs`: Contains status logs, POC findings, and planned actions
- `./misc/notebook`: Contains drafts' notebook, mostly for POC purpose
- `./misc/POC_Results`: Contains results/artifacts collected from POC runs (local & Colab)
- `./misc/scripts`: Useful tools/scripts that helps process datasets
- `*.ipynb`: Main notebook that will only be updated when POC shows promising direction

You are highly encouraged to read the `./docs/AgentPlan & POCReport` to understand what has been tried and findings from it. 

For any quick, local Python analysis, use `.venv` setup in this project. Update/Modify it if missing any essential packages needed. 