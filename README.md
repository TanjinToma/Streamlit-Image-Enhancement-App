# Stremlit-Image-Enhancement-App
This is a python-based GUI for Microscopy image enhancement developed in the [streamlit framework](https://www.streamlit.io/), which is 
an open-source app framework for Machine Learning and Data Science. The app contains two different algorithms for microscopy image enhancement-(1) CAEFI [1] enhancement algorithm 
for the enhancement of images with filamentous structures (e.g., Neurons), (2) VBET [2] algorithm to enhance images with joint Blob-and-Vessel like 
structures (e.g., Microglia cells, Astrocytes).

To use this image enhancement app, user requires to install Streamlit following the documentation from: https://docs.streamlit.io/en/stable/. With installation completed, run the 
following command to start the app:

```
streamlit run Image_Enhancement_App_main.py
```
## Citation
#### [1] H. Jeelani, H. Liang, S. T. Acton and D. S. Weller, "Content-Aware Enhancement of Images With Filamentous Structures," in IEEE Transactions on Image Processing, vol. 28, no. 7, pp. 3451-3461, July 2019, doi: 10.1109/TIP.2019.2897289.
#### [2] T. T. Toma, K. Bisht, U. Eyo and D. S. Weller, "VBET: VESSELNESS AND BLOB ENHANCEMENT TECHNIQUE FOR 2D AND 3D MICROSCOPY IMAGES OF MICROGLIA," 2020 54th Asilomar Conference on Signals, Systems, and Computers, Pacific Grove, CA, 2020. (to be appeared on IEEE Xplore)
