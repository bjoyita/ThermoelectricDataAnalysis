The code "FeatureImportance.py" uses pymatgen, matminer, scikit-learn and shap libraries to rank the descriptors/predictors of 573 inorganic thermoelectric materials listed in the csv file PF_ZT.csv. This dataset is a subset of the original dataset used in https://github.com/ngs00/DopNet/tree/main/dataset. To run the code, one needs to install two open-source python-based materials packages - pymatgen and matminer.

To install pymatgen and matminer, you will require python > 3.7 and pip. Run the following to install pytmagen and matminer: pip install pymatgen; pip install matminer (follow the sequene)

The code uses "composition" featurizers from pymatgen and matminer. To learn more about the libraries, visit https://hackingmaterials.lbl.gov/matminer, https://pymatgen.org

The code extracts elements and their atomic fractions from the compound formula given in the dataset. Here I show the use of random forest algorithm to rank the features impacting the thermoelectric efficiency (ZT)

The code uses scikit-learn to find the correlation matrix for different features. It also uses shap library to explain the output of the model.




