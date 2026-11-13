# Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

Official page: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset  
DOI: https://doi.org/10.24432/C5HS5C  
License: CC BY 4.0

The training script retrieves the dataset using `ucimlrepo` if `data/ai4i2020.csv` is not present. The downloaded dataset is intentionally excluded from Git by `.gitignore`; users should retrieve it from the official source.

Citation:
Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C.

The dataset is synthetic and contains 10,000 rows. The model inputs used here are `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, and `Tool wear [min]`. The target is `Machine failure`. ID columns and the component failure-mode target columns are not predictors.
