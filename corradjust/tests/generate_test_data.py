import numpy as np
import pandas as pd
import os


"""
There are 1,000 features in total.
Features 0-99, 100-199, 200-299, 300-399, 400-499 have within-group correlations
of 0.5, 0.6, 0.7, 0.8, and 0.9, respectively.
These are "real" correlations as these groups form
5 reference feature sets according to the GMT file.

Features 500-599, 600-699, 700-799 have within-group correlations
of 0.55, 0.85, and 0.95, respectively.
These are "noise" correlations, as there are no reference
sets linking these features.

With these correlations and equal group sizes, real factors
are represented by PCs 2, 4, 5, 6, 8, and noise factors
are represented by PCs 1, 3, 7
"""

np.random.seed(17)

n_samples = 1000
n_features = 1000

# Initialize with random noise
X = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))

# First, add the "real" correlations
n_real_factors = 5
real_factors = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_real_factors))
real_target_corrs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
real_factors *= np.sqrt(real_target_corrs / (1 - real_target_corrs))

real_coefs = np.zeros((n_real_factors, n_features))
for k in range(n_real_factors):
    real_coefs[k, 100 * k:100 * (k + 1)] = 1.0

X += real_factors @ real_coefs

n_noise_factors = 3
noise_factors = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_noise_factors))
noise_target_corrs = np.array([0.55, 0.85, 0.95])
noise_factors *= np.sqrt(noise_target_corrs / (1 - noise_target_corrs))

noise_coefs = np.zeros((n_noise_factors, n_features))
for k in range(n_noise_factors):
    noise_coefs[k, 100 * (n_real_factors + k):100 * (n_real_factors + k + 1)] = 1.0

X += noise_factors @ noise_coefs

# Make input data frames for CorrAdjust
df_data = pd.DataFrame(X)
df_data.index = "Sample" + df_data.index.astype(str)
df_data.columns = "Feature" + df_data.columns.astype(str)
df_data_train = df_data.iloc[::2]
df_data_test = df_data.iloc[1::2]
df_feature_ann = pd.DataFrame(
    {"feature_name": df_data.columns, "feature_type": "mRNA"},
    index=df_data.columns
)

# Save the tables
script_dir = os.path.dirname(os.path.abspath(__file__))
df_data_train.to_csv(f"{script_dir}/test_data/df_data_train.tsv", sep="\t")
df_data_test.to_csv(f"{script_dir}/test_data/df_data_test.tsv", sep="\t")
df_feature_ann.to_csv(f"{script_dir}/test_data/df_feature_ann.tsv", sep="\t")

# Make GMT file
gmt_file = open(f"{script_dir}/test_data/ref_feature_sets.gmt", "w")

# Real feature sets
for k in range(n_real_factors):
    feature_names = "\t".join(df_feature_ann.iloc[100 * k:100 * (k + 1)]["feature_name"].tolist())
    print(f"Real_set{k}\tNA\t{feature_names}", file=gmt_file)

# Pair up features 500-999 to unrelated ones, so that they are not ignored by CorrAdjust
for i in range(500, 1000):
    feature_names = f"Feature{i}\tFeature{i-100}\tFeature{i-199}\tFeature{i-298}\tFeature{i-397}\tFeature{i-496}"
    print(f"Technical_set{i}\tNA\t{feature_names}", file=gmt_file)

gmt_file.close()