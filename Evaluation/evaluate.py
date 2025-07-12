import numpy as np
from scipy.stats import t

# Example data with multiple rows (each row: [ground truth mean, ground truth sd, sample mean, sample sd, ...])
data = np.array([
    [6.29, 2.38, 6.61, 2.23, 5.99, 1.81, 6.21, 2.03, 6.33,	2.2,	6.2,	1.83,	6.1,	1.94,	6.01,	1.82,	6.48,	2.54],  # row 1
    [6.31, 1.98, 6.41, 2.0, 6.3, 1.57, 6.03, 1.79, 6.08,	1.59,	5.77,	1.76,	5.84,	1.9,	6.26,	2.37, 6.32, 1.88]      # row 2
])

n_gt = 10  # sample size of ground truth 
n_sample = 10  # sample size of each compared sample 

p_values_all = []

for row in data:
    gt_mean, gt_sd = row[0], row[1]
    sample_means = row[2::2]
    sample_sds = row[3::2]

    p_values_row = []
    for mean, sd in zip(sample_means, sample_sds):
        # Compute Welch's t
        numerator = gt_mean - mean
        denominator = np.sqrt((gt_sd**2 / n_gt) + (sd**2 / n_sample))
        t_score = numerator / denominator

        # Degrees of freedom (Welch-Satterthwaite)
        num_df = (gt_sd**2 / n_gt + sd**2 / n_sample)**2
        den_df = ((gt_sd**2 / n_gt)**2) / (n_gt - 1) + ((sd**2 / n_sample)**2) / (n_sample - 1)
        df = num_df / den_df

        # Two-tailed p-value
        p_val = 2 * t.sf(np.abs(t_score), df)
        p_values_row.append(p_val)

    p_values_all.append(p_values_row)

# Convert to NumPy array for better presentation
p_values_all = np.array(p_values_all)

print("P-values for each row compared to ground truth:")
print(np.round(p_values_all,3))