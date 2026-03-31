# Collusion_btwn_Sellers-n-Buyers
Demo illustration of model for scoring of collusion between Buyer &amp; Seller

 [ Loading-n-Shaping_Collusion_btwn_Sellers-n-Buyers ]
# CREATE GCP PROJECT AND BIGQUERY DATASET
Load BQ table data for processing
Scales score to a range of 0-1000. Values beyond 1000 are capped at 1000 and values below 0 are capped at 0

XBG Classifier, mem buffer of data in Pickle
Shap_Explainer with Pickle

Preprocess -n- Predict

# Generate reason codes

# Scale output before final pipeline


[ Collusion_btwn_Sellers-n-Buyers ]
# Combine preprocessing steps

# Create dataframes with preprocessing done

# Separate features and labels
# Split data 75%-25% into training set and test set
# XGBoost model - declaring and fitting xgb classifier

# Prediction
# is_good explanability
# is_median_cb_bad_and_refund_bad explanability
# is_severe_refund explanability
# is_severe_cbk_bad explanability

# gains_chart
# compute SHAP values
# Code for reason code for every class
# Create dataframes with preprocessing done
[ END ]


