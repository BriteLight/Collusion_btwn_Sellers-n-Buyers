### Step 1: BQ Data loading


```python
import mlutils
from mlutils import dataset
from mlutils import connector
import pandas as pd
from datetime import datetime
from os import listdir
from os.path import isfile, join
import smtplib
from string import Template
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase 
from email import encoders
from datetime import timedelta
import numpy as np
import gc
import csv
import math
import h2o
import xgboost as xgb
from pickle import load
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import shap
```


```python
#Assigning key for run
key = 'jul18'
```


```python
bq_df1 = dataset.load(name="BigQuery", query="SELECT * FROM `<gcp-proj-id>.<bq-dataset>.bsc_mvp_final_vars_jul18_selected`")
bq_df1.to_csv("Data/bsc_mvp_final_vars"+key+"_selected.csv", index = False)
```

#### Step 2: Initialize pre defined functions

**Score scaling function**
Scales score to a range of 0-1000. Values beyond 1000 are capped at 1000 and values below 0 are capped at 0
Method utilizes credit score techniques leveraging Points to Double Odds as well as different offsets for each class as seen below


```python
def scaling_median_risk(x):
    score = round(800 + (75*np.log(2))*np.log(x/(1-x)))
    if score < 1:
        return 1
    elif score>1000:
        return 1000
    else:
        return score

def scaling_cb_risk(x):
    score = round(600 + (60*np.log(2))*np.log(x/(1-x)))
    if score < 1:
        return 1
    elif score > 1000:
        return 1000
    else:
        return score
    
def scaling_refund_risk(x):
    score = round(600 + (52*np.log(2))*np.log(x/(1-x)))
    if score < 1:
        return 1
    elif score > 1000:
        return 1000
    else:
        return score
```

#### Step 3: Load dataset, model, scaler, shap explainer


```python
#Load dataset
df = pd.read_csv("Data/bsc_mvp_final_vars_jul18_selected.csv")
```


```python
#Load scaler pickle
scaler = load(open('Pipeline/Files/scaler.pkl', 'rb'))
```


```python
#Load SHAP explainer pickle
explainer = load(open('Pipeline/Files/shap_explainer.pkl', 'rb'))
```


```python
#Load model using load_model to avoid pickle version dependency issue
model = xgb.XGBClassifier()
model.load_model("Pipeline/Files/xgb_model_57vars.json")
```

#### Step 4: Preprocess data, Predict results


```python
df_0 = df.drop(columns = ['partner_id'])
```


```python
#Transform data
scaled_vals = scaler.transform(df_0)
df_scaled = pd.DataFrame(scaled_vals,columns = df_0.columns)
```


```python
#Predict output
pred = model.predict_proba(df_scaled)
```

#### Step 5: Generate reason codes, scale final output


```python
#Generate shap values based on trained explainer 
shap_values_df = explainer.shap_values(df_scaled)
```

    ntree_limit is deprecated, use `iteration_range` or model slicing instead.



```python
#Create dataframe with probability and partner_id
df_gen = pd.DataFrame(data = pred,columns = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad'])
df_gen.drop(columns = ['is_good'],inplace = True)
df_gen['partner_id'] = df['partner_id']
```


```python
#Selected reason codes to identify collusion
reason_code_vals = ["distinct_ato_vulnerable_cust_cnt_rate_180d",
"distinct_ato_vulnerable_cust_cnt_rate_30d",
"distinct_dormant_cust_cnt_rate_180d",
"distinct_inactive_cust_cnt_rate",
"distinct_inactive_cust_cnt_rate_180d",
"distinct_new_active_cust_cnt_rate",
"distinct_sf_like_cust_cnt_rate_180d",
"distinct_sf_like_cust_cnt_rate_30d",
"distinct_synthetic_bot_cust_cnt_rate_180d",
"distinct_young_cust_cnt_rate_180d",
"first_time_buyer_rate",
"max_ipregion_sale_contribution_180d",
"max_ipzip_sale_contribution_180d",
"max_product_sale_contribution_180d",
"max_product_sale_contribution_30d",
"max_shipcity_sale_contribution_180d",
"max_shipcity_sale_contribution_30d",
"max_shipstate_sale_contribution_180d",
"max_shipzip_sale_180d",
"mismatch_rate_30D",
"mismatch_rate_90D",
"Reuse_rate_60D",
"Reuse_rate_90D",
"sales_from_ato_vulnerable_contribution_180d",
"sales_from_ato_vulnerable_contribution_30d",
"sales_from_dormant_cust_contribution",
"sales_from_dormant_cust_contribution_180d",
"sales_from_inactive_cust_contribution",
"sales_from_inactive_cust_contribution_180d",
"sales_from_new_active_cust_contribution",
"sales_from_new_active_cust_contribution_180d",
"sales_from_new_cust_contribution",
"sales_from_new_cust_contribution_180d",
"sales_from_sf_contribution",
"sales_from_sf_contribution_180d",
"sales_from_sf_contribution_30d",
"sales_from_synthetic_bot_account",
"sales_from_synthetic_bot_account_contribution_180d",
"sales_from_young_cust_contribution_180d",
"seller_tof",
"SlrRspns_14D",
"SlrRspns_90D",
"total_sellers_connected_device",
"total_suspended_sellers_connected_device",
"total_terminated_sellers_connected_device",
"VTR_14",
"VTR_90",
"sam_seller",
'Fraud_Cancel_Rate_14D','Fraud_Cancel_Rate_30D']
```


```python
#Code to generate reason code for every class
reason_code_cbk = []
reason_code_med = []
reason_code_rfnd = []
for idx, row in df_gen.iterrows():
    shap_row1 = pd.DataFrame(shap.Explanation(values=shap_values_df[1][idx],
                                             base_values=explainer.expected_value[0], data=df_scaled.iloc[idx],
                                             feature_names=df_scaled.columns.tolist()).data).reset_index()
    
    shap_row2 = pd.DataFrame(shap.Explanation(values=shap_values_df[2][idx],
                                             base_values=explainer.expected_value[0], data=df_scaled.iloc[idx],
                                             feature_names=df_scaled.columns.tolist()).data).reset_index()
    
    shap_row3 = pd.DataFrame(shap.Explanation(values=shap_values_df[3][idx],
                                             base_values=explainer.expected_value[0], data=df_scaled.iloc[idx],
                                             feature_names=df_scaled.columns.tolist()).data).reset_index()
    
    shap_row1.sort_values(by = idx, ascending = False, inplace = True)
    shap_row2.sort_values(by = idx, ascending = False, inplace = True)
    shap_row3.sort_values(by = idx, ascending = False, inplace = True)
    
    reason_code1 = ';'.join(shap_row1[(shap_row1['index'].isin(reason_code_vals)) & (shap_row1[idx] > 0)][:5]['index'].values)
    reason_code2 = ';'.join(shap_row2[(shap_row2['index'].isin(reason_code_vals)) & (shap_row2[idx] > 0)][:5]['index'].values)
    reason_code3 = ';'.join(shap_row3[(shap_row3['index'].isin(reason_code_vals)) & (shap_row3[idx] > 0)][:5]['index'].values)
    reason_code_med.append(reason_code1)
    reason_code_rfnd.append(reason_code2)
    reason_code_cbk.append(reason_code3)
    
df_gen['top_reason_code_class_median'] = reason_code_med
df_gen['top_reason_code_class_rfnd'] = reason_code_rfnd
df_gen['top_reason_code_class_cbk'] = reason_code_cbk
```


```python
#Apply scoring to probabilities
df_gen['is_median_cb_bad_and_refund_bad_scaled_score'] = df_gen['is_median_cb_bad_and_refund_bad'].apply(scaling_median_risk)
df_gen['is_severe_cb_bad_scaled_score'] = df_gen['is_severe_cb_bad'].apply(scaling_cb_risk)
df_gen['is_severe_refund_bad_scaled_score'] = df_gen['is_severe_refund_bad'].apply(scaling_refund_risk)
```


```python
col_vals = ['partner_id','is_median_cb_bad_and_refund_bad_scaled_score','is_severe_cb_bad_scaled_score', 
            'is_severe_refund_bad_scaled_score','top_reason_code_class_median','top_reason_code_class_rfnd', 
            'top_reason_code_class_cbk']
```


```python
#Final Output 
df_share = df_gen[col_vals]
```


```python
#Output to be shared with the team
df_share.to_csv("Pipeline/Output/bsc_output_"+key+".csv",index = False)
```
