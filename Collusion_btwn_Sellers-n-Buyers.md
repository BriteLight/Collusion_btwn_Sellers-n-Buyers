```python
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from matplotlib import pyplot as plt
%matplotlib inline
import sklearn
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error,roc_auc_score,precision_score
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets, ensemble, model_selection
import shap


top_cols = [
'cust_cnt',
'distinct_ato_vulnerable_cust_cnt_rate_180d',
'distinct_ato_vulnerable_cust_cnt_rate_30d',
'distinct_dormant_cust_cnt_rate_180d',
'distinct_inactive_cust_cnt_rate',
'distinct_inactive_cust_cnt_rate_180d',
'distinct_new_active_cust_cnt_rate',
'distinct_sf_like_cust_cnt_rate_180d',
'distinct_sf_like_cust_cnt_rate_30d',
'distinct_synthetic_bot_cust_cnt_rate_180d',
'distinct_young_cust_cnt_rate_180d',
'first_time_buyers',
'international_seller',
'max_ipregion_sale_contribution_180d',
'max_ipzip_sale_contribution_180d',
'max_product_sale_contribution_180d',
'max_product_sale_contribution_30d',
'max_shipcity_sale_contribution_180d',
'max_shipcity_sale_contribution_30d',
'max_shipstate_sale_contribution_180d',
'max_shipzip_sale_180d',
'mismatch_rate_30D',
'mismatch_rate_90D',
'Reuse_rate_60D',
'Reuse_rate_90D',
'sales_from_ato_vulnerable_contribution_180d',
'sales_from_ato_vulnerable_contribution_30d',
'sales_from_dormant_cust_contribution',
'sales_from_dormant_cust_contribution_180d',
'sales_from_inactive_cust_contribution',
'sales_from_inactive_cust_contribution_180d',
'sales_from_new_active_cust_contribution',
'sales_from_new_active_cust_contribution_180d',
'sales_from_new_cust_contribution',
'sales_from_new_cust_contribution_180d',
'sales_from_sf_contribution',
'sales_from_sf_contribution_180d',
'sales_from_sf_contribution_30d',
'sales_from_synthetic_bot_account',
'sales_from_synthetic_bot_account_contribution_180d',
'sales_from_young_cust_contribution_180d',
'seller_amt_30d',
'seller_amt_7d',
'seller_cnt_14d',
'seller_tof',
'SlrRspns_14D',
'SlrRspns_90D',
'total_buyers',
'total_sales',
'total_sellers_connected_device',
'total_suspended_sellers_connected_device',
'total_terminated_sellers_connected_device',
'VTR_14',
'VTR_90',
'sam_seller',
'Fraud_Cancel_Rate_14D','Fraud_Cancel_Rate_30D',
'sim_date',
'PRTNR_SRC_ORG_CD',
'seller_status', 'label']


bsc_all_train = pd.read_csv("Data/bsc_train_correct.csv", usecols = top_cols)
bsc_all_oot = pd.read_csv("Data/bsc_oot_correct.csv", usecols = top_cols)


bsc_all_train = bsc_all_train[bsc_all_train['label'] != 0]
bsc_all_oot = bsc_all_oot[bsc_all_oot['label'] != 0]
bsc_all_train['label_new'] = 0
bsc_all_train.loc[bsc_all_train['label']==1,'label_new'] = 0
bsc_all_train.loc[bsc_all_train['label']==2,'label_new'] = 1
bsc_all_train.loc[bsc_all_train['label']==3,'label_new'] = 2
bsc_all_train.loc[bsc_all_train['label'] == 4,'label_new']= 3
bsc_all_oot['label_new'] = 0
bsc_all_oot.loc[bsc_all_oot['label']==1,'label_new'] = 0
bsc_all_oot.loc[bsc_all_oot['label']==2,'label_new'] = 1
bsc_all_oot.loc[bsc_all_oot['label']==3,'label_new'] = 2
bsc_all_oot.loc[bsc_all_oot['label'] == 4,'label_new']= 3


bsc_all_train['label'] = bsc_all_train['label_new']
bsc_all_oot['label'] = bsc_all_oot['label_new']


df_train = bsc_all_train[top_cols].copy()
df_oot = bsc_all_oot[top_cols].copy()


df_train.to_csv('final_bsc_all_train_correct.csv', index = False)
df_oot.to_csv('final_bsc_all_oot_correct.csv', index = False)


display(df_train.shape)
display(df_oot.shape)


df_oot_onedate = df_oot[df_oot['sim_date'] == 'feb23']


df_train['first_time_buyer_rate'] = df_train['first_time_buyers']/df_train['total_buyers']
df_oot['first_time_buyer_rate'] = df_oot['first_time_buyers']/df_oot['total_buyers']
df_oot_onedate['first_time_buyer_rate'] = df_oot_onedate['first_time_buyers']/df_oot_onedate['total_buyers']

df_train.drop(columns = ['first_time_buyers','sim_date'], inplace = True)
df_oot.drop(columns = ['first_time_buyers','sim_date'], inplace = True)
df_oot_onedate.drop(columns = ['first_time_buyers','sim_date'], inplace = True)


top_cols = top_cols + ['first_time_buyer_rate']
top_cols.remove('first_time_buyers')
top_cols.remove('sim_date')


df_train_x = df_train[top_cols].drop(columns = ['label','PRTNR_SRC_ORG_CD','seller_status'])
df_oot_x = df_oot[top_cols].drop(columns = ['label','PRTNR_SRC_ORG_CD','seller_status'])
df_oot_onedate_x = df_oot_onedate[top_cols].drop(columns = ['label','PRTNR_SRC_ORG_CD','seller_status'])

y_train_df = df_train['label']
y_oot_df = df_oot['label']
y_oot_onedate_df = df_oot_onedate['label']

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_features = list(df_train_x.select_dtypes(include=numerics).columns)

numeric_transformer = Pipeline(steps=[
    ('num_imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())])

# Combine preprocessing steps
preprocess_steps = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

#Create dataframes with preprocessing done 
X_train_df = pd.DataFrame(preprocess_steps.fit_transform(df_train_x),columns = numeric_features)
X_oot_df = pd.DataFrame(preprocess_steps.transform(df_oot_x),columns = numeric_features)
X_oot_onedate_df = pd.DataFrame(preprocess_steps.transform(df_oot_onedate_x),columns = numeric_features)


from pickle import dump
dump(preprocess_steps, open('Pipeline/scaler.pkl', 'wb'))


X_oot_df.shape


# Separate features and labels
bsc_X, bsc_y = X_train_df.values, y_train_df.values

# Split data 75%-25% into training set and test set
x_bsc_train, x_bsc_test, y_bsc_train, y_bsc_test = train_test_split(bsc_X, bsc_y,test_size=0.25,random_state=0,stratify=bsc_y)


## ---------- XGBoost model ----------

# declaring and fitting xgb classifier
xgb_clf = xgb.XGBClassifier(objective='multi:prob', 
                            num_class=4, 
                            missing=1,
                            gamma=0, # default gamma value
                            learning_rate=0.3,
                            max_depth=12, 
                            reg_lambda=1, # default L2 value
                            #subsample=0.8, # tried but not ideal
                            #colsample_bytree=0.3, # tried but not ideal
                            #early_stopping_rounds=10,
                            eval_metric=['merror','mlogloss'],
                            seed=42)
xgb_clf.fit(x_bsc_train, 
            y_bsc_train,
            verbose=0, # set to 1 to see xgb training round intermediate results
           # sample_weight=sample_weights, # class weights to combat unbalanced 'target', tried but overfitting 
            eval_set=[(x_bsc_train, y_bsc_train), (x_bsc_test, y_bsc_test)])

##---------Prediction---------

y_pred = xgb_clf.predict_proba(x_bsc_test)
y_pred_oot = xgb_clf.predict_proba(X_oot_df)
y_pred_oot_onedate = xgb_clf.predict_proba(X_oot_onedate_df)


import pickle
#file_name = "xgb_model_57vars.pkl"

# save
#pickle.dump(xgb_clf, open(file_name, "wb"))


#did not work
#xgb_clf.save_model("xgb_model_57vars.json")


xgb_clf.save_model("Pipeline/xgb_model_57vars.txt")


df_test_predictions = pd.DataFrame(data = y_pred,columns = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad'])
#df_test_predictions.to_csv('df_test_predictions.csv', index = False)
auc = roc_auc_score(y_bsc_test, y_pred ,multi_class='ovr')
print('AUC score for test:', auc)

auc = roc_auc_score(y_oot_df.values, y_pred_oot ,multi_class='ovr')
print('AUC score for oot:', auc)

auc = roc_auc_score(y_oot_onedate_df.values, y_pred_oot_onedate ,multi_class='ovr')
print('AUC score for oot feb 23:', auc)


classes_combinations = []
classes = xgb_clf.classes_
class_list = list(classes)
for i in range(len(class_list)):
    for j in range(i+1, len(class_list)):
        classes_combinations.append([class_list[i], class_list[j]])
        classes_combinations.append([class_list[j], class_list[i]])


classes_combinations


roc_auc_ovo = {}
for i in range(len(classes_combinations)):
    # Gets the class
    comb = classes_combinations[i]
    c1 = comb[0]
    c2 = comb[1]
    c1_index = class_list.index(c1)
    title = str(c1) + " vs " + str(c2)
    
    # Prepares an auxiliar dataframe to help with the plots
    df_aux = X_oot_df.copy()
    df_aux['class'] = y_oot_df.values
    df_aux['prob'] = y_pred_oot[:, c1_index]
    
    # Slices only the subset with both classes
    df_aux = df_aux[(df_aux['class'] == c1) | (df_aux['class'] == c2)]
    df_aux['class'] = [1 if y == c1 else 0 for y in df_aux['class']]
    df_aux = df_aux.reset_index(drop = True)
    
    # Calculates the ROC AUC OvO
    roc_auc_ovo[title] = roc_auc_score(df_aux['class'], df_aux['prob'])


roc_auc_ovo


df_oot_predictions = pd.DataFrame(data = y_pred_oot,columns = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad'])


df_sum = df_oot_predictions['is_good']+df_oot_predictions['is_median_cb_bad_and_refund_bad']+df_oot_predictions['is_severe_refund_bad']+df_oot_predictions['is_severe_cb_bad']


df_oot_predictions.head(10)


feature_imp = pd.DataFrame({'Value':xgb_clf.feature_importances_,'Feature':X_train_df.columns})
display(feature_imp.sort_values(by = 'Value', ascending = False))
display(feature_imp.shape)


feature_imp.sort_values(by = 'Value', ascending = False)[:10]['Feature'].values


y_pred_abs = [np.argmax(line) for line in y_pred]
pd.crosstab(y_bsc_test, np.array(y_pred_abs), rownames=['Actual Values'], colnames=['Predicted Values'])


y_pred_oot_abs = [np.argmax(line) for line in y_pred_oot]
pd.crosstab(y_oot_df.values, np.array(y_pred_oot_abs), rownames=['Actual Values'], colnames=['Predicted Values'])


y_pred_oot_onedate_abs = [np.argmax(line) for line in y_pred_oot_onedate]
pd.crosstab(y_oot_onedate_df.values, np.array(y_pred_oot_onedate_abs), rownames=['Actual Values'], colnames=['Predicted Values'])


# compute SHAP values
explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_train_df)


shap_values_oot = explainer.shap_values(X_oot_df)


df_oot_predictions_is_bad = pd.DataFrame(data = y_pred_oot,columns = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad'])
df_oot_predictions_is_bad.head()


df_oot_predictions_is_bad.drop(columns = ['is_good'],inplace = True)
df_oot_predictions_is_bad['max_bad_score_column'] = df_oot_predictions_is_bad.idxmax(axis = 1)
df_oot_predictions_is_bad['max_bad_score'] = df_oot_predictions_is_bad.max(axis = 1)
df_oot_predictions_is_bad['partner_id'] = df_oot['PRTNR_SRC_ORG_CD']


class_names = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad']


fig = plt.figure()
shap.summary_plot(shap_values, X_train_df.values, plot_type="bar", class_names= class_names, max_display = 15,feature_names = X_train_df.columns, show =False)
plt.xlim([0,2])
plt.legend(loc = 'lower right')
plt.show()


#is_good explanability
shap.summary_plot(shap_values[0], X_train_df.values, feature_names = X_train_df.columns)


#is_median_cb_bad_and_refund_bad explanability
shap.summary_plot(shap_values[1], X_train_df.values, feature_names = X_train_df.columns)


#is_severe_refund explanability
shap.summary_plot(shap_values[2], X_train_df.values, feature_names = X_train_df.columns)


#is_severe_cbk_bad explanability
shap.summary_plot(shap_values[3], X_train_df.values, feature_names = X_train_df.columns)


top_cols = [
'cust_cnt',
'distinct_ato_vulnerable_cust_cnt_rate_180d',
'distinct_ato_vulnerable_cust_cnt_rate_30d',
'distinct_dormant_cust_cnt_rate_180d',
'distinct_inactive_cust_cnt_rate',
'distinct_inactive_cust_cnt_rate_180d',
'distinct_new_active_cust_cnt_rate',
'distinct_sf_like_cust_cnt_rate_180d',
'distinct_sf_like_cust_cnt_rate_30d',
'distinct_synthetic_bot_cust_cnt_rate_180d',
'distinct_young_cust_cnt_rate_180d',
'first_time_buyers',
'international_seller',
'max_ipregion_sale_contribution_180d',
'max_ipzip_sale_contribution_180d',
'max_product_sale_contribution_180d',
'max_product_sale_contribution_30d',
'max_shipcity_sale_contribution_180d',
'max_shipcity_sale_contribution_30d',
'max_shipstate_sale_contribution_180d',
'max_shipzip_sale_180d',
'mismatch_rate_30D',
'mismatch_rate_90D',
'Reuse_rate_60D',
'Reuse_rate_90D',
'sales_from_ato_vulnerable_contribution_180d',
'sales_from_ato_vulnerable_contribution_30d',
'sales_from_dormant_cust_contribution',
'sales_from_dormant_cust_contribution_180d',
'sales_from_inactive_cust_contribution',
'sales_from_inactive_cust_contribution_180d',
'sales_from_new_active_cust_contribution',
'sales_from_new_active_cust_contribution_180d',
'sales_from_new_cust_contribution',
'sales_from_new_cust_contribution_180d',
'sales_from_sf_contribution',
'sales_from_sf_contribution_180d',
'sales_from_sf_contribution_30d',
'sales_from_synthetic_bot_account',
'sales_from_synthetic_bot_account_contribution_180d',
'sales_from_young_cust_contribution_180d',
'seller_amt_30d',
'seller_amt_7d',
'seller_cnt_14d',
'seller_tof',
'SlrRspns_14D',
'SlrRspns_90D',
'total_buyers',
'total_sales',
'total_sellers_connected_device',
'total_suspended_sellers_connected_device',
'total_terminated_sellers_connected_device',
'VTR_14',
'VTR_90',
'sam_seller',
'Fraud_Cancel_Rate_14D','Fraud_Cancel_Rate_30D',
'sim_date',
'PRTNR_SRC_ORG_CD',
'seller_status',
'label']


df_train = pd.read_csv("final_bsc_all_train_correct.csv")
df_oot = pd.read_csv("final_bsc_all_oot_correct.csv")


df_oot.shape


df_oot_predictions['partner_id'] = df_oot['PRTNR_SRC_ORG_CD']
df_oot_predictions['sim_date'] = df_oot['sim_date']
df_oot_predictions['actual label'] = df_oot['label']


df_oot_predictions[df_oot_predictions['partner_id'] == 10001069312]


df_oot_predictions[df_oot_predictions['partner_id'] == 10001150015]


df_oot_predictions[df_oot_predictions['partner_id'] == 10001213198]


df_oot_predictions[df_oot_predictions['partner_id'] == 10001289156]


df_oot_predictions[df_oot_predictions['partner_id'] == 10001334271]


row = 12841
shap.waterfall_plot(shap.Explanation(values=shap_values_oot[3][row], 
                                              base_values=explainer.expected_value[0], data=X_oot_df.iloc[row],  
                                         feature_names=X_oot_df.columns.tolist()))


def gen_gains_chart_v3(df_input, y, wgts_ind=0, nbins=50):
    preds = df_input.copy()
    if wgts_ind == 0:
        preds['wgt'] = 1 
    else:
        a = pd.DataFrame()
        for w in preds['wgt'].unique():
            dup = pd.concat([preds[preds['wgt'] == w]]*w, ignore_index=True)
            a = a.append(dup)
        preds = a.copy()
        preds['wgt'] = 1 
        
    preds['Group'] = pd.qcut(preds['p1'].rank(method='first',ascending=False),nbins,labels=range(1,nbins+1)) 
    preds['actual'] = preds[y]

    aggregations = {
        'actual':{
           ('n_bad','sum'),
            ('n_recs', 'count')
            },
    'p1':{
        ('avg_pos_prob','mean'),
        ('min_pos_prob', 'min'),
        ('max_pos_prob','max')
        }
    }
    gains = pd.DataFrame(preds.groupby('Group').agg(aggregations))
    gains.columns = gains.columns.droplevel()
    gains.reset_index(inplace=True)
    for col in list(gains.columns):
        gains[col] = pd.to_numeric(gains[col])
    n_recs = gains.n_recs.sum()
    n_bad_cnt = gains.n_bad.sum()
    gains['n_recs_cum'] = gains['n_recs'].cumsum()
    gains['n_bad_cum'] = gains['n_bad'].cumsum()
    gains['n_good'] = gains['n_recs'] - gains['n_bad']
    gains['bad_catch_pct'] = round(gains['n_bad']/gains['n_recs'] * 100,2)
    gains['cum_bad_catch_pct'] = round(gains['n_bad_cum']/gains['n_recs_cum']* 100,2)
    gains['cum_bad_pct'] = round(gains['n_bad_cum']/n_bad_cnt* 100,2)
    gains['bad_pct_per_group'] = round(gains['n_bad']/n_bad_cnt* 100,2)
    gains['cum_bad_percent'] = round(gains['bad_pct_per_group'].cumsum() * 100,2)
    gains['FP Rate'] = round(gains['n_good']/gains['n_recs'],2)
    gains['cum FP Rate'] =  gains['FP Rate'].cumsum()
    gains['n_pct'] = round(gains['n_recs']/n_recs * 100,2)
    gains['cum_n_pct'] = round(gains['n_pct'].cumsum())
    gains_sub = gains[['Group','cum_n_pct','n_recs','n_bad',
                       'bad_catch_pct','cum_bad_pct','cum_bad_catch_pct','bad_pct_per_group',
          'FP Rate','cum FP Rate','min_pos_prob','avg_pos_prob','max_pos_prob']]
    gains_sub.columns = ['Group','Percentile','Number of transactions','Number of bad transactions',
                         'Bad Catch Percent','Cum Bad Catch Rate','Cum Bad Catch Percent','Bad Percent Per Group',
                         'FP Rate','Cum FP Rate','Min Pos Prob','Avg Pos Prob','Max Pos Prob']
    return gains_sub


def gains_chart(val,df,label_df):
    gains_df = pd.DataFrame()
    gains_df['p1'] = df[val]
    df_labels_test = pd.DataFrame()
    df_labels_test['label'] =  label_df
    df_labels_test['label_x'] = 0
    df_labels_test.loc[df_labels_test['label'] == val,['label_x']] = 1
    gains_df['is_bad'] = df_labels_test['label_x']
    y = 'is_bad'
    gains = gen_gains_chart_v3(gains_df,y,wgts_ind=0)
    return gains


def gen_gains_chart_amt_base(df_input, y, wgts_ind=0, nbins=50):
    preds = df_input.copy()
    if wgts_ind == 0:
        preds['wgt'] = 1 
    else:
        a = pd.DataFrame()
        for w in preds['wgt'].unique():
            dup = pd.concat([preds[preds['wgt'] == w]]*w, ignore_index=True)
            a = a.append(dup)
        preds = a.copy()
        preds['wgt'] = 1 
        
    preds['Group'] = pd.qcut(preds['p1'].rank(method='first',ascending=False),nbins,labels=range(1,nbins+1)) 
    preds['actual'] = preds[y]
    gains = pd.DataFrame(preds.groupby('Group').agg(
        n_bad = ('actual',sum),
        n_recs = ('actual','count'),
        avg_pos_prob = ('p1',np.mean),
        min_pos_prob = ('p1',np.min),
        max_pos_prob = ('p1',np.max),
        total_fraud_cb_amt = ('total_fraud_cb_amount',np.sum),
        total_LAD_rfnd = ('total_LAD_refunds',np.sum),
        total_SI_rfnd = ('total_SI_refunds',np.sum)
    ))
    gains.reset_index(inplace=True)
    for col in list(gains.columns):
        gains[col] = pd.to_numeric(gains[col])
    n_recs = gains.n_recs.sum()
    n_bad_cnt = gains.n_bad.sum()
    gains['n_recs_cum'] = gains['n_recs'].cumsum()
    gains['n_bad_cum'] = gains['n_bad'].cumsum()
    gains['n_good'] = gains['n_recs'] - gains['n_bad']
    gains['bad_catch_pct'] = round(gains['n_bad']/gains['n_recs'] * 100,2)
    gains['cum_bad_catch_pct'] = round(gains['n_bad_cum']/gains['n_recs_cum']* 100,2)
    gains['cum_bad_pct'] = round(gains['n_bad_cum']/n_bad_cnt* 100,2)
    gains['bad_pct_per_group'] = round(gains['n_bad']/n_bad_cnt* 100,2)
    gains['cum_bad_percent'] = round(gains['bad_pct_per_group'].cumsum() * 100,2)
    gains['FP Rate'] = round(gains['n_good']/gains['n_recs'],2)
    gains['cum FP Rate'] =  gains['FP Rate'].cumsum()
    gains['n_pct'] = round(gains['n_recs']/n_recs * 100,2)
    gains['cum_n_pct'] = round(gains['n_pct'].cumsum())
    gains_sub = gains[['Group','cum_n_pct','n_recs','n_bad',
                       'bad_catch_pct','cum_bad_pct','cum_bad_catch_pct','bad_pct_per_group',
          'FP Rate','cum FP Rate','min_pos_prob','avg_pos_prob','max_pos_prob','total_fraud_cb_amt','total_LAD_rfnd','total_SI_rfnd']]
    gains_sub.columns = ['Group','Percentile','Number of transactions','Number of bad transactions',
                         'Bad Catch Percent','Cum Bad Catch Rate','Cum Bad Catch Percent','Bad Percent Per Group',
                         'FP Rate','Cum FP Rate','Min Pos Prob','Avg Pos Prob','Max Pos Prob','Total Fraud Amt','Total LAD rfnd','Total SI rfnd']
    return gains_sub


def gains_chart_amt(val,df,label_df):
    gains_df = pd.DataFrame()
    gains_df['p1'] = df[val]
    gains_df['total_fraud_cb_amount'] = df['total_fraud_cb_amount']
    gains_df['total_LAD_refunds'] = df['total_LAD_refunds']
    gains_df['total_SI_refunds'] = df['total_SI_refunds']
    df_labels_test = pd.DataFrame()
    df_labels_test['label'] =  label_df
    df_labels_test['label_x'] = 0
    df_labels_test.loc[df_labels_test['label'] == int(val),['label_x']] = 1
    gains_df['is_bad'] = df_labels_test['label_x']
    y = 'is_bad'
    gains = gen_gains_chart_amt_base(gains_df,y,wgts_ind=0)
    return gains


instance_train , instance_test, y_instance_train, y_instance_test = train_test_split(df_train[['sim_date','PRTNR_SRC_ORG_CD']].values, bsc_y,test_size=0.25,random_state=0,stratify=bsc_y)
df_instance_test = pd.DataFrame(instance_test)
df_instance_test.rename(columns = {0:'sim_date', 1 : 'partner_id'},inplace = True)


df_instance_test['0'] =pd.DataFrame(y_pred)[0]
df_instance_test['1'] =pd.DataFrame(y_pred)[1]
df_instance_test['2'] =pd.DataFrame(y_pred)[2]
df_instance_test['3'] =pd.DataFrame(y_pred)[3]


df_kpi_vals = pd.read_csv("BQ_Data/bsc_raw_kpi_values_train/bsc_raw_kpi_values_train.csv")


df_kpi_vals.shape


df_kpi_instance = pd.merge(df_instance_test,df_kpi_vals,how = 'left', left_on =['partner_id','sim_date'], right_on =['PRTNR_SRC_ORG_CD','sim_date'])


df_kpi_instance.drop(columns = ['PRTNR_SRC_ORG_CD'],inplace = True)


gains_chart_amt('2',df_kpi_instance,y_bsc_test)


gains_chart(3,pd.DataFrame(y_pred_oot_onedate),y_oot_onedate_df.values)


#gains_chart(3,pd.DataFrame(y_pred_oot_onedate),y_oot_onedate_df.values).to_csv("share_with_pmt/gc_oot_feb23_with_fraud_cncl_cbk.csv", index = False)
#gains_chart(2,pd.DataFrame(y_pred_oot_onedate),y_oot_onedate_df.values).to_csv("share_with_pmt/gc_oot_feb23_with_fraud_cncl_rfnd.csv", index = False)
#gains_chart(1,pd.DataFrame(y_pred_oot_onedate),y_oot_onedate_df.values).to_csv("share_with_pmt/gc_oot_feb23_with_fraud_cncl_cbk&rfnd.csv", index = False)


gains_chart(2,pd.DataFrame(y_pred_oot),y_oot_df.values)


#gains_chart(3,pd.DataFrame(y_pred_oot),y_oot_df.values).to_csv("share_with_pmt/gc_oot_with_fraud_cncl_cbk.csv", index = False)
#gains_chart(2,pd.DataFrame(y_pred_oot),y_oot_df.values).to_csv("share_with_pmt/gc_oot_with_fraud_cncl_rfnd.csv", index = False)
#gains_chart(1,pd.DataFrame(y_pred_oot),y_oot_df.values).to_csv("share_with_pmt/gc_oot_with_fraud_cncl_cbk&rfnd.csv", index = False)


df_gen_jul10 = pd.read_csv("Data/July10/marketplace_bsc_mvp_final_vars_jul10_selected.csv")
df_gen_jul18 = pd.read_csv("BQ_Data/July18_2023/bsc_final_vars_jul18.csv")


display(df_gen_jul10.shape)
display(df_gen_jul18.shape)


df_gen_jul10_x = df_gen_jul10.drop(columns = ['partner_id'])
df_gen_jul18_x = df_gen_jul18.drop(columns = ['partner_id'])

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_features = list(df_gen_jul10_x.select_dtypes(include=numerics).columns)

#Create dataframes with preprocessing done 
X_gen_jul10_df = pd.DataFrame(preprocess_steps.transform(df_gen_jul10_x),columns = numeric_features)
X_gen_jul18_df = pd.DataFrame(preprocess_steps.transform(df_gen_jul18_x),columns = numeric_features)


X_gen_jul18_df.head()


display(X_gen_jul10_df.shape)
display(X_gen_jul18_df.shape)


y_gen_jul10_pred = xgb_clf.predict_proba(X_gen_jul10_df)
y_gen_jul18_pred = xgb_clf.predict_proba(X_gen_jul18_df)


df_gen_jul10_predictions = pd.DataFrame(data = y_gen_jul10_pred,columns = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad'])
df_gen_jul18_predictions = pd.DataFrame(data = y_gen_jul18_pred,columns = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad'])
df_gen_jul18_predictions.head()


y_gen_jul10_pred_abs = [np.argmax(line) for line in y_gen_jul10_pred]
y_gen_jul18_pred_abs = [np.argmax(line) for line in y_gen_jul18_pred]


pd.Series(y_gen_jul10_pred_abs).value_counts()


pd.Series(y_gen_jul18_pred_abs).value_counts()


# compute SHAP values
explainer = shap.TreeExplainer(xgb_clf)
shap_values_jul10 = explainer.shap_values(X_gen_jul10_df)
shap_values_jul18 = explainer.shap_values(X_gen_jul18_df)


pickle.dump(explainer, open('shap_explainer.pkl', "wb"))


df_gen_jul10_predictions_is_bad = pd.DataFrame(data = y_gen_jul10_pred,columns = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad'])
df_gen_jul10_predictions_is_bad.drop(columns = ['is_good'],inplace = True)
df_gen_jul10_predictions_is_bad['max_bad_score_column'] = df_gen_jul10_predictions_is_bad.idxmax(axis = 1)
df_gen_jul10_predictions_is_bad['max_bad_score'] = df_gen_jul10_predictions_is_bad.max(axis = 1)
df_gen_jul10_predictions_is_bad['partner_id'] = df_gen_jul10['partner_id']


df_gen_jul18_predictions_is_bad = pd.DataFrame(data = y_gen_jul18_pred,columns = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad'])
df_gen_jul18_predictions_is_bad.drop(columns = ['is_good'],inplace = True)
df_gen_jul18_predictions_is_bad['max_bad_score_column'] = df_gen_jul18_predictions_is_bad.idxmax(axis = 1)
df_gen_jul18_predictions_is_bad['max_bad_score'] = df_gen_jul18_predictions_is_bad.max(axis = 1)
df_gen_jul18_predictions_is_bad['partner_id'] = df_gen_jul18['partner_id']


label_mapping = pd.DataFrame(np.array([[0,'is_good'], [1,'is_median_cb_bad_and_refund_bad'],[2,'is_severe_refund_bad'],[3,'is_severe_cb_bad']]),
                   columns=['index_val','col_name'])


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
'Fraud_Cancel_Rate_14D','Fraud_Cancel_Rate_30D',]


reason_code_all_jul10 = []
for idx, row in df_gen_jul10_predictions_is_bad.iterrows():
    max_col = row['max_bad_score_column']
    max_col_idx = int(label_mapping[label_mapping['col_name'] == max_col]['index_val'].values[0])
    shap_row = pd.DataFrame(shap.Explanation(values=shap_values_jul10[max_col_idx][idx],
                                             base_values=explainer.expected_value[0], data=X_gen_jul10_df.iloc[idx],
                                             feature_names=X_gen_jul10_df.columns.tolist()).data).reset_index()
    shap_row.sort_values(by = idx, ascending = False, inplace = True)
    reason_code = ';'.join(shap_row[shap_row['index'].isin(reason_code_vals) & shap_row[idx] > 0][:5]['index'].values)
    reason_code_all_jul10.append(reason_code)


reason_code_all_jul18 = []
for idx, row in df_gen_jul18_predictions_is_bad.iterrows():
    max_col = row['max_bad_score_column']
    max_col_idx = int(label_mapping[label_mapping['col_name'] == max_col]['index_val'].values[0])
    shap_row = pd.DataFrame(shap.Explanation(values=shap_values_jul18[max_col_idx][idx],
                                             base_values=explainer.expected_value[0], data=X_gen_jul18_df.iloc[idx],
                                             feature_names=X_gen_jul18_df.columns.tolist()).data).reset_index()
    shap_row.sort_values(by = idx, ascending = False, inplace = True)
    reason_code = ';'.join(shap_row[shap_row['index'].isin(reason_code_vals) & shap_row[idx] > 0][:5]['index'].values)
    reason_code_all_jul18.append(reason_code)


#Code for reason code for every class
#reason_code_all_oot_all_classes = []
#for idx, row in df_oot_predictions_is_bad.iterrows():
#    shap_row1 = pd.DataFrame(shap.Explanation(values=shap_values_oot[1][idx],
#                                             base_values=explainer.expected_value[0], data=X_gen_df.iloc[idx],
#                                             feature_names=X_gen_df.columns.tolist()).data).reset_index()
#    
#    shap_row2 = pd.DataFrame(shap.Explanation(values=shap_values_oot[2][idx],
#                                             base_values=explainer.expected_value[0], data=X_gen_df.iloc[idx],
#                                             feature_names=X_gen_df.columns.tolist()).data).reset_index()
#    
#    shap_row3 = pd.DataFrame(shap.Explanation(values=shap_values_oot[3][idx],
#                                             base_values=explainer.expected_value[0], data=X_gen_df.iloc[idx],
#                                             feature_names=X_gen_df.columns.tolist()).data).reset_index()
#    
#    shap_row1.sort_values(by = idx, ascending = False, inplace = True)
#    shap_row2.sort_values(by = idx, ascending = False, inplace = True)
#    shap_row3.sort_values(by = idx, ascending = False, inplace = True)
#    
#    reason_code1 = ';'.join(shap_row[(shap_row['index'].isin(reason_code_vals)) & (shap_row1[idx] > 0)][:3]['index'].values)
#    reason_code2 = ';'.join(shap_row[(shap_row['index'].isin(reason_code_vals)) & (shap_row2[idx] > 0)][:3]['index'].values)
#    reason_code3 = ';'.join(shap_row[(shap_row['index'].isin(reason_code_vals)) & (shap_row3[idx] > 0)][:3]['index'].values)
#    reason_code_all = {'median_cbk_rfnd': reason_code1 , 'rfnd': reason_code2, 'cbk': reason_code3}
#    reason_code_all_oot_all_classes.append(reason_code_all)


#df_oot_predictions_is_bad['reason_code_all_classes'] = reason_code_all_oot_all_classes


#df_oot_predictions_is_bad['reason_code_for_max_bad_class'] = reason_code_all_oot


df_gen_jul10_predictions_is_bad['reason_code_for_max_bad_class'] = reason_code_all_jul10
df_gen_jul18_predictions_is_bad['reason_code_for_max_bad_class'] = reason_code_all_jul18


df_gen_jul18_predictions_is_bad[df_gen_jul18_predictions_is_bad['partner_id'] == 10001420980]


#df_gen_jul10_predictions_is_bad['is_severe_refund_bad'].describe()


df_gen_jul10_predictions_is_bad['is_severe_refund_bad'].describe()


df_gen_jul18_predictions_is_bad['is_severe_refund_bad'].describe()


#df_gen_jul10_predictions_is_bad.to_csv("share_with_pmt/updated_results/df_gen_predictions_is_bad_with_fraud_cancel_July10.csv", index=False)
#df_gen_jul18_predictions_is_bad.to_csv("share_with_pmt/updated_results/df_gen_predictions_is_bad_with_fraud_cancel_July18.csv", index=False)


#df_gen_predictions_is_bad.sort_values(by = 'is_severe_refund_bad', ascending = False).head(100)


#df_gen_predictions_is_bad.to_csv("df_gen_predictions_is_bad_with_fraud_cancel_new.csv", index=False)


#df_oot_predictions_is_bad.to_csv("df_oot_predictions_is_bad_with_fraud_cancel_all_reason_code.csv", index=False)


#df_oot_predictions_is_bad.to_csv("df_oot_predictions_is_bad_with_fraud_cancel.csv", index=False)


df_all_oot = pd.read_csv("/home/jupyter/Data/bsc_oot_correct.csv", usecols = top_cols)


df_all_oot_feb23 = df_all_oot[df_all_oot['sim_date'] == 'feb23']
df_all_oot_jan26 = df_all_oot[df_all_oot['sim_date'] == 'jan26']
df_all_oot_mar23 = df_all_oot[df_all_oot['sim_date'] == 'mar23']


df_all_oot_mar23['first_time_buyer_rate'] = df_all_oot_mar23['first_time_buyers']/df_all_oot_mar23['total_buyers']
df_all_oot_mar23.drop(columns = ['first_time_buyers','sim_date'], inplace = True)


df_all_oot_feb23['first_time_buyer_rate'] = df_all_oot_feb23['first_time_buyers']/df_all_oot_feb23['total_buyers']
df_all_oot_feb23.drop(columns = ['first_time_buyers','sim_date'], inplace = True)


df_all_oot_jan26['first_time_buyer_rate'] = df_all_oot_jan26['first_time_buyers']/df_all_oot_jan26['total_buyers']
df_all_oot_jan26.drop(columns = ['first_time_buyers','sim_date'], inplace = True)


df_all_oot_jan26_x = df_all_oot_jan26.drop(columns = ['PRTNR_SRC_ORG_CD','label'])
df_all_oot_mar23_x = df_all_oot_mar23.drop(columns = ['PRTNR_SRC_ORG_CD','label'])
df_all_oot_feb23_x = df_all_oot_feb23.drop(columns = ['PRTNR_SRC_ORG_CD','label'])

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_features = list(df_all_oot_mar23_x.select_dtypes(include=numerics).columns)

#Create dataframes with preprocessing done 
X_oot_jan26_df = pd.DataFrame(preprocess_steps.transform(df_all_oot_jan26_x),columns = numeric_features)
X_oot_feb23_df = pd.DataFrame(preprocess_steps.transform(df_all_oot_feb23_x),columns = numeric_features)
X_oot_mar23_df = pd.DataFrame(preprocess_steps.transform(df_all_oot_mar23_x),columns = numeric_features)


y_oot_jan26_pred = xgb_clf.predict_proba(X_oot_jan26_df)
y_oot_feb23_pred = xgb_clf.predict_proba(X_oot_feb23_df)
y_oot_mar23_pred = xgb_clf.predict_proba(X_oot_mar23_df)


df_all_oot_jan26['label_new'] = 4
df_all_oot_jan26.loc[df_all_oot_jan26['label'] == 1, 'label_new'] = 0
df_all_oot_jan26.loc[df_all_oot_jan26['label'] == 2, 'label_new'] = 1
df_all_oot_jan26.loc[df_all_oot_jan26['label'] == 3, 'label_new'] = 2
df_all_oot_jan26.loc[df_all_oot_jan26['label'] == 4, 'label_new'] = 3


df_all_oot_feb23['label_new'] = 4
df_all_oot_feb23.loc[df_all_oot_feb23['label'] == 1, 'label_new'] = 0
df_all_oot_feb23.loc[df_all_oot_feb23['label'] == 2, 'label_new'] = 1
df_all_oot_feb23.loc[df_all_oot_feb23['label'] == 3, 'label_new'] = 2
df_all_oot_feb23.loc[df_all_oot_feb23['label'] == 4, 'label_new'] = 3


df_all_oot_mar23['label_new'] = 4
df_all_oot_mar23.loc[df_all_oot_mar23['label'] == 1, 'label_new'] = 0
df_all_oot_mar23.loc[df_all_oot_mar23['label'] == 2, 'label_new'] = 1
df_all_oot_mar23.loc[df_all_oot_mar23['label'] == 3, 'label_new'] = 2
df_all_oot_mar23.loc[df_all_oot_mar23['label'] == 4, 'label_new'] = 3


explainer = shap.TreeExplainer(xgb_clf)
shap_values_jan26 = explainer.shap_values(X_oot_jan26_df)
shap_values_feb23 = explainer.shap_values(X_oot_feb23_df)
shap_values_mar23 = explainer.shap_values(X_oot_mar23_df)


df_oot_jan26_predictions_is_bad = pd.DataFrame(data = y_oot_jan26_pred,columns = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad'])
df_oot_jan26_predictions_is_bad.drop(columns = ['is_good'],inplace = True)
df_oot_jan26_predictions_is_bad['max_bad_score_column'] = df_oot_jan26_predictions_is_bad.idxmax(axis = 1)
df_oot_jan26_predictions_is_bad['max_bad_score'] = df_oot_jan26_predictions_is_bad.max(axis = 1)
df_oot_jan26_predictions_is_bad['partner_id'] = df_all_oot_jan26['PRTNR_SRC_ORG_CD'].values


df_oot_feb23_predictions_is_bad = pd.DataFrame(data = y_oot_feb23_pred,columns = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad'])
df_oot_feb23_predictions_is_bad.drop(columns = ['is_good'],inplace = True)
df_oot_feb23_predictions_is_bad['max_bad_score_column'] = df_oot_feb23_predictions_is_bad.idxmax(axis = 1)
df_oot_feb23_predictions_is_bad['max_bad_score'] = df_oot_feb23_predictions_is_bad.max(axis = 1)
df_oot_feb23_predictions_is_bad['partner_id'] = df_all_oot_feb23['PRTNR_SRC_ORG_CD'].values


df_oot_mar23_predictions_is_bad = pd.DataFrame(data = y_oot_mar23_pred,columns = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad'])
df_oot_mar23_predictions_is_bad.drop(columns = ['is_good'],inplace = True)
df_oot_mar23_predictions_is_bad['max_bad_score_column'] = df_oot_mar23_predictions_is_bad.idxmax(axis = 1)
df_oot_mar23_predictions_is_bad['max_bad_score'] = df_oot_mar23_predictions_is_bad.max(axis = 1)
df_oot_mar23_predictions_is_bad['partner_id'] = df_all_oot_mar23['PRTNR_SRC_ORG_CD'].values


reason_code_all_jan26 = []
for idx, row in df_oot_jan26_predictions_is_bad.iterrows():
    max_col = row['max_bad_score_column']
    max_col_idx = int(label_mapping[label_mapping['col_name'] == max_col]['index_val'].values[0])
    shap_row = pd.DataFrame(shap.Explanation(values=shap_values_jan26[max_col_idx][idx],
                                             base_values=explainer.expected_value[0], data=X_oot_jan26_df.iloc[idx],
                                             feature_names=df_oot_jan26_predictions_is_bad.columns.tolist()).data).reset_index()
    shap_row.sort_values(by = idx, ascending = False, inplace = True)
    reason_code = ';'.join(shap_row[shap_row['index'].isin(reason_code_vals) & shap_row[idx] > 0][:5]['index'].values)
    reason_code_all_jan26.append(reason_code)


reason_code_all_feb23 = []
for idx, row in df_oot_feb23_predictions_is_bad.iterrows():
    max_col = row['max_bad_score_column']
    max_col_idx = int(label_mapping[label_mapping['col_name'] == max_col]['index_val'].values[0])
    shap_row = pd.DataFrame(shap.Explanation(values=shap_values_feb23[max_col_idx][idx],
                                             base_values=explainer.expected_value[0], data=X_oot_feb23_df.iloc[idx],
                                             feature_names=df_oot_feb23_predictions_is_bad.columns.tolist()).data).reset_index()
    shap_row.sort_values(by = idx, ascending = False, inplace = True)
    reason_code = ';'.join(shap_row[shap_row['index'].isin(reason_code_vals) & shap_row[idx] > 0][:5]['index'].values)
    reason_code_all_feb23.append(reason_code)


reason_code_all_mar23 = []
for idx, row in df_oot_mar23_predictions_is_bad.iterrows():
    max_col = row['max_bad_score_column']
    max_col_idx = int(label_mapping[label_mapping['col_name'] == max_col]['index_val'].values[0])
    shap_row = pd.DataFrame(shap.Explanation(values=shap_values_mar23[max_col_idx][idx],
                                             base_values=explainer.expected_value[0], data=X_oot_mar23_df.iloc[idx],
                                             feature_names=df_oot_mar23_predictions_is_bad.columns.tolist()).data).reset_index()
    shap_row.sort_values(by = idx, ascending = False, inplace = True)
    reason_code = ';'.join(shap_row[shap_row['index'].isin(reason_code_vals) & shap_row[idx] > 0][:5]['index'].values)
    reason_code_all_mar23.append(reason_code)


df_oot_jan26_predictions_is_bad['reason_code_max_score'] = reason_code_all_jan26
df_oot_feb23_predictions_is_bad['reason_code_max_score'] = reason_code_all_feb23
df_oot_mar23_predictions_is_bad['reason_code_max_score'] = reason_code_all_mar23


#df_oot_mar23_predictions_is_bad.to_csv('share_with_pmt/df_oot_mar23_predictions_is_bad_correct.csv')
#df_oot_feb23_predictions_is_bad.to_csv('share_with_pmt/df_oot_feb23_predictions_is_bad_correct.csv')
#df_oot_jan26_predictions_is_bad.to_csv('share_with_pmt/df_oot_jan26_predictions_is_bad_correct.csv')


df_all_train = pd.read_csv("/home/jupyter/Data/bsc_train_correct.csv", usecols = top_cols)


df_all_train['first_time_buyer_rate'] = df_all_train['first_time_buyers']/df_all_train['total_buyers']
df_all_train.drop(columns = ['first_time_buyers','sim_date'], inplace = True)


df_all_train_x = df_all_train.drop(columns = ['PRTNR_SRC_ORG_CD','label'])

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_features = list(df_all_train_x.select_dtypes(include=numerics).columns)

#Create dataframes with preprocessing done 
X_all_train_df = pd.DataFrame(preprocess_steps.transform(df_all_train_x),columns = numeric_features)


y_all_train_pred = xgb_clf.predict_proba(X_all_train_df)


df_all_train_predictions = pd.DataFrame(data = y_all_train_pred,columns = ['is_good','is_median_cb_bad_and_refund_bad','is_severe_refund_bad','is_severe_cb_bad'])


df_all_train_predictions.to_csv('df_all_train_predictions_correct.csv',index = False)






```
