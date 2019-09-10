import  pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc , f1_score

import collections
from collections import Counter


def classify_one_vs_many(df, model_name, model, feature_to_class, type_class, type_0_class=None):
    GH_df_reduced_one_vs_many = df.copy()
    if type_0_class is None:
        others_df = GH_df_reduced_one_vs_many[(GH_df_reduced_one_vs_many[feature_to_class] != type_class)].copy()
        others_df.loc[:, 'ml_type'] = type_0_class = 'others'
    else:
        others_df = GH_df_reduced_one_vs_many[(GH_df_reduced_one_vs_many[feature_to_class] == type_0_class)].copy()
        others_df.loc[:, 'ml_type'] = type_0_class
    category_df = GH_df_reduced_one_vs_many[GH_df_reduced_one_vs_many[feature_to_class] == type_class].copy()
    category_df.loc[:, 'ml_type'] = type_class

    df_merged = pd.concat([others_df, category_df], ignore_index=True)

    # print df_merged.groupby(['ml_type','category'])['analizo_accm_mean'].count()
    X = df_merged.select_dtypes(include=[np.number])
    y = df_merged.loc[:, 'ml_type']
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # ros = RandomOverSampler(random_state=0)
    # if len(y_train.unique()) < 2:
    #     print ('cannot fit for {}'.format(type_class))
    #     return None
    # X_resampled, Y_resampled = ros.fit_resample(X_train, y_train)

    # print('Training target statistics: {}'.format(Counter(y)))
    if Counter(y)[type_class] == 1:
        print ('cannot fit for {}'.format(type_class))
        return

    model.fit(X_train, y_train)
    # print model.score(X_test,y_test)
    rfe = RFE(model, 4)
    fit = rfe.fit(X_train, y_train)
    # print "Selected features : " + str(X.columns[fit.support_])
    pred = model.predict(X_test)
    #     print Counter(pred)
    #     df_accurarcy  = set_wrong_type(pred,y, df_merged,type_class)
    # calculate_accurarcy(df_accurarcy,pred,y,type_class)
    fpr = tpr = roc_auc = None
    t = True
    try:

        y_pred = model.predict_proba(X_test)[:, 1]
    except:
        t = False
    if t:
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=type_class)
        roc_auc = auc(tpr, fpr)
    f1 = f1_score(y_test, pred, pos_label=type_class)
    agent_type = category_df['agent_type'].unique()[0]
    return {'model_name': model_name, 'agent_type': agent_type, 'feature_importance': fit, 'model': model, 'fpr': fpr,
            'tpr': tpr, 'auc': roc_auc, 'f1_score': f1, 'class 0': type_0_class, 'class 1': type_class}