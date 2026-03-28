"""
author:  Alex, Elixir
date: 20230328
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from metrics import *
import pandas as pd
import random
import pickle
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge
SEED = 88


def train_and_eval(x, y, trn_data, tst_data, model_name, models):
    random.seed(SEED)
    np.random.seed(SEED)
    for index, model in enumerate(models):
        file_name = output_file + model_name[index] + "/"
        if not os.path.exists(file_name): os.makedirs(file_name)
        # 5-fold
        for iter_ in range(len(trn_data)):
            rmse, pearson, spearman, ci, r_square_score, MedAE = [], [], [], [], [], [],
            model.fit(x[trn_data[iter_]], y[trn_data[iter_]])
            y_pred = model.predict(x[tst_data[iter_]])
            pickle.dump(model, open(file_name + str(iter_ + 1) + 'fold' + 'model.sav', 'wb'))

            regscore = reg_scores(y[tst_data[iter_]], y_pred)
            rmse.append(np.array(regscore[0]))
            pearson.append(np.array(regscore[1]))
            spearman.append(regscore[2])
            ci.append(regscore[3])
            r_square_score.append(regscore[4])
            MedAE.append(regscore[5])
            with open(file_name + "preds_reg.csv", 'a+') as f:
                f.write(
                    "date" + ',' + "fold" + ',' + "rmse" + ',' + "pearson" + ',' + "spearman" + ',' + "ci" + ',' + "r_square_score" + ',' + "MedAE" + '/n')
                f.write(str(date) + ',' + str(iter_ + 1) + ',' + str(regscore[0]) + ',' + str(regscore[1]) + ',' + str(
                    regscore[2]) + ',' + str(regscore[3]) + ',' + str(regscore[4]) + ',' + str(regscore[5]) + '/n')
            f.close()

            with open(file_name + "preds_score.csv", 'a+') as f:
                f.write('score' + ',' + 'pred' + '/n')
                for num, sc in enumerate(y[test]):
                    score = str(y[tst_data[iter_]][num])
                    pred = str(y_pred[num])
                    f.write(score + ',' + pred + '/n')
            f.close()
        print(
            "[data: %s][train_name:%s] [rmse: %f] [pearson: %f] [spearman: %f] [ci: %f] [r_square_score: %f]  [MedAE: %f] "
            % (date, model_name[index], np.mean(rmse),
               np.mean(pearson), np.mean(spearman), np.mean(ci), np.mean(r_square_score), np.mean(MedAE),
               ))


if __name__ == '__main__':
    date = "20230615"
    model_name = ['KNeighborsRegressor','SVR','BayesianRidge', 'RandomForestRegressor',
                  'GradientBoostingRegressor', 'AdaBoostRegressor', 'XGBRegressor']
    models = [KNeighborsRegressor(n_neighbors=5, weights='uniform'),
              SVR(),
              BayesianRidge(compute_score=True),
              RandomForestRegressor(n_estimators=500,n_jobs=-1,oob_score=True,max_features='sqrt'),
              GradientBoostingRegressor(n_estimators=500,learning_rate=0.1,max_depth=15,max_features='sqrt',min_samples_leaf=10,min_samples_split=10,loss='ls',random_state =42)
            , AdaBoostRegressor(),XGBRegressor(max_depth=7, n_estimators=200, learning_rate=0.1, use_label_encoder=False,
                             objective="rank:pairwise")]
    output_file = '../../input/Property_Evaluation/Train/LY6C1/'
    if not os.path.exists(output_file): os.makedirs(output_file)
    checkpoint_dir_D = '../../checkpoint/Property_Evaluation/LY6C1/'
    if not os.path.exists(checkpoint_dir_D): os.makedirs(checkpoint_dir_D)
    path = '../../input/Property_Evaluation/Train/LY6C1_desc_feature.csv'
    kf = KFold(5, shuffle=True,random_state=42)
    train_data = pd.read_csv(path, encoding="utf8", index_col=0)
    x, y = train_data.iloc[:, 1:-1].values, train_data['LY6C1_log2enr'].values
    total_trnset, total_tstset = [], []
    for train, test in kf.split(x):
        total_trnset.append(train), total_tstset.append(test)
    pickle.dump(total_trnset, file=open(output_file + 'total_trnset.pkl', 'wb'), protocol=0)
    pickle.dump(total_tstset, file=open(output_file + 'total_tstset.pkl', 'wb'), protocol=0)
    trn_data = pickle.load(file=open(output_file + 'total_trnset.pkl', 'rb'))
    tst_data = pickle.load(file=open(output_file + 'total_tstset.pkl', 'rb'))
    train_and_eval(x, y, trn_data, tst_data, model_name, models)
