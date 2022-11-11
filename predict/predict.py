#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: BigZhuang
@file: linear_model.py
@time: 2022/10/8 15:08
@version:
@desc:
"""
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt    #画图的
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import SVR,LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor,RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import path
from sklearn.metrics import explained_variance_score
# import MSLE
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# loss与score
def MSLE(y, y_hut, flag=0):
    n = len(y)
    err = 0
    for i, j in zip(y, y_hut):
		# loss:
        err += 1/n*(pow((np.log(1 + i) - np.log(1 + j)), 2))
    if flag:
        # return 1 / (err* 20 + 1)
	    return 0
    else:
        #score
        return 1 / (err + 1),err

warnings.filterwarnings('ignore')
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 200)

# 用来进行测试准确度，data是从01-01到04-30，这个方法就是用来求解一个新的数据集，将数据集进行了缩小
def fueat(data):
	feat_col = []
	for i in range(7, 15):
		# 按照flow_id进行分组，然后进行数据偏移
		data[f"flow_{i}"] = data.groupby(["flow_id"])["flow"].shift(24*i)
		feat_col.extend([f"flow_{i}"])
		for func in ["mean", "max", "min"]:
			if func == "mean":
				data[f"flow_{i}_roll_{func}"] = data.groupby(["flow_id"])["flow"].shift(24 * i).rolling(6).mean()
			if func == "max":
				data[f"flow_{i}_roll_{func}"] = data.groupby(["flow_id"])["flow"].shift(24 * i).rolling(6).max()
			if func == "min":
				data[f"flow_{i}_roll_{func}"] = data.groupby(["flow_id"])["flow"].shift(24 * i).rolling(6).min()
			feat_col.extend([f"flow_{i}_roll_{func}"])
	data["flow_mean"] = data[["flow_7", "flow_8", "flow_9", "flow_10", "flow_11", "flow_12", "flow_13", "flow_14"]].mean(axis=1)
	data["flow_max"] = data[["flow_7", "flow_8", "flow_9", "flow_10", "flow_11", "flow_12", "flow_13", "flow_14"]].max(axis=1)
	data["flow_min"] = data[["flow_7", "flow_8", "flow_9", "flow_10", "flow_11", "flow_12", "flow_13", "flow_14"]].min(axis=1)
	data["flow_sum"] = data[["flow_7", "flow_8", "flow_9", "flow_10", "flow_11", "flow_12", "flow_13", "flow_14"]].sum(axis=1)
	data["flow_std"] = data[["flow_7", "flow_8", "flow_9", "flow_10", "flow_11", "flow_12", "flow_13", "flow_14"]].std(axis=1)
	feat_col.extend(["flow_mean", "flow_max", "flow_min", "flow_sum", "flow_std"])
	for i in range(24*7, 24*7+13):
		data[f"flow_lag_{i}"] = data.groupby(["flow_id"])["flow"].shift(i)
		data[f"flow_roll_{i}"] = data.groupby(["flow_id"])["flow"].shift(i).rolling(i+1).mean()
		feat_col.extend([f"flow_lag_{i}", f"flow_roll_{i}"])
	data["time"] = pd.to_datetime(data["time"])
	data["month"] = data["time"].dt.month
	# data["week"] = data["time"].dt.week
	# data["day"] = data["time"].dt.day
	data["dayofweek"] = data["time"].dt.dayofweek
	# data["dayofyear"] = data["time"].dt.dayofyear
	data["hour"] = data["time"].dt.hour
	data['sin_hour'] = np.sin(2 * np.pi * data["hour"] / 24)
	data['cos_hour'] = np.cos(2 * np.pi * data["hour"] / 24)
	feat_col.extend(["month", "dayofweek","hour",'sin_hour','cos_hour'])
	data["month_mean"] = data.groupby(["flow_id", "month"])["flow"].transform("mean")
	data["dayofweek_mean"] = data.groupby(["flow_id", "dayofweek"])["flow"].transform("mean")
	data["hour_mean"] = data.groupby(["flow_id", "hour"])["flow"].transform("mean")
	feat_col.extend(["hour_mean"])
	data.to_csv(path.predict_data_path + "data.csv", index=False)
	print("feat")
	print(feat_col)
	# feat_col.to_csv(conf.predict_data_path + "feat.csv", index=False)
	# data从01-01到04-30，以小时为单位
	# feat_col是data的列名
	return data, feat_col

# 返回利用不同模型时所预测的结果
def model(X_train, X_test, y_train, y_test, flag=0, if_train=True):
	if flag == 0:
		model_list = {
					  "svr": SVR(),
					  "randomtree": RandomForestRegressor(random_state=2022),
					  "xgboost": xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=100)
					  }
		# svr，随机森林,xgboost
		if if_train:
			pre_list_all = np.zeros((len(X_test),))
			pre_list_svr = []
			pre_list_rand = []
			pre_list_xgb=[]
			for key, model in model_list.items():
				print(f"============{key}========")
				model.fit(X_train, y_train)
				# pre即为预测的值，是一个列表
				pre = model.predict(X_test)
				if key == "xgboost":
					pre_list_xgb.extend(pre)
				elif key == "svr":
					pre_list_svr.extend(pre)
				else:
					pre_list_rand.extend(pre)
				pre_list_all += pre / len(model_list)
		return pre_list_all,  pre_list_svr, pre_list_rand,pre_list_xgb
		# 		返回的就是用各种模型预测的结果
	if flag == 1:
		estimators = [
			('knn', KNeighborsRegressor()),
			('svr', SVR()),
			('randomtree', RandomForestRegressor(random_state=2022)),
			('xgboost',xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=100))
		]
		reg = StackingRegressor(
			estimators=estimators,
			final_estimator=LinearRegression(),
			cv=5
		)
		#返回的是集成学习返回的结果
		if if_train:
			reg.fit(X_train, y_train)
			pre = reg.predict(X_test)
			return pre

# 查看准确度：看准确度时省略一些数据，选取03-01到04-30
def train(data, flag = 0):
	train1 = data[(data['time'] >= '2022-01-01 01:00:00') & (data['time'] < '2022-05-01 01:00:00')].reset_index(
		drop=True)
	train2 = data[(data['time'] >= '2022-05-08 01:00:00') & (data['time'] < '2022-06-01 01:00:00')].reset_index(
		drop=True)
	train3 = data[(data['time'] >= '2022-06-08 01:00:00') & (data['time'] < '2022-07-21 01:00:00')].reset_index(
		drop=True)
	train4 = data[(data['time'] >= '2022-07-28 01:00:00') & (data['time'] < '2022-08-21 01:00:00')].reset_index(
		drop=True)
	train1["pre"] = 0
	# data从01-01到04-30，以小时为单位
	data, feat = fueat(train1)
	# 训练集为01-01到04-30
	train_ = data[data['time'] >= '2022-03-01 01:00:00']
	# 记录集成学习的预测结果
	pre_list = []
	# 记录各类机器学习方法的预测结果
	pre_list_svr = []
	pre_list_rand = []
	pre_list_xgb=[]
	# 存储真实值
	test_list = []
	for id in [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]:
		print(f"+++++++++++++++++{id}+++++++++++++++++++++")
		# 划分训练集，
		train = train_[train_["flow_id"] == id]
		# 区分x和y  x和y共同组成了所有的列
		X_data, y_data = train[feat], train["flow"].values
		# 标准化数据
		stand = MinMaxScaler()
		X_data = stand.fit_transform(X_data)
		# 划分测试集和训练集（在03-01到04-30）
		X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=2022)
		if flag==1:
			# 返回的是集成学习的预测结果
			pre = model(X_train, X_test, y_train, y_test, flag=flag, if_train=True)
			# 预测值
			pre_list.extend(pre)
			# 真实值
			test_list.extend(y_test)
			print("score:", MSLE(test_list, pre_list, flag=0)[0])
			print("loss:", MSLE(test_list, pre_list, flag=0)[1])
			print("accuracy",explained_variance_score(test_list, pre_list))
		if flag==0:
			pre_all,pre_svr,pre_rand, pr_xgb= model(X_train, X_test, y_train, y_test,
																		 flag=flag, if_train=True)
			# 记录各个方法的预测值
			pre_list.extend(pre_all)
			pre_list_svr.extend(pre_svr)
			pre_list_rand.extend(pre_rand)
			pre_list_xgb.extend(pr_xgb)
			# 记录真实值
			test_list.extend(y_test)
			# #画图，看预测值和真实值之间的区别
			if id=='flow_2':
				plt.plot(test_list, color="r", label="actual")
				plt.plot(pre_list_xgb, color="b", label="xgboost")
				plt.plot(pre_list_svr, color="m", label="svr")
				plt.plot(pre_list_rand, color="k", label="rand")
				# plt.plot(pre_list,color="m", label="stacking")
				plt.xlabel("时间")  # x轴命名表示
				plt.ylabel("需水量")  # y轴命名表示
				plt.axis([0, 5, 0, 100])  # 设定x轴 y轴的范围
				plt.title("实际值与预测值折线图")
				plt.legend()  # 增加图例
				plt.show()  # 显示图片
			print("svr score:", MSLE(test_list, pre_list_svr, flag=0)[0])
			print("svr loss:", MSLE(test_list, pre_list_svr, flag=0)[1])
			print("svr_accuracy", explained_variance_score(test_list, pre_list_svr))
			print("rand score:", MSLE(test_list, pre_list_rand, flag=0)[0])
			print("rand loss:", MSLE(test_list, pre_list_rand, flag=0)[1])
			print("rand_accuracy", explained_variance_score(test_list, pre_list_rand))
			print("xgb score:", MSLE(test_list, pre_list_xgb, flag=0)[0])
			print("xgb loss:", MSLE(test_list, pre_list_xgb, flag=0)[1])
			print("xgb_accuracy", explained_variance_score(test_list, pre_list_xgb))
			# print("score:", MSLE(test_list, pre_list, flag=0))
def model_test(data):
	orinal_data = data
	# 在原来的基础上增加以下四列
	orinal_data["xgboost"] = 0
	orinal_data["svr"] = 0
	# orinal_data["bagging"] = 0
	orinal_data["randomtree"] = 0
	model_list = {
				  "svr": SVR(),
				  "randomtree": RandomForestRegressor(random_state=2022),
				  "xgboost":xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=100)
				  }
	test_list = ["test1", "test2", "test3", "test4"]
	estimators = [
		( 'xgboost',xgb.XGBRegressor(max_depth=6, learning_rate=0.05, n_estimators=100)),
		('svr', SVR()),
		('randomtree', RandomForestRegressor(random_state=2022))
	]
	# 集成学习模型
	reg = StackingRegressor(
		estimators=estimators,
		final_estimator=LinearRegression(),
		cv=5
	)

	for i, time in enumerate(['2022-05-08 00:00:00', '2022-06-08 00:00:00', '2022-07-28 00:00:00', '2022-08-28 00:00:00']):
		print(f"+++++++++++++++++{time}+++++++++++++++++++++")
		data_ = orinal_data[orinal_data["time"] <= time]
		feat_data, feat = fueat(data_)
		# 对于05-08，这里的数据集为03-01到05-08
		train_ = feat_data[feat_data['time'] >= '2022-03-01 01:00:00']
		# 对其进行标准化
		stand = StandardScaler()
		stand_train = stand.fit_transform(train_[feat])
		# 将标准化的数据胡转化为dataframe
		stand_df = pd.DataFrame(stand_train, columns=feat, index=train_.index)
		# 将不是test的划分到train，因为还是那个上面划分数据集的时候把test和train都当成了train_（已知的）
		train = train_[train_["train or test"] != test_list[i]]
		# 将时test的划分到test，这里的test就是最后要预测的结果（未知的）
		test = train_[train_["train or test"] == test_list[i]]
		# 存储预测结果
		pre_final = []
		for id in [
			"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10",
			"flow_11",
			"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
		]:
			print(f"+++++++++++++++++{id}+++++++++++++++++++++")
			# 对不同的flow进行划分为不同的train和预测的，这里的训练集就是就训练集，不会再对训练集进行划分为训练集和测试集，利用训练集来预测要求预测的数据
			id_train = train[train["flow_id"] == id]
			id_test = test[test["flow_id"] == id]
			X_train, y_train = stand_df.loc[id_train.index, :], id_train["flow"]
			X_test = stand_df.loc[id_test.index, :]
			pre_ser = np.zeros((id_test.shape[0],))  # 存储预测结果
			# 由于前几个模型效果较好，因此利用多个模型进行预测，预测结果取平均值
			for key, model in model_list.items():
				model.fit(X_train, y_train)
				pre = model.predict(X_test)
				# 存储的是平均预测结果
				pre_ser += pre/len(model_list)
				# 将预测结果加入到原来的数据集中，目的是为了后面的预测，可以利用到该预测结果
				orinal_data.loc[X_test.index, key] = pre
			pre_final.extend(pre_ser.tolist())
			# 利用集成学习进行预测
			# reg.fit(X_train, y_train)
			# pre = reg.predict(X_test)
			# pre_final.extend(pre)
		# 将预测结果拼接回原始数据，目的是为了后面的预测，可以利用到该预测结果
		orinal_data.loc[test.index, "flow"] = pre_final
	orinal_data.to_csv(path.predict_data_path + "orinal_data.csv", index=False)
	# 保存提交结果
	# 选取test集，产生提交结果
	all_data = orinal_data[orinal_data["train or test"] != "train"]
	flow_id = [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10",
		"flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]
	sub = pd.DataFrame()
	for i, flow in enumerate(flow_id):
		temp = all_data[all_data["flow_id"] == flow].reset_index(drop=True)
		if i == 0:
			sub = temp.loc[:, ["time", "flow"]]
		else:
			sub = pd.concat([sub, temp.loc[:, ["flow"]]], axis=1)
	sub.columns = ["time",
				   "flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9",
				   "flow_10",
				   "flow_11",
				   "flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
				   ]
	sub.to_csv(path.predict_data_path + "sub.csv", index=False)

if __name__ == '__main__':

	data = pd.read_csv(path.produce_data_path+"all_data_new.csv")
	# 训练数据，看准确度,用的是集成学习
	train(data,0)
	# 预测，并生成结果
	model_test(data)