# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         data
# Description:
# Author:       xinzhuang
# Date:         2022/9/12
# Function:
# Version：
# Notice:
# -------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import path
from normal import abnormal_data, fill_nan

# 设置value的显示长度为200，默认为50
pd.set_option('max_colwidth', 200)
# 显示所有列，把行显示设置成最大
pd.set_option('display.max_columns', None)
# 显示所有行，把列显示设置成最大
pd.set_option('display.max_rows', None)

"""
整个数据 2022-01-01 01:00:00 开始
训练集划分为
test1 开始 2022-04-01 01:00:00 2160  结束 2022-04-08 00:00:00  2328
"""
"""
test1 开始 2022-05-01 01:00:00 2881  结束 2022-05-08 00:00:00  3048
test2 开始 2022-06-01 01:00:00 3625  结束 2022-06-08 00:00:00  3792
test3 开始 2022-07-21 01:00:00 4825  结束 2022-07-28 00:00:00  4992
test4 开始 2022-08-21 01:00:00 5569  结束 2022-08-28 00:00:00  5736
"""

def get_all_data():
	data_ = pd.read_csv(path.train_data_path + "hourly_dataset.csv")
	# time_index列的数据是从1开始到5736，这是属于新增加的一列
	data_["time_index"] = np.arange(1, data_.shape[0] + 1)
	# flow_id也是一列
	flow_id = [
		"flow_1", "flow_2", "flow_3", "flow_4", "flow_5", "flow_6", "flow_7", "flow_8", "flow_9", "flow_10", "flow_11",
		"flow_12", "flow_13", "flow_14", "flow_15", "flow_16", "flow_17", "flow_18", "flow_19", "flow_20"
	]
	all_data = pd.DataFrame()
	# 下面这个循环就是将数据变成横向排列，就是列名只有flow_id，然后一个flow占5000多行，总共20个，所以占11万多行
	for i, flow in enumerate(flow_id):
		if i == 0:
			data = data_.loc[:, ["time", "time_index", flow, "train or test"]].rename(columns={flow: "flow"})
			data["flow_id"] = flow
			all_data = data
		else:
			data = data_.loc[:, ["time", "time_index", flow, "train or test"]].rename(columns={flow: "flow"})
			data["flow_id"] = flow
			all_data = pd.concat([all_data, data], axis=0)
	all_data["day_time"] = list(map(lambda x: str(x)[:10], all_data["time"]))
	data_["day_time"] = list(map(lambda x: str(x)[:10], data_["time"]))
	# 读入天气数据以及疫情数据
	weather_data = pd.read_csv(path.train_data_path + "new_weather.csv")
	epi_data = pd.read_csv(path.train_data_path + "epidemic.csv").rename(columns={"jzrq": "day_time"})
    # 用0来填充疫情数据中的缺失值

	epi_data = epi_data.fillna(0)
	weather_data = weather_data.interpolate(method='nearest')


	# weather_data=weather_data.interpolate(method='nearest')
	# 将原来的小时数据和天气数据以及疫情数据进行合并
	all_data = all_data.merge(weather_data.loc[:, ["time", "R", "fx", "T", "U", "fs", "V", "P"]], on=["time"],
							  how="left")
	all_data = all_data.merge(epi_data.loc[:, ["day_time", "zz", "wz", "glzl", "yxgc", "xzqz", "xzcy", "xzsw"]],
							  on=["day_time"], how="left")
	data_ = data_.merge(weather_data.loc[:, ["time", "R", "fx", "T", "U", "fs", "V", "P"]], on=["time"],
						how="left")
	data_ = data_.merge(epi_data.loc[:, ["day_time", "zz", "wz", "glzl", "yxgc", "xzqz", "xzcy", "xzsw"]],
						on=["day_time"], how="left")
	del all_data["day_time"]
	del data_["day_time"]

    # 异常值识别，返回u正常的数据
	all_data = abnormal_data(all_data)
	all_data.to_csv(path.produce_data_path + "all_data.csv")
	# 处理缺失值的方法
	all_data = fill_nan(all_data)
	# all_data["time"] = pd.to_datetime(all_data["time"])
	# all_data["dayofyear"] = all_data["time"].dt.dayofyear

	data_.to_csv(path.produce_data_path + "hour_data.csv", index=False)
	all_data.reset_index(drop=True, inplace=True)
	return all_data


if __name__ == '__main__':
	all_data = get_all_data()
	all_data.to_csv(path.produce_data_path + "all_data_new.csv", index=False)