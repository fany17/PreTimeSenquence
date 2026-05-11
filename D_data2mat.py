import pandas as pd
from scipy.io import savemat
filename = 'LSTMdata.pkl'
# filename = 'market_data_DOGE_new_with_position.pkl'
df = pd.read_pickle(filename)

# 将 DataFrame 转换为字典，方便保存为 .mat 文件
data_dict = {col: df[col].values for col in df.columns}

# 使用 scipy.io.savemat 保存为 mat 文件
savemat("LSTMdata.mat", data_dict)

# savemat("LSTMdata_test.mat", data_dict)
