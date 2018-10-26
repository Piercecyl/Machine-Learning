from sklearn.model_selection import train_test_split
train= loan_data.iloc[0: 55596, :]  
test= loan_data.iloc[55596:, :]  
# 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)  
train_X,test_X, train_y, test_y = train_test_split(train,  
                                                   target,  
                                                   test_size = 0.2,  
                                                   random_state = 0)  
train_y= train_y['label']  
test_y= test_y['label']  

print(train ,test ,train_y ,test_y)