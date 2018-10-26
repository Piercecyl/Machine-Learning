import pandas as pd
import quandl ,math 
import numpy as np
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#這裡以Google先前股票歷史資料為例
df = quandl.get("WIKI/GOOGL")

#資料前處理(選取裡面較有用的資料)
#The features(=X) are the descriptive attributes, and the label(=Y) is what you're attempting to predict or forecast.
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
#print(df.head())

#在這裡什麼是Feature什麼是Label?(題目為預測未來股市價格) => 我們應該用目前價格當作Feature(=X)來預測未來價格(=Y)，未來價格是Label
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True) #缺失值填入-99999，當然可以用其他數值代替
forecast_out = int(math.ceil(0.01 * len(df))) #ceil回傳無條件進位 =>不會有小數點
df['label'] = df[forecast_col].shift(-forecast_out) #為何要將資料全部往上移35天? => 用於預測35天後的資料


#建立input值 X 與output值 Y
X = np.array(    df.drop(['label'] ,1)   ) #X指的是Feature： Adj. Close、HL_PCT、PCT_change、Adj. Volume 這四個會影響Label的Feature
y = np.array(df['label'])
print(X[:1] ,y[:1] ,df)

#Generally, you want your features in machine 
#learning to be in a range of -1 to 1. This may do nothing, but it usually speeds up processing and can also help with accuracy
# X = preprocessing.scale(X) 
# y = np.array(df['label'])   


#X_train,y_train:構成了訓練集
#X_test,y_test：構成了測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 )
clf = LinearRegression()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)


