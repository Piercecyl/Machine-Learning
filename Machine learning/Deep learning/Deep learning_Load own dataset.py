import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "D:\python\PetImages"
CATEGORIES = ["Dog", "Cat"]




#讀檔:http://johnliutw.logdown.com/posts/1726445
#創建Training data，這裡只在做將檔案內資料全部當成training data
training_data = []
def create_training_data():
    for category in CATEGORIES:  #將路徑內檔案讀出(cat & dog)，os.path.join => 用於改變檔案路徑

        path = os.path.join(DATADIR,category)  #Dog D:\python\PetImages\(Dog or cat)  
        class_num = CATEGORIES.index(category)  # 將Dog = 1,Cat = 0進行分類

        for img in tqdm(os.listdir(path)):  # tqdm是可以簡易產生進度條(COOL!)#listdir( )這個function，會找遍傳進去的路徑底下的所有檔案。 #因此就可以使用for in迴圈把所有的檔案印出來， 變數img：0.jpg 
            try:                            # try...except是指當try內錯誤時，會有更友善的except介面(這裡怕圖檔讀不進等等錯誤)
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # 呼叫 cv2.imread 即可將圖片讀取進來，cv2.IMREAD_GRAYSCALE:灰階
                new_array = cv2.resize(img_array, (150, 150))   # resize
                training_data.append([new_array, class_num])    # add this to our training_data  
    
            except Exception as e:  # 當錯誤，直接pass
                pass



create_training_data()


#目前問題：1. 資料輸進去會是全部都是Dog再來全部都是Cat
#         2. dog 和 cat數量不一致
import random
random.shuffle(training_data)
for sample in training_data[:10]: #隨機給予編號(0、1)
    print(sample[1])

#變成1*X的矩形
X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)
print(X[0].reshape(-1, 150, 150, 1)) #?????????????????????????????????
X = np.array(X).reshape(-1, 150, 150, 1)


#將training data存起來，方便日後使用，使用模組：pikle
import pickle

pickle_out = open("X.pickle","wb") #open('模組名稱' , 'wb = 電腦懂得形式')
pickle.dump(X, pickle_out) #dump
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#將存起來的資料讀取
pickle_in = open("X.pickle","rb") #rb = read
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)









