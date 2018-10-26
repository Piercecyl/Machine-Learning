import tensorflow as tf
import matplotlib.pyplot as plt
#0~9手寫字(28 * 28)
mnist = tf.keras.datasets.mnist

#將dataset裡分成train & test data
(x_train ,y_train) , (x_test ,y_test) = mnist.load_data()

#歸一化(0~1) => 更容易執行
x_train = tf.keras.utils.normalize(x_train ,axis=1)
x_test = tf.keras.utils.normalize(x_test ,axis=1)

#Build the model
model = tf.keras.models.Sequential()

#可以看到NN是平面而非多維，這邊將數據拉平(從28*28變成 1*784)：此層為Input層
model.add(tf.keras.layers.Flatten(input_shape = (28,28)))

#建立 Hidden layer，使用簡單的Dense-Connect layer，總共有128個神經元，
# 使用RELU當作激勵函數，這邊使用兩層Hidden layer。
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#建立output layer，10代表數字0~9。這裡使用
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#接著最後是編譯我們的model，以下所選的function是比較基礎的function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#接著將data做訓練，epochs代表要做幾次訓練(並不是做越多越好)
model.fit(x_train, y_train, epochs=3)

#為了防止overfitting，我們必須用test data做測試
val_loss, val_acc = model.evaluate(x_test, y_test)
#print(val_loss)
#print(val_acc)

#可以將model存起來方便下次用
#model.save('model name')

model.save('epic_num_reader.model')
#Load it back
new_model = tf.keras.models.load_model('epic_num_reader.model')

#當確定完成後，我們可以用導入新的數據進行預測
prediction = new_model.predict(x_test)
print(prediction)

#視覺化
import numpy as np
print(np.argmax(prediction[0]))
plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()






