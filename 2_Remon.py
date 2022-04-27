
import tensorflow as tf
import pandas as pd


레모네이드 = pd.read_csv("C:/Users/yzz07/Desktop/PROGRAMMING/Training_Tensorflow/레모네이드.csv")
레모네이드.head()


독립 = 레모네이드[['온도']]
종속 = 레모네이드[['판매량']]
print(독립.shape, 종속.shape)


X = tf.keras.layers.Input(shape=[1]) # 독립변수 1개
Y = tf.keras.layers.Dense(1)(X) # 종속변수 1개 
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')


model.fit(독립, 종속, epochs=10000, verbose=0)   # verbose = 0 은 화면출력 안한다는 명령
model.fit(독립, 종속, epochs=10)    # epochs 학습활 횟수


print(model.predict(독립))  # 독립변수일때 종속변수 값 예측
print(model.predict([[15]])) # 15일경우 무슨값이 나올지 예측