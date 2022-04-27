#%% 라이브러리
import tensorflow as tf
import pandas as pd

아이리스 = pd.read_csv("C:/Users/yzz07/Desktop/PROGRAMMING/Training_Tensorflow/아이리스.csv")
print(아이리스.shape)
print(아이리스.columns)
print(아이리스.head(10))
print(아이리스.tail(10))

인코딩 = pd.get_dummies(아이리스)
인코딩.head()

독립 = 인코딩[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
종속 = 인코딩[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(독립.shape, 종속.shape)

# %% 
X = tf.keras.layers.Input(shape=[4])
H = tf.keras.layers.Dense(8, activation="swish")(X)
H = tf.keras.layers.Dense(8, activation="swish")(H)
H = tf.keras.layers.Dense(8, activation="swish")(H)
Y = tf.keras.layers.Dense(3, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# %% 학습
model.fit(독립, 종속, epochs=1000, verbose=0)
model.fit(독립, 종속, epochs=10)

# %% 확인

# 맨 처음 데이터 5개
print(model.predict(독립[:5]))
print(종속[:5])
 
# 맨 마지막 데이터 5개
print(model.predict(독립[-5:]))
print(종속[-5:])

# %%
# weights & bias 출력
print(model.get_weights())

# %%
