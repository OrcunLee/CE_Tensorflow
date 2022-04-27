#%% 라이브러리
import tensorflow as tf
import pandas as pd

#%% 
보스턴 = pd.read_csv("C:/Users/yzz07/Desktop/PROGRAMMING/Training_Tensorflow/보스턴.csv")
print(보스턴.shape)
print(보스턴.columns)
print(보스턴.head())

# %%
독립 = 보스턴[['crim', 'zn', 'indus', 'chas', 'nox', 
            'rm', 'age', 'dis', 'rad', 'tax',
            'ptratio', 'b', 'lstat']]
종속 = 보스턴[['medv']]
print(독립.shape, 종속.shape)

# %% 모델 생성
X = tf.keras.layers.Input(shape=[13])
H = tf.keras.layers.Dense(10, activation='swish')(X)
Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# %% 모델을 학습
model.fit(독립, 종속, epochs=1000, verbose=0)   # verbose = 0 은 화면출력 안한다는 명령
model.fit(독립, 종속, epochs=10)    # epochs 학습활 횟수
#%%
model.summary()
# %%
print(model.predict(독립[5:10]))  # 독립 변수일때 종속변수 값 예측
print(종속[5:10]) # 종속변수 확인

# %% 모델의 수식 확인
print(model.get_weights())