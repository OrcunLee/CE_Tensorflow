import tensorflow as tf
import pandas as pd

Irese= "https://raw.githubusercontent.com/Hot6x/Tensorflow_CE/main/Data/Irese.csv"
Irese = pd.read_csv(Irese)
print(Irese.shape)
print(Irese.columns)
print(Irese.head(10))
print(Irese.tail(10))

인코딩 = pd.get_dummies(Irese)
인코딩.head()

독립 = 인코딩[['petal_length', 'petal_width', 'calyx_length', 'calyx_width']]
종속 = 인코딩[['kind_setosa', 'kind_versicolor', 'kind_virginica']]
print(독립.shape, 종속.shape)

X = tf.keras.layers.Input(shape=[4])
H = tf.keras.layers.Dense(8, activation="swish")(X)
H = tf.keras.layers.Dense(8, activation="swish")(H)
H = tf.keras.layers.Dense(8, activation="swish")(H)
Y = tf.keras.layers.Dense(3, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

model.fit(독립, 종속, epochs=1000, verbose=0)
model.fit(독립, 종속, epochs=10)


print(model.predict(독립[:5]))
print(종속[:5])

print(model.predict(독립[-5:]))
print(종속[-5:])

# weights & bias 출력
print(model.get_weights())
