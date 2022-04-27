import pandas as pd

Lemonade = pd.read_csv(
    "C:/Users/yzz07/Desktop/PROGRAMMING/Training_Tensorflow/Lemonade.csv")
Boston = pd.read_csv(
    "C:/Users/yzz07/Desktop/PROGRAMMING/Training_Tensorflow/Boston.csv")
Irese = pd.read_csv(
    "C:/Users/yzz07/Desktop/PROGRAMMING/Training_Tensorflow/Irese.csv")

print(Lemonade.shape)
print(Lemonade.columns)

print(Boston.shape)
print(Boston.columns)

print(Irese.shape)
print(Irese.columns)

독립 = Lemonade[['온도']]
종속 = Lemonade[['판매량']]
print(독립.shape, 종속.shape)

독립 = Boston[['crim', 'zn', 'indus', 'chas', 'nox',
            'rm', 'age', 'dis', 'rad', 'tax',
             'ptratio', 'b', 'lstat']]
종속 = Boston[['medv']]
print(독립.shape, 종속.shape)

독립 = Irese[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
종속 = Irese[['품종']]
print(독립.shape, 종속.shape)

print(Lemonade.head())
print(Boston.head())
print(Irese.head())
