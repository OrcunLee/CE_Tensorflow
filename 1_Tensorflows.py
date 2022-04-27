#%%
import pandas as pd


Lemonade = "https://raw.githubusercontent.com/Hot6x/Tensorflow_CE/main/Data/Lemonade.csv"
Lemonade = pd.read_csv(Lemonade)

Boston = "https://raw.githubusercontent.com/Hot6x/Tensorflow_CE/main/Data/Boston.csv"
Boston = pd.read_csv(Boston)

Irese= "https://raw.githubusercontent.com/Hot6x/Tensorflow_CE/main/Data/Irese.csv"
Irese = pd.read_csv(Irese)

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

독립 = Irese[['petal_length', 'petal_width', 'calyx_length', 'calyx_width']]
종속 = Irese[['kind']]
print(독립.shape, 종속.shape)

print(Lemonade.head())
print(Boston.head())
print(Irese.head())
# %%
