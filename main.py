import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)


def parse_engine(engine_str):
    if pd.isna(engine_str):
        return np.nan, np.nan, np.nan, 0, 0

    is_electric = 0
    is_hybrid = 0

    electric_match = re.search(r'Electric|Dual Motor|Battery', engine_str, re.IGNORECASE)
    hybrid_match = re.search(r'Hybrid|Gas/Electric|Electric/Gas|Plug-In', engine_str, re.IGNORECASE)
    if hybrid_match:
        is_hybrid = 1
    elif electric_match:
        is_electric = 1

    hp = np.nan
    hp_match = re.search(r'(\d+\.?\d*)\s*HP', engine_str, re.IGNORECASE)
    if hp_match:
        hp = float(hp_match.group(1))

    liters = np.nan
    liters_match = re.search(r'(\d+\.?\d*)\s*(L|Liter)', engine_str, re.IGNORECASE)
    if liters_match:
        liters = float(liters_match.group(1))
    else:
        liters_match_pre = re.search(r'(\d+\.?\d*)\s*L/', engine_str, re.IGNORECASE)
        if liters_match_pre:
            liters = float(liters_match_pre.group(1))

    cyl = np.nan
    cyl_match = re.search(r'(\d+)\s*Cylinder', engine_str, re.IGNORECASE)
    if cyl_match:
        cyl = int(cyl_match.group(1))
    else:
        cyl_patterns = [
            r'V-?(\d+)', r'I-?(\d+)', r'Flat-?(\d+)', r'H-?(\d+)', r'W-?(\d+)'
        ]
        for pat in cyl_patterns:
            match = re.search(pat, engine_str, re.IGNORECASE)
            if match:
                cyl = int(match.group(1))
                break

    return hp, liters, cyl, is_electric, is_hybrid

def parse_transmission(trans):
    if pd.isna(trans):
        return np.nan, np.nan

    t = str(trans).upper().replace('-', ' ')

    speeds = np.nan
    speeds_match = re.search(r'(\d+)\s*speed', t, re.IGNORECASE)
    if speeds_match:
        speeds = int(speeds_match.group(1))
    else:
        speeds_match2 = re.search(r'^(\d+)\s*speed', t, re.IGNORECASE)
        if speeds_match2:
            speeds = int(speeds_match2.group(1))

    transmission_type = 'unknown'
    if any(x in t for x in ['A/T', 'AUTO', 'AUTOMATIC', 'AT/MT']):
        transmission_type = 'automatic'
    elif any(x in t for x in ['M/T', 'MANUAL']):
        transmission_type = 'manual'
    elif any(x in t for x in ['CVT', 'CVT-F', 'F']):
        transmission_type = 'cvt'

    return speeds, transmission_type


df = pd.read_csv('data.csv')
print("Описание датасета:")
df.info()
print("\nДубликаты:", df.duplicated().sum())
miss = df.isnull().sum()
miss_percent = (miss[miss > 0] / len(df)) * 100
print(f"Пропуски по столбцам:\n {pd.DataFrame({'количество пропусков': miss[miss > 0],
                                               'процент': miss_percent}).sort_values('процент', ascending=False)}")
print(f"\nСредняя цена: {df['price'].mean():.0f}")
print(f"Медианная цена: {df['price'].median():.0f}")
print(f"Min цена: {df['price'].min()}, Max цена: {df['price'].max()}\n")

df['car_age'] = 2025 - df['model_year']
df['log_price'] = np.log(df['price'])

cat_cols = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} уникальных значений")


results = df['engine'].apply(parse_engine)
df[['hp', 'liters', 'cylinders', 'is_electric', 'is_hybrid']] = pd.DataFrame(results.tolist(), index=df.index)
print("\nПропуски после парсинга двигателей:")
print(df[['hp', 'liters', 'cylinders', 'is_electric', 'is_hybrid']].isnull().sum())
print(df[['hp', 'liters', 'cylinders', 'is_electric', 'is_hybrid']].describe())

df[['trans_speeds', 'trans_type']] = df['transmission'].apply(
    lambda x: pd.Series(parse_transmission(x))
)
print("\nПропуски после парсинга трансмиссий:")
print(df[['trans_type', 'trans_speeds']].isnull().sum())
print(f'\n{df['trans_type'].value_counts()}')
print(f'\n{df['trans_speeds'].describe()}')

num_cols_ext = ['car_age', 'milage', 'hp', 'liters', 'cylinders', 'log_price',
                'is_electric', 'is_hybrid', 'trans_speeds']
corr_matrix_ext = df[num_cols_ext].corr()

# Графики
plt.figure()
plt.subplot(1, 2, 1)
sns.histplot(df['log_price'], bins=100, kde=True, color='green')
plt.title('Распределение log(price)')

plt.subplot(1, 2, 2)
sns.boxplot(df['log_price'], color='lightgreen')
plt.title('Боксплот log(price)')
plt.savefig('log(price).png', dpi=600)

plt.figure()
sns.heatmap(corr_matrix_ext, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица')
plt.savefig('corr.png', dpi=600)

plt.show()




print(f"\nБыло строк: {len(df)}")

q_low = df["price"].quantile(0.005)
q_high = df["price"].quantile(0.995)

mask = df["price"].between(q_low, q_high)
df = df[mask].copy()

df['log_price'] = np.log(df['price'])

print(f"Стало строк после удаления выбросов: {len(df)}")
print(f"Цена от {df['price'].min():,} до {df['price'].max():,}")
print(f"log_price от {df['log_price'].min():.3f} до {df['log_price'].max():.3f}\n")

y = df['log_price']
X = df.drop(['id', 'price', 'log_price', 'model_year', 'engine', 'transmission'], axis=1)

top_models = X['model'].value_counts().head(60).index
X['model'] = X['model'].where(X['model'].isin(top_models), 'Other')

main_colors = X['ext_col'].value_counts().head(6).index
X['ext_col'] = X['ext_col'].where(X['ext_col'].isin(main_colors), 'Other')
X['int_col'] = X['int_col'].where(X['int_col'].isin(X['int_col'].value_counts().head(4).index), 'Other')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_features = ['car_age', 'milage', 'hp', 'liters', 'cylinders', 'trans_speeds']
cat_features = ['brand', 'model', 'fuel_type', 'ext_col', 'int_col',
                'accident', 'clean_title', 'trans_type',
                'is_electric', 'is_hybrid']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

lasso_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lasso', Lasso(max_iter=10000, random_state=42, warm_start=True))
])

alphas = np.logspace(-4, -2, 20)

param_grid = {'lasso__alpha': alphas}

grid = GridSearchCV(
    lasso_pipe,
    param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print(f"Лучший alpha: {grid.best_params_['lasso__alpha']:.6f}")
print(f"Лучший CV RMSE (log): {-grid.best_score_:.4f}")

best_model = grid.best_estimator_
y_pred_log = best_model.predict(X_test)
y_pred = np.exp(y_pred_log)
y_true = np.exp(y_test.values)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# print("y_pred (первые 5):    ", y_pred[:5])
# print("y_true (первые 5):    ", y_true[:5])

print("\nФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ LASSO")
print(f"RMSE:      ${rmse:,.0f}")
print(f"R²:        {r2:.4f}")
print(f"Mean %: {mape:.1f}%")
print(f"Средняя цена:   ${y_true.mean():,.0f}")