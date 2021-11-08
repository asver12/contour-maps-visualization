"""
@author: Philipp Lucas

This data set is taken from: https://github.com/hadley/data-fuel-economy
"""
import pandas as pd
import os

_csvfilepath = os.path.splitext(__file__)[0] + ".csv"

# that should about be all useful variables:
_cols = ['year', 'car_size', 'displacement', 'cylinder', 'mpg_city', 'mpg_highway', 'drv', 'transmission', 'turbo']

# temporal variables:
_temp_cols = ['year']
# year is first year when car has been sold

# numerical variables
_num_cols = ['displacement', 'cylinder', 'mpg_city', 'mpg_highway']
# displ is displacement of engine
# cyl is number of cylinders
# cty is miles per gallon in city
# hwy is miles per gallon on highway

# categorical variables:
_cat_cols = ['car_size', 'drv', 'transmission', 'turbo']
# class is ""compact car" and such
# drv is RWD or FWD
# trans is transmission type


def _read(file=_csvfilepath):
    df = pd.read_csv(file)
    df = pd.DataFrame(df, columns=['trans', 'cyl', 'T', 'class', 'year', 'cty', 'hwy', 'displ'])
    df = df.rename(columns={'cty': 'mpg_city', 'hwy': 'mpg_highway', 'trans': 'transmission',
                            'T': 'turbo', 'cyl': 'cylinder', 'class': 'car_size', 'displ': 'displacement'})
    return df


def cg_4cat3cont(file=_csvfilepath, do_not_change=[]):
    df = _read(file)
    df = pd.DataFrame(df, columns=['transmission', 'cylinder', 'turbo', 'car_size', 'year', 'mpg_city', 'mpg_highway', 'displacement'])

    df.car_size.replace(to_replace=
                        {'pickup': 'large',
                         'suv': 'midsize',
                         'station wagon': 'midsize',
                         'compact': 'small',
                         'passenger van': 'large',
                         'cargo van': 'large',
                         'two seater': 'small',
                         'large car': 'large',
                         'midsize car': 'midsize',
                         'compact car': 'small'}, inplace=True)

    if 'cylinder' not in do_not_change:
        # df.cylinder.replace(to_replace={2: 'few', 3: 'few',
        #                                 4: 'medium', 5: 'medium',
        #                                 6: 'many', 7: 'many', 8: 'many', 10: 'many', 12: 'many', 16: 'many'}, inplace=True)
        df.cylinder.replace(to_replace={2: 'few', 3: 'few', 4: 'few',
                                        5: 'medium', 6: 'medium',
                                        7: 'many', 8: 'many', 10: 'many', 12: 'many', 16: 'many'}, inplace=True)

    df.replace(to_replace={'transmission': {
        '.*auto.*': 'auto'}}, inplace=True, regex=True)
    
    df.replace(to_replace={'transmission': {
        'lock-up.*': 'lock-up'}}, inplace=True, regex=True)
    
    df.replace(to_replace={'transmission': {
        'manual.*': 'manual'}}, inplace=True, regex=True)
    
    df = df.drop(df[df.transmission == 'semi-auto'].index)
    df = df.drop(df[df.transmission == 'creeper(C5)'].index)
    df = df.drop(df[df.car_size == 'spv'].index)
    df = df.dropna(axis=0)
    df.reset_index(drop=True, inplace=True)

    for col in ['transmission', 'cylinder', 'turbo', 'car_size']:
        df[col] = df[col].astype('str')

    del df['turbo']

    return df
