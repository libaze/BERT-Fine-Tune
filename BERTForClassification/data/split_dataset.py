# -*- coding: utf-8 -*-
"""
@Author : LIBAZE
@Time   : 2024/11/23 21:49
@File   : split_dataset.py
@desc   : 
"""
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    df = pd.read_csv('./waimai_10k.csv')
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=666)
    print(train_df)
    print(temp_df)
    dev_df, test_df = train_test_split(temp_df, test_size=0.6, random_state=666)
    print(dev_df)
    print(test_df)
    train_df.to_csv('train.csv', index=False)
    dev_df.to_csv('dev.csv', index=False)
    test_df.to_csv('test.csv', index=False)

