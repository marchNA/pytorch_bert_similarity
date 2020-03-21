import pandas as pd
import os

data_path = os.getcwd()

def get_demo():

    train_path = os.path.join(data_path, 'train.csv')
    train_df = pd.read_csv(train_path, encoding='utf-8')
    train_df[:1000].to_csv(os.path.join(data_path, 'train_demo.csv'), index=0)

    dev_path = os.path.join(data_path, 'dev.csv')
    dev_df = pd.read_csv(dev_path, encoding='utf-8')
    dev_df[:100].to_csv(os.path.join(data_path, 'dev_demo.csv'), index=0)

    test_path = os.path.join(data_path, 'test.csv')
    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df[:100].to_csv(os.path.join(data_path, 'test_demo.csv'), index=0)

get_demo()