from .utils import timeit, log
import pandas as pd
from turbodbc import connect
import os
from sklearn.model_selection import train_test_split


class GeneralMethods:
    """
    Class with features sets and general usage methods for model building at every stage.

    Attributes:
        data_version (int): version for directory generation and data storage
        directory (str): directory to store model data
        connect: current connection to Teradata
        cursor: cursor provided by connection
        dmsc_col_names (list of str): list of all NO_DATA columns
        meta_vars (set of str): set of meta columns
        bad_vars (set of str): set of bad columns to not use them as features
        kurt_sk_vars (set of str): set of kurtosis and skew columns
        cat_vars (set of str): set of categorical columns with several categories
        bin_vars (set of str): set of categorical columns with two categories
        all_cats (set of str): set of all categorical columns (bin_vars + cat_vars)
        wcat_vars (set of str): set of columns with wrong data types after download from database
        vars (list of str): list of features (w/o 'age' and similar)
    """
    def __init__(self, data_version: int = 1):
        """
        Constructor for GeneralMethods class.

        Args:
            data_version: data version to specify data folder name
        """

        self.data_version = data_version
        self.directory = 'data_v' + str(self.data_version)
        self.connect = None
        self.cursor = None
        self.dmsc_col_names = [
            'NO_DATA']

        self.meta_vars = {
            'NO_DATA'}

        self.bad_vars = {
            'NO_DATA'}

        self.kurt_sk_vars = {
            'NO_DATA'}

        self.cat_vars = {
            'NO_DATA'}

        self.bin_vars = {
            'NO_DATA'}

        self.all_cats = self.cat_vars.union(self.bin_vars)

        self.wcat_vars = {
            'NO_DATA'}

        self.vars = (set(self.dmsc_col_names + ['label'])
                     - self.meta_vars
                     - self.bad_vars
                     - self.kurt_sk_vars)

        self.vars = [var for var in self.vars if ('age' not in var)]

    @timeit
    def wcat_features_fix(self, dt: pd.DataFrame) -> pd.DataFrame:
        """
        Fix wcat_vars allocated in df.

        Args:
            dt (DataFrame): pandas DataFrame to transform

        Returns:
            dt (DataFrame)
        """
        for column in self.wcat_vars:
            if column in dt.columns:
                dt[column] = dt[column].astype('str')
                dt[column] = dt[column].str.replace(',', '.')
                dt[column] = dt[column].astype('float32')
        return dt

    def create_data_directory(self):
        """
        Creates directory to store data and model, if it`s not exists.
        """
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    @timeit
    def establish_connection(self, dsn: str = 'NO_DATA'):
        """
        Establish connection for specified dsn with autocommit = True.
        """
        self.connect = connect(dsn=dsn)
        self.cursor = self.connect.cursor()
        self.connect.autocommit = True
    
    @staticmethod
    @timeit
    def split(dt: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        Perform train/test split with reset_index = True for specified df.

        Args:
            dt (DataFrame): df to split

        Returns:
            df_train, df_test (DataFrame, DataFrame): splitted parts of df
        """
        df_train, df_test = train_test_split(dt, test_size=0.20, random_state=333)
        df_train.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)
        return df_train, df_test

    @timeit
    def cats_fit_transform(self, dt: pd.DataFrame, code_dict: dict, code_vars: list or set) -> (pd.DataFrame, dict):
        """
        Apply label encoding fit and transform for specified df and code_vars.

        Args:
            dt (DataFrame): data frame to fit encoding dict and then transform
            code_dict (dict): dict to store codes
            code_vars (list/set): list of features to encode

        Returns:
            dt, code_dict (DataFrame, dict): transformed df and code_dict with stored codes
        """
        dt_vars = [var for var in code_vars if var in dt.columns]

        for var in dt_vars:
            dt.loc[:, var] = dt[var].astype('category')
            code_dict[var] = dt[var].cat.categories
            dt.loc[:, var] = dt[var].cat.codes

        return dt, code_dict

    @timeit
    def cats_transform(self, dt: pd.DataFrame, code_dict: dict) -> pd.DataFrame:
        """
        Apply label encoding to df according to code_dict.

        Args:
            dt (DataFrame): data frame to category features transform
            code_dict (dict): dict with stored codes

        Returns:
            dt (DataFrame): transformed data frame
        """
        for var, _ in code_dict.items():
            if var in dt.columns:
                dt.loc[:, var] = pd.Categorical(dt[var].values, categories=code_dict[var])
                dt.loc[:, var] = dt[var].cat.codes
        return dt

    @timeit
    def download_data(self, file_path: str = '', query: str = '') -> pd.DataFrame:
        """
        Tries to download df from file, in case of error - from Teradata and then save to file.

        Args:
            file_path (str): path including filename to download or save
            query (str): SQL Teradata query

        Returns:
            dt (DataFrame): downloaded data frame
        """
        try:
            dt = pd.read_feather(file_path)
        except IOError:
            print('File {} not found and will be downloaded\n '.format(file_path))
            self.establish_connection()
            self.cursor.execute(query)
            dt = pd.DataFrame(self.cursor.fetchallnumpy())
            dt.columns = [col.lower() for col in dt.columns]
            self.create_data_directory()
            dt.to_feather(file_path)
            self.connect.close()
        return dt
