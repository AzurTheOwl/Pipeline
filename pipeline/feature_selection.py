import lightgbm as lgb
import seaborn as sns
from eli5.permutation_importance import get_score_importances
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from .utils import timeit, log
from .general_methods import GeneralMethods


class FeatureSelection(GeneralMethods):
    """
    Class for all feature selection actions - from data loading to top-features saving.

    Attributes:
        report_date (str): train data report date
        table_name (str): name of table with ids and labels

    """

    def __init__(self,
                 table_name: str = '',
                 report_date: str = '',
                 data_version: int = 1,
                 top_amount: int = 30
                 ):
        """
        Constructor for FeatureSelection class.

        Args:
            table_name (str): name of table with ids and labels
            report_date (str): report date for default data downloading
            data_version (int): version for directory generation and data storage
            top_amount (int): top-features amount to select and save
        """
        self.data_version = data_version
        self.directory = 'data_v' + str(self.data_version)
        self.report_date = report_date
        super().__init__(self.data_version)

        self.table_name = table_name
        self.top_amount = top_amount

        self.bin_dict = {}
        self.cat_dict = {}

        self.top_features = []
        self.top_cats = []

        self.ext_features = {}
        self.full_ext_features = {}

        self.top_features_file = ''
        self.ext_features_file = ''

        self.replacers = {'table_name': self.table_name,
                          'report_date': self.report_date,
                          'directory': self.directory}

    @timeit
    def bad_features_postfilter(self, dt: pd.DataFrame, ex_cl: bool = True, ex_bc: bool = True) -> pd.DataFrame:
        """
        Additional filter for bad features from NO_DATA.

        Args:
            dt (DataFrame): data frame to exclude features
            ex_cl (bool): keep contact list features flag
            ex_bc (bool): keep branch count features flag

        Returns:
            dt (DataFrame): data frame with dropped features
        """
        cols_to_exclude = []
        if ex_cl:
            cl_cols = [col for col in self.dmsc_col_names if 'cl_' in col]
            cols_to_exclude += cl_cols
        if ex_bc:
            bc_cols = [col for col in self.dmsc_col_names if 'bc_' in col]
            cols_to_exclude += bc_cols
        kurt_cols = [col for col in self.dmsc_col_names if 'kurt_' in col]
        sd_cols = [col for col in self.dmsc_col_names if 'sd_' in col]
        sk_cols = [col for col in self.dmsc_col_names if 'sk_' in col]
        sms_cols = [col for col in self.dmsc_col_names if ('sms_' in col)
                    & (col not in ['NO_DATA'])]

        cols_to_exclude += kurt_cols + sd_cols + sk_cols + sms_cols
        dt.drop([col for col in dt.columns if col in cols_to_exclude], axis=1, inplace=True)

        return dt

    def download_data(self, tablename: str = '', sub_query: str = None, aff: str = None) -> pd.DataFrame:
        """
        Tries to download df from file, in case of error - from Teradata and then save to file.

        Args:
            tablename (str): table name to download from
            sub_query (str): sub-query for labels and subs_id
            aff (str): file affix to form filename

        Returns:
            dt (DataFrame): downloaded data frame
        """
        if sub_query is None:
            sub_query = """
                SELECT subs_id, label
                from {table_name}
                where report_date = DATE'{report_date}'
                """.format(**self.replacers)

        if (not tablename) or (tablename.lower() == 'NO_DATA'):
            tablename = 'NO_DATA'

            query = "SELECT b.*, " + ', '.join(self.vars) + \
                    """
                    FROM {} a
                    join ({}) b
                    on a.subs_id = b.subs_id
                    WHERE a.REPORT_DATE = DATE'{}'
                    """.format(tablename, sub_query, self.report_date)
        else:
            query = "SELECT a.* " + \
                    """
                    FROM {} a
                    join ({}) b
                    on a.subs_id = b.subs_id
                    WHERE a.REPORT_DATE = DATE'{}'
                    """.format(tablename, sub_query, self.report_date)

        if aff is None:
            aff = self.table_name.split('.')[1]

        file_path = r'{}/{}_{}_{}.feather'.format(
            self.replacers['directory'],
            aff,
            self.replacers['report_date'],
            tablename.split('.')[1]
            )

        dt = super().download_data(file_path, query)

        return dt

    @timeit
    def process_data(self, dt: pd.DataFrame, ex_cl: bool = True, ex_bc: bool = True) -> pd.DataFrame:
        """
        Applies postfilter, fix and encoding to data frame.

        Args:
            dt (DataFrame): data frame to apply fixes and transform variables
            ex_cl (bool): keep contact list features flag
            ex_bc (bool): keep branch count features flag

        Returns:
            dt (DataFrame): processed data frame
        """
        dt = self.bad_features_postfilter(dt, ex_cl, ex_bc)
        dt = super().wcat_features_fix(dt)
        if not self.bin_dict:
            dt, self.bin_dict = self.cats_fit_transform(dt, code_dict=self.bin_dict, code_vars=self.bin_vars)
        else:
            dt = self.cats_transform(dt, code_dict=self.bin_dict)
        if not self.cat_dict:
            dt, self.cat_dict = self.cats_fit_transform(dt, code_dict=self.cat_dict, code_vars=self.cat_vars)
        else:
            dt = self.cats_transform(dt, code_dict=self.cat_dict)
        return dt

    def save_top(self, n_features: int = None):
        """
        Saves top n selected features.

        Args:
            n_features (int): number of features to save
        """
        if n_features is None:
            n_features = self.top_amount
        self.top_features_file = r'{}/top_{}_features.npy'.format(
            self.replacers['directory'],
            n_features)
        np.save(self.top_features_file, self.top_features[:n_features])
        print('Top features saved to file.')

    def save_ext(self):
        """
        Saves external features which contained in top n selected features.
        """
        for table in list(self.ext_features):
            items = [item for item in self.ext_features[table] if item in self.top_features]
            if not len(items):
                self.ext_features.pop(table)
            else:
                self.ext_features[table] = items
        self.ext_features_file = r'{}/ext_features.npy'.format(
            self.replacers['directory'])
        np.save(self.ext_features_file, self.ext_features)
        print('External features saved to file.')

    def add_ext_features(self, dt: pd.DataFrame, tables: list, sub_query: str = None, aff: str = None) -> pd.DataFrame:
        """
        Adds external features to specified data frame.
        Args:
            dt (DataFrame): data frame to fill with external features
            tables (list of str): list of tables to download and merge with dt
            sub_query (str): sub-query for labels and subs_id
            aff (str): file affix to form filename

        Returns:
            dt (DataFrame): data frame with added external features
        """
        dt = dt.copy(deep=True)
        for tablename in tables:
            df = self.download_data(tablename=tablename, sub_query=sub_query, aff=aff)
            df = df.drop('report_date', axis=1)
            if tablename not in self.full_ext_features.keys():
                self.full_ext_features[tablename] = [feature for feature in df.columns if feature not in self.meta_vars]
            dt = dt.merge(df, on='subs_id', how='left')
        return dt

    @timeit
    def dmx_check(self, dt: pd.DataFrame,
                  tables: list = None,
                  params: dict = None,
                  how: str = 'pair',
                  perm: bool = True):
        """
        Performs DMX mart feature selection for given tables list.

        Args:
            dt (DataFrame): data frame to use for ids and labels or to pair with
            tables (list): list of DMX parts to check
            params (dict): LGBMClassifier parameter dictionary
            how (str): 'only' to perform feature selection only for DMX part table,
                'pair' to select features on DMSC + DMX part,
                'all' to use all specified DMX part + DMSC simultaneously
            perm (bool): use permutation importance flag
        """
        if tables is None:
            tables = []

        if how == 'pair':
            for table in tables:
                log('Currently on checking: {}'.format(table))
                df = dt.loc[:, ['subs_id'] + list(self.vars) + self.full_ext_features[table]]
                self.feature_selection(df, params, perm=perm)
                table_features = [feature for feature in self.top_features
                                  if (feature not in self.vars)]
                # print(table_features)
                self.ext_features[table] = table_features

        elif how == 'only':
            for table in tables:
                log('Currently on checking: {}'.format(table))
                df = dt.loc[:, ['subs_id', 'label'] + self.full_ext_features[table]]
                self.feature_selection(df, params, perm=perm)
                table_features = [feature for feature in self.top_features
                                  if (feature not in self.vars)]
                # print(table_features)
                self.ext_features[table] = table_features

        elif how == 'all':
            log('All specified DMX parts will be checked simultaneously')
            dt = dt.copy(deep=True)
            self.feature_selection(dt, params, perm=perm)
            for table in tables:
                table_features = [feature for feature in self.top_features
                                  if (feature not in self.vars) and (feature in self.full_ext_features[table])]
                # print(table_features)
                self.ext_features[table] = table_features

    @timeit
    def feature_selection(self, dt: pd.DataFrame,
                          params: dict = None,
                          drop_list: list = None,
                          perm: bool = True,
                          use_ext: bool = False) -> pd.DataFrame:
        """
        Performs feature selection for given data frame.

        Refresh object`s features and categorical features, than perform cross-validation, calculate importances and
        do cross-validation again for top selected features. Then plot importances.

        Args:
            dt (DataFrame): data frame to feature selection on
            params (dict): LGBMClassifier parameters dictionary
            drop_list (list): list to explicitly drop feature before selection
            perm (bool): use permutation importance flag
            use_ext (bool): use external (DMX) features flag

        Returns:
            res (DataFrame): feature selection results
        """
        if drop_list is None:
            drop_list = []

        if params is None:
            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': 16,
                'is_unbalance': True,
                # 'max_depth': 4,
                'learning_rate': 0.05,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 1,
                'random_state': 321
            }

        if use_ext:
            for table in self.ext_features.keys():
                drop_list += list(set(self.full_ext_features[table]) - set(self.ext_features[table]))

        features = dt.columns[~dt.columns.isin(list(self.meta_vars) + ['label'])].tolist()
        features = list(set(features) - set(drop_list))
        cat_features = [feature for feature in features if feature in self.all_cats]

        num_boost_round = 100

        lgb_train = lgb.Dataset(dt[dt['label'] >= 0][features].values,
                                label=dt[dt['label'] >= 0]['label'].values,
                                feature_name=features,
                                categorical_feature=cat_features,
                                free_raw_data=False)

        bst = lgb.train(params, lgb_train, num_boost_round=num_boost_round)

        def score(x, y):
            return roc_auc_score(y, bst.predict(x))

        cv = lgb.cv(params, lgb_train, num_boost_round=num_boost_round)
        log('Score before feature selection:')
        log('Max CV ROC AUC score: {}'.format(max(cv['auc-mean'])))
        log('Min CV ROC AUC score: {}'.format(min(cv['auc-mean'])))
        log('Average CV ROC AUC score: {}\n'.format(sum(cv['auc-mean']) / len(cv['auc-mean'])))

        pred = bst.predict(dt[features].values)
        sns.distplot(pred)
        plt.show()
        if perm:
            _, score_decreases = get_score_importances(score, dt[features].values, dt['label'].values)
            feature_importances = np.mean(score_decreases, axis=0)
        else:
            feature_importances = bst.feature_importance(importance_type='gain')

        res = pd.DataFrame({'name': features, 'fi': feature_importances})

        self.top_features = res.sort_values(by='fi', ascending=False).head(self.top_amount).name.values.tolist()
        self.top_cats = [feature for feature in self.top_features if feature in self.all_cats]

        lgb_train = lgb.Dataset(dt[dt['label'] >= 0][self.top_features].values,
                                label=dt[dt['label'] >= 0]['label'].values,
                                feature_name=self.top_features,
                                categorical_feature=self.top_cats,
                                free_raw_data=False)

        cv = lgb.cv(params, lgb_train, num_boost_round=num_boost_round)
        log('Score on top {} selected features:'.format(self.top_amount))
        log('Max CV ROC AUC score: {}'.format(max(cv['auc-mean'])))
        log('Min CV ROC AUC score: {}'.format(min(cv['auc-mean'])))
        log('Average CV ROC AUC score: {}\n'.format(sum(cv['auc-mean']) / len(cv['auc-mean'])))

        res = res.sort_values(by='fi', ascending=False).head(self.top_amount)
        sns.barplot(res.fi, res.name)
        plt.show()
        return res
