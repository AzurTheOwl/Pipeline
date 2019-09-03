from .utils import timeit
from .general_methods import GeneralMethods
from .feature_selection import FeatureSelection
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve, auc)

pd.options.display.max_columns = None


class ModelCreator(GeneralMethods):
    """
    Class with all modeling routine. Also contains adversarial validation.
    """
    def __init__(self,
                 table_name: str = '',
                 report_date: str = '',
                 data_version: int = 1,
                 features_file: str = 'top_30_features.npy',
                 ext_features_file: str = 'ext_features.npy',
                 feature_selection: FeatureSelection = None
                 ):
        """
        Constructor for ModelCreator class.

        Args:
            table_name (str): name of table with ids and labels
            report_date (str): report date for default data downloading
            data_version (int): version for directory generation and data storage
            features_file (str): features filename
            ext_features_file (str): external features filename
            feature_selection (FeatureSelection): object of FeatureSelection class to get constructor parameters
        """
        if feature_selection is not None:
            self.data_version = feature_selection.data_version
            self.directory = 'data_v{}'.format(self.data_version)
            self.report_date = feature_selection.report_date
            self.features_file = feature_selection.top_features_file
            self.ext_features_file = feature_selection.ext_features_file
            self.table_name = feature_selection.table_name
        else:
            self.data_version = data_version
            self.directory = 'data_v{}'.format(self.data_version)
            self.report_date = report_date
            self.features_file = r'{}/{}'.format(self.directory, features_file)
            self.ext_features_file = r'{}/{}'.format(self.directory, ext_features_file)
            self.table_name = table_name

        super().__init__(self.data_version)

        self.features = []
        try:
            self.features = list(np.load(self.features_file))
        except FileNotFoundError:
            print('No features loaded.')
            pass

        self.ext_features = {}
        try:
            self.ext_features = np.load(self.ext_features_file).item()  # type: dict
        except FileNotFoundError:
            print('No external features loaded.')
            pass

        self.all_cats_dict = {}
        self.lgb_model = None

        self.replacers = {'table_name': self.table_name,
                          'report_date': self.report_date,
                          'directory': self.directory}

    def process_dt_list(self, dt_list: list, code_dict: dict = None, code_vars: list or set = None):
        """
        Apply fix and encoding to specified list of data frames.

        Args:
            dt_list (list): list of data frames
            code_dict (dict): .
            code_vars (list or set): .

        Returns:

        """

        save_self = False

        if code_dict is None:
            code_dict = self.all_cats_dict
            save_self = True

        if code_vars is None:
            code_vars = self.all_cats

        for dt in dt_list:
            dt = super().wcat_features_fix(dt)
            if code_dict:
                self.cats_transform(dt, code_dict)
            else:
                _, code_dict = self.cats_fit_transform(dt, code_dict, code_vars)

        if save_self:
            self.all_cats_dict = code_dict
            return
        else:
            return code_dict

    def add_ext_features(self, dt: pd.DataFrame, sub_query: str = None, aff: str = None) -> pd.DataFrame:
        """
        Adds external features to specified data frame.
        Args:
            dt (DataFrame): data frame to fill with external features
            sub_query (str): sub-query for labels and subs_id
            aff (str): file affix to form filename

        Returns:
            dt (DataFrame): data frame with added external features
        """
        for tablename in self.ext_features.keys():
            df = self.download_data(tablename, sub_query, aff)
            dt = dt.merge(df, on='subs_id', how='left')
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
        if not sub_query:
            sub_query = """
                SELECT subs_id, label
                from {table_name}
                where report_date = DATE'{report_date}'   
                """.format(**self.replacers)

        if (not tablename) or (tablename.lower() == 'NO_DATA'):
            tablename = 'NO_DATA'

            # if aff == 'to_score':
            #     placeholder = ''
            # else:
            placeholder = "b.label, "

            query = "SELECT a.SUBS_ID, {}".format(placeholder) + ', '.join(self.vars) + \
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

        if not aff:
            aff = self.table_name.split('.')[1]

        file_path = r'{}/{}_{}_{}.feather'.format(
            self.replacers['directory'],
            aff,
            self.replacers['report_date'],
            tablename.split('.')[1])

        # print(query)

        dt = super().download_data(file_path, query)

        dt = dt[[col for col in dt.columns if (col in self.features) | (col in ['subs_id', 'label'])]]
        return dt

    @staticmethod
    def plot_roc_auc(label: list, predict: list, g_label: list, figsize: (int, int) = (10, 8)):
        """
        Plots ROC AUC curve for given labels and predictions.

        Args:
            label: list of true labels
            predict: list of predictions
            g_label: list of plot labels
            figsize: desired plot size
        """
        if (str(type(label)) != "<class 'list'>") | \
                (str(type(predict)) != "<class 'list'>") | \
                (str(type(g_label)) != "<class 'list'>"):
            return 'Error: label, predict or g_label is not a list'
        elif (len(label) != len(predict)) & (len(label) != len(g_label)):
            return 'Error: lengths of label, predict and g_label mismatch'
        else:
            plt.figure(figsize=figsize)

            for i in range(len(label)):
                fpr, tpr, _ = roc_curve(label[i], predict[i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label='AUC={}, {}'.format(np.round(roc_auc, 4), g_label[i]))

            plt.plot([0, 1], [0, 1], color='black', linestyle='--')
            plt.legend()
            plt.show()

    @timeit
    def adversarial_validation(self, dt_ones: pd.DataFrame,
                               dt_zeros: pd.DataFrame,
                               zeros_part: float = 0.1,
                               params: dict = None,
                               cat_features: list = None,
                               drop_list: list = None):
        """
        Performs adversarial validation procedure.

        For two given data frames calculate their "similarity": creates model which divide them and then calculate AUC.

        Args:
            dt_ones (DataFrame): data frame to label as ones
            dt_zeros (DataFrame): data frame to label as zeros
            zeros_part (float): if zeros data frame is too large - set part to keep
            params (dict): LGBClassifier parameters dictionary
            cat_features (list): categorical features to use
            drop_list (list): list of features which should be dropped
        """
        if drop_list is None:
            drop_list = []

        dt_zeros = dt_zeros.copy()
        dt_ones = dt_ones.copy()

        drop_indices = np.random.choice(dt_zeros.index, int(dt_zeros.shape[0] * (1 - zeros_part)), replace=False)
        dt_zeros = dt_zeros.drop(drop_indices)

        dt_ones['label'] = 1
        dt_zeros['label'] = 0

        dt_zeros = pd.concat([dt_ones, dt_zeros.loc[~dt_zeros['subs_id'].isin(list(dt_ones['subs_id']))]], axis=0)

        dt_train, dt_val_0 = self.split(dt_zeros)
        dt_list = [dt_train, dt_val_0]

        model = self.model_create(dt_train, dt_list, params, cat_features, drop_list, save_self=False)
        feature_importances = model.feature_importance(importance_type='gain')
        res = pd.DataFrame({'name': model.feature_name(), 'fi': feature_importances})
        res = res.sort_values(by='fi', ascending=False)
        sns.barplot(res.fi, res.name)
        plt.show()

    @timeit
    def model_create(self, dt_train: pd.DataFrame,
                     dt_list: list,
                     params: dict = None,
                     cat_features: list = None,
                     drop_list: list = None,
                     save_self: bool = True) -> lgb.Booster:
        """
        Creates LGBClassifier model.

        Args:
            dt_train (DataFrame): train data frame
            dt_list (list): list of validation data frames
            params (dict): LGBMClassifier parameters dictionary
            cat_features (list): list of categorical features to use
            drop_list (list): list of features to drop explicitly
            save_self (bool): assign created model as object`s model

        Returns:
            model (Booster): LGBMClassifier model
        """
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

        if cat_features is None:
            cat_features = list(item for item in self.all_cats_dict.keys() if item in self.features)

        if drop_list is None:
            drop_list = []

        features = dt_train.columns[~dt_train.columns.isin(list(self.meta_vars) + ['label', 'preds'])].tolist()

        features = list(set(features) - set(drop_list))
        cat_features = list(set(cat_features) - set(drop_list))

        # print(features, cat_features)

        lgb_train = lgb.Dataset(dt_train[features].values,
                                label=dt_train['label'].values,
                                feature_name=features,
                                free_raw_data=False,
                                categorical_feature=cat_features)

        valid_sets = [lgb.Dataset(dt_val[features].values,
                                  label=dt_val['label'].values,
                                  feature_name=features,
                                  free_raw_data=False,
                                  categorical_feature=cat_features) for dt_val in dt_list]

        model = lgb.train(params,
                          lgb_train,
                          num_boost_round=100,
                          valid_sets=valid_sets)
        for dt in dt_list:
            dt["preds"] = model.predict(dt[model.feature_name()])

        self.plot_roc_auc(label=[dt['label'] for dt in dt_list],
                          predict=[dt['preds'] for dt in dt_list],
                          # g_label=[[name for name in globals() if globals()[name] is item][0] for item in dt_list])
                          g_label=["valid_{}".format(i) for i, _ in enumerate(dt_list)])

        if save_self:
            self.lgb_model = model
        return model

    @timeit
    def model_evaluate(self, dt: pd.DataFrame, prob: float = 0.5, model: lgb.Booster = None):
        """
        Evaluate model on given data frame.

        Produce probability plots, AUC, average PR, F1, Precision, Recall and confusion matrix.

        Args:
            dt: data frame with labels and scores to evaluate
            prob: threshold to count probabilities as ones
            model: model to evaluate
        """
        if not model:
            model = self.lgb_model

        dt_eval = dt
        dt_eval["preds"] = model.predict(dt_eval[model.feature_name()])
        dt_eval["preds"].head()

        sns.distplot(dt_eval["preds"], axlabel='Full distribution')
        plt.show()
        sns.distplot(dt_eval.loc[dt_eval['label'] == 1, "preds"], axlabel='Ones distribution')
        plt.show()
        sns.distplot(dt_eval.loc[dt_eval['label'] == 0, "preds"], axlabel='Zeros distribution')
        plt.show()
        sns.distplot(dt_eval.loc[dt_eval['label'] == 1, "preds"], axlabel='Ones distribution', kde=False)
        sns.distplot(dt_eval.loc[dt_eval['label'] == 0, "preds"], axlabel='Zeros distribution', kde=False)
        plt.show()

        preds = [0 if x < prob else 1 for x in dt_eval["preds"]]
        cm = confusion_matrix(dt_eval['label'].values, preds)
        df_cm = pd.DataFrame(cm)
        sns.heatmap(df_cm, annot=True)
        plt.show()

        a_score = accuracy_score(dt_eval['label'].values, preds, normalize=True)
        print("Accuracy score: {}\n".format(a_score))

        class_report = classification_report(dt_eval['label'].values, preds, target_names=["Zeros", "Ones"])
        print(class_report)

        total = sum(dt_eval['label'].values)
        predicted = sum(preds)
        print("Total positive labels: {}. Positive labels predicted: {}\n".format(total, predicted))

        average_precision = average_precision_score(dt_eval['label'], dt_eval['preds'])
        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))

        precision, recall, _ = precision_recall_curve(dt_eval['label'], dt_eval['preds'], pos_label=1)

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
        plt.show()

    def model_save(self, filename: str = 'model'):
        """
        Saves object`s model to file.

        Args:
            filename: output filename
        """
        self.lgb_model.save_model(r'{}/{}.lgb'.format(self.replacers['directory'], filename))
        print('Model successfully saved as {}.lgb'.format(filename))

    def cat_dict_save(self, filename: str = 'cat_dict'):
        """
        Saves object`s codes dictionary to file.

        Args:
            filename: output filename
        """
        np.save(r'{}/{}.npy'.format(self.replacers['directory'], filename), self.all_cats_dict)
        print('Categories dictionary successfully saved as {}.npy'.format(filename))
