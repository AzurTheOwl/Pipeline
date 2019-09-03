import os
from .utils import timeit
from .general_methods import GeneralMethods
import lightgbm as lgb
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class ScoringWithLGBM(GeneralMethods):
    """
    Class to download, predict and then upload results to Teradata.
    """

    def __init__(self,
                 data_version: int = 1,
                 ext_date: str = '',
                 dmsc_date: str = '',
                 load_table: str = '',
                 model_file: str = 'model.lgb',
                 ext_features_file: str = 'ext_features.npy',
                 cat_dict_file: str = 'cat_dict.npy',
                 teradata_version: str = 'NO_DATA',
                 teradata_login: str = 'NO_DATA',
                 teradata_pwd: str = 'NO_DATA',
                 teradata_ip: str = 'NO_DATA'
                 ):
        """
        ScoringWithLGBM class constructor.

        Args:
            data_version (int): data version to specify data folder name
            model_file (str): model filename
            ext_features_file (str): external features filename
            cat_dict_file  (str): categorical features filename
            ext_date (str): dmx and other date to data download
            dmsc_date (str): dmsc and subs date to data download
            load_table (str): table name to upload data
            teradata_version (str): Teradata version
            teradata_login (str): Teradata user login
            teradata_pwd (str): Teradata user password
            teradata_ip (str): Teradata IPv4 address
        """

        self.data_version = data_version
        self.directory = 'data_v' + str(self.data_version)
        self.ext_date = ext_date
        self.dmsc_date = dmsc_date
        super().__init__(self.data_version)

        self.cat_dict = np.load(r'{}/{}'.format(self.directory, cat_dict_file)).item()  # type: dict
        self.model = lgb.Booster(model_file=r'{}/{}'.format(self.directory, model_file))
        self.features = self.model.feature_name()
        self.ext_features = np.load(r'{}/{}'.format(self.directory, ext_features_file)).item()  # type: dict

        self.load_table = load_table
        self.table_e1 = self.load_table + '_e1'
        self.table_e2 = self.load_table + '_e2'
        self.load_tables = [
            self.load_table,
            self.table_e1,
            self.table_e2
        ]

        self.replacers = {
            'score_date': self.dmsc_date,
            'report_date': self.ext_date,
            'teradata_version': teradata_version,
            'teradata_ip': teradata_ip,
            'teradata_login': teradata_login,
            'teradata_pwd': teradata_pwd,
            'load_table': self.load_table,
            'table_e1': self.table_e1,
            'table_e2': self.table_e2,
            'directory': self.directory
        }
        self.result_file_name = 'result_{score_date}.csv'.format(**self.replacers)
        self.result_path = r'{}/{}/{}'.format(os.getcwd(), self.replacers['directory'], self.result_file_name)
        self.replacers['result_path'] = self.result_path

    def reinit(self,
               model_file: str = 'model.lgb',
               ext_features_file: str = 'ext_features.npy',
               cat_dict_file: str = 'cat_dict.npy',
               ):
        self.__init__(data_version=self.data_version,
                      ext_date=self.ext_date,
                      dmsc_date=self.dmsc_date,
                      load_table=self.load_table,
                      model_file=model_file,
                      ext_features_file=ext_features_file,
                      cat_dict_file=cat_dict_file,
                      teradata_version=self.replacers['teradata_version'],
                      teradata_login=self.replacers['teradata_login'],
                      teradata_pwd=self.replacers['teradata_pwd'],
                      teradata_ip=self.replacers['teradata_ip'])
        return

    # @timeit
    # def cats_transform(self, dt: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Applies label encoding to df according to code_dict.
    #
    #     Args:
    #         dt (DataFrame): data frame to category features transform
    #
    #     Returns:
    #         dt (DataFrame): transformed data frame
    #     """
    #     code_dict = self.cat_dict
    #
    #     dt = super().cats_transform(dt, code_dict)
    #
    #     return dt

    def drop_if_exists(self, table_name: str):
        """
        Checks table for existence and drop if exists.
        Args:
            table_name: table name
        """
        try:
            drop_sql = 'drop table {}'.format(table_name)
            self.cursor.execute(drop_sql)
        except ConnectionError:
            print('Load table does not exist and will be created')
            pass

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
                NO_DATA
                """.format(**self.replacers)

        if (not tablename) or (tablename.lower() == 'NO_DATA'):
            tablename = 'NO_DATA'

            query = "SELECT a.SUBS_ID, " + \
                    ', '.join([var for var in self.vars if var in self.features]) + \
                    """
                FROM {} a
                join ({}) b
                on a.subs_id = b.subs_id
                WHERE a.REPORT_DATE = DATE'{}'
                """.format(tablename, sub_query, self.dmsc_date)
        else:

            query = "SELECT a.SUBS_ID, " + \
                    ', '.join([var for var in self.ext_features[tablename]]) + \
                    """
                FROM {} a
                join ({}) b
                on a.subs_id = b.subs_id
                WHERE a.REPORT_DATE = DATE'{}'
                """.format(tablename, sub_query, self.ext_date)

        if not aff:
            aff = 'to_score'

        file_path = r'{}/{}_{}_{}.feather'.format(
            self.replacers['directory'],
            aff,
            self.replacers['report_date'],
            tablename.split('.')[1])

        dt = super().download_data(file_path, query)

        dt = dt[[col for col in dt.columns if (col in self.features) |
                 (col in ['subs_id', 'label'])]]
        return dt

    @timeit
    def process_data(self, dt: pd.DataFrame) -> pd.DataFrame:
        """
        Applies fix and encoding to data frame.

        Args:
            dt (DataFrame): data frame to apply fixes and transform variables

        Returns:
            dt (DataFrame): processed data frame
        """
        dt = self.wcat_features_fix(dt)
        dt = self.cats_transform(dt, self.cat_dict)
        return dt

    @timeit
    def score_data(self, dt: pd.DataFrame) -> pd.DataFrame:
        """
        Scores data frame with object`s model.

        Args:
            dt: data frame to score

        Returns:
            dt (DataFrame): data frame with predictions column
        """
        # dt = self.download_data(query=self.dmsc_sql_query)
        dt['preds'] = self.model.predict(dt[self.features])
        dt = dt.round({'preds': 10})

        sns.distplot(dt['preds'], label='Predictions')
        plt.show()

        return dt

    @timeit
    def upload_data(self, dt: pd.DataFrame, bucketise: bool = False, num_bins: int = 20):
        """
        Uploads score to Teradata.

        Args:
            dt (DataFrame): data frame with scores
            bucketise (bool): use buckets flag
            num_bins (int): in case of bucketise=True number of buckets
        """
        self.establish_connection()

        if bucketise:
            result = dt.sort_values(by='preds', ascending=False)[:int(4e6)].copy()
            bins = np.percentile(result['preds'], np.linspace(1, 100, num=num_bins))
            result['bucket_value'] = num_bins - np.digitize(result['preds'], bins, right=True)
            result['probability'] = 1.01 - result['bucket_value'] / (num_bins * 2)
        else:
            result = dt.sort_values(by='preds', ascending=False)[:].copy()
            result['bucket_value'] = 1
            result['probability'] = result['preds']

        result['subs_id'] = result['subs_id'].astype(str)
        result[['subs_id', 'probability', 'bucket_value']].to_csv(self.result_path,
                                                                  index=None,
                                                                  sep=';',
                                                                  header=None)

        for table_name in self.load_tables:
            self.drop_if_exists(table_name)

        create_table_sql = """
                        CREATE MULTISET TABLE {load_table},
                        NO FALLBACK,
                        NO BEFORE JOURNAL,
                        NO AFTER JOURNAL,
                        CHECKSUM = DEFAULT,
                        DEFAULT MERGEBLOCKRATIO
                        (
                            SUBS_ID DECIMAL(12,0),
                            PROBABILITY DECIMAL(18,6),
                            BUCKET_VALUE DECIMAL(18,6)
                        )
                        PRIMARY INDEX ( SUBS_ID )
                        """.format(**self.replacers)

        self.cursor.execute(create_table_sql)

        path_to_teradata = 'C://Program Files (x86)//Teradata//Client//{teradata_version}//bin'.format(**self.replacers)

        # Create txt loading file
        load_file_name = 'load_file.txt'
        fast_load_file = "{}/{}/{}".format(os.getcwd(), self.replacers['directory'], load_file_name)
        # self.replacers['fl_file'] = fast_load_file

        load_file = open(fast_load_file, "w+")
        load_text = """
            SET SESSION CHARSET 'LATIN1'; 
            SET SESSION CHARSET 'UTF8';
            .logon {teradata_ip}/{teradata_login}, {teradata_pwd};            
            .SET RECORD VARTEXT ";";
            DEFINE
            SUBS_ID (VARCHAR(50)),
            PROBABILITY (VARCHAR(20)),
            BUCKET_VALUE (VARCHAR(20)),

            FILE {result_path};
            BEGIN LOADING {load_table} ERRORFILES {table_e1}, {table_e2}
            CHECKPOINT 300000;

            INSERT INTO {load_table} 
            VALUES
            (
                :SUBS_ID,
                :PROBABILITY,
                :BUCKET_VALUE
            );
            
            END LOADING;
            .LOGOFF;
            .QUIT;
            """.format(**self.replacers)

        load_file.write(load_text)
        load_file.close()

        os.system('cd {}'.format(path_to_teradata))

        os.system('fastload.exe < {}'.format(fast_load_file))

        self.connect.close()
