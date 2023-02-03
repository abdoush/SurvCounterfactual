import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

class CMAPSS:
    def __init__(self, path, max_cycle=None, max_rul=130, corr_threshold=1.0):
        self._filepath = path
        self.max_cycle = max_cycle
        self.max_rul = max_rul
        
        self.feature_names = self.get_feature_names()
        self.stationary_features = []
        self.redundant_features = []
        self.df = self._load_dataframe()
        self.scaler = None


    def _load_dataframe(self):
        
        feature_names = self.get_feature_names()
        names = ['unit_nr', 'time', 'operation_setting_1', 'operation_setting_2',
             'condition_3']
        names += feature_names
        
        df = pd.read_csv(self._filepath, sep=' ', header=None, index_col=False)
        df = df.dropna(axis=1) # removes two nan colums at the end
        df.columns = names
        
        self.find_stationary_features(df)
        df = self.drop_stationary_features(df)
        self.find_redundant_features(df)
        df = self.drop_redundant_features(df)
        
        self.update_features()
        df = self.limit_and_label_events(df)
        df = self.calculate_time_to_event(df)

        return df
    
    def get_feature_names(self, name="short"):
        sensor_names = pd.DataFrame(columns=['sensor_id', 'sensor_name', 'sensor_description'],
        data=[
            ['sensor_01', 'T2', 'Total temperature at fan inlet'],
            ['sensor_02', 'T24', 'Total temperature at LPC outlet'],
            ['sensor_03', 'T30', 'Total temperature at HPC outlet'],
            ['sensor_04', 'T50', 'Total temperature at LPT outlet'],
            ['sensor_05', 'P2', 'Pressure at fan inlet'],
            ['sensor_06', 'P15', 'Total pressure in bypass-duct'],
            ['sensor_07', 'P30', 'Total pressure at HPC outlet'],
            ['sensor_08', 'Nf', 'Physical fan speed'],
            ['sensor_09', 'Nc', 'Physical core speed'],
            ['sensor_10', 'epr', 'Engine pressure ratio (P50/P2)'],
            ['sensor_11', 'Ps30', 'Static pressure at HPC outlet'],
            ['sensor_12', 'phi', 'Ratio of fuel flow to Ps30'],
            ['sensor_13', 'NRf', 'Corrected fan speed'],
            ['sensor_14', 'NRc', 'Corrected core speed'],
            ['sensor_15', 'BPR', 'Bypass Ratio'],
            ['sensor_16', 'farB', 'Burner fuel-air ratio'],
            ['sensor_17', 'htBleed', 'Bleed Enthalpy'],
            ['sensor_18', 'Nf_dmd', 'Demanded fan speed'],
            ['sensor_19', 'PCNfR_dmd', 'Demanded corrected fan speed'],
            ['sensor_20', 'W31', 'HPT coolant bleed'],
            ['sensor_21', 'W32', 'LPT coolant bleed']
        ])
        
        if name == "full":
            feature_names = sensor_names.sensor_description.to_list()
        elif name == "short":
            feature_names = sensor_names.sensor_name.to_list()
        elif name == "id":
            feature_names = sensor_names.sensor_id.to_list()
        else:
            raise Warning("Name of columns not set correctly")
            feature_names = ["S" + str(i+1) for i in range(21)]
            
        return feature_names
    
    def limit_and_label_events(self, df):
        dfc = df.copy(deep=True)
        
        if self.max_cycle is not None:
            max_cycle = self.max_cycle
        else:
            max_cycle = dfc["time"].max() + 1
        
        
        for unit in dfc["unit_nr"].unique():
            max_cycle_unit = dfc.loc[dfc["unit_nr"] == unit, "time"].max()
            dfc.loc[dfc["unit_nr"] == unit, "E"] = max_cycle_unit <= max_cycle
        
        dfc = dfc.loc[dfc["time"] <= max_cycle]
        
        return dfc
    
    def find_stationary_features(self, df):
        features = self.get_feature_names()
        
        for col in features:
            unique_values = len(df[col].value_counts())

            if unique_values <=1:
                self.stationary_features.append(col)
    
    def find_redundant_features(self, df):       
        features = self.get_feature_names()
        df = df.loc[:, df.columns.isin(features)]
        
        corr_matrix = df.corr()
        n = len(df.columns)
        for i in range(0, n):
            for j in range(i+1, n):
                first_col = df.columns[i]
                second_col = df.columns[j]
                corr = corr_matrix.loc[first_col, second_col]

                if abs(corr) > 0.93 and second_col in features:
                    self.redundant_features.append(second_col)

    
    def drop_stationary_features(self, df):
        df = df.drop(self.stationary_features, axis=1)
        return df
    
    
    def drop_redundant_features(self, df):
        df = df.drop(self.redundant_features, axis=1)
        return df
    
    def update_features(self):
        for col in self.stationary_features:
            if col in self.feature_names:
                self.feature_names.remove(col)
                
        for col in self.redundant_features:
            if col in self.feature_names:
                self.feature_names.remove(col)
        
    
    def calculate_time_to_event(self, df):
        dfc = df.copy(deep=True)
        for unit in dfc["unit_nr"].unique():
            max_cycle = dfc.loc[dfc["unit_nr"] == unit, "time"].max()
            dfc.loc[df["unit_nr"] == unit, "T"] = max_cycle - dfc.loc[dfc["unit_nr"] == unit, "time"]
        
        if self.max_rul is not None:
            dfc["T"] = dfc["T"].clip(upper=self.max_rul)
        
        return dfc
    
    
    def _get_idx(self, df):
        return df["unit_nr"].values
    
    def _get_X(self, df):
        return df[self.feature_names].values
    
    def _get_T(self, df):
        return df["T"].values
    
    def _get_E(self, df):
        return df["E"].values
    
    
    def fit_scaler(self, X):
        self.scaler = MinMaxScaler()
        self.scaler.fit(X)
    
    def get_train_test_val(self, test_size, val_size, fit_scaler=True, scale=True, random_state=42):
        units = self.df["unit_nr"].unique()
        
        train_size = 1 - test_size - val_size
        test_size_sub = test_size / (test_size + val_size)
        units_train, units_test = train_test_split(units, train_size=train_size, shuffle=True, random_state=random_state)
        units_test, units_val = train_test_split(units_test, test_size=test_size_sub, shuffle=True, random_state=random_state)
        
        loc_train = self.df["unit_nr"].isin(units_train)
        loc_test = self.df["unit_nr"].isin(units_test)
        loc_val = self.df["unit_nr"].isin(units_val)
        
        idx_train = self._get_idx(self.df.loc[loc_train])
        X_train = self._get_X(self.df.loc[loc_train])
        E_train = self._get_E(self.df.loc[loc_train])
        T_train = self._get_T(self.df.loc[loc_train])
        
        idx_test = self._get_idx(self.df.loc[loc_test])
        X_test = self._get_X(self.df.loc[loc_test])
        E_test = self._get_E(self.df.loc[loc_test])
        T_test = self._get_T(self.df.loc[loc_test])
        
        idx_val = self._get_idx(self.df.loc[loc_val])
        X_val = self._get_X(self.df.loc[loc_val])
        E_val = self._get_E(self.df.loc[loc_val])
        T_val = self._get_T(self.df.loc[loc_val])
        
        if fit_scaler:
            self.fit_scaler(X_train)
        
        if scale:
            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_val = self.scaler.transform(X_val)
            
        return ((idx_train, X_train, T_train, E_train),
                (idx_test, X_test, T_test, E_test),
                (idx_val, X_val, T_val, E_val)
               )
    
    
    def get_failure_mode(self, df):
        """ WORKS ONLY WITH FD003 !!!!"""        
        # scale data for clustering algorithm
        X = self._get_X(df)
        scaler = MinMaxScaler()
        scaler.fit(X)
        
        # get data about last cycle of each unit
        df_end = df.drop_duplicates(subset=["unit_nr"], keep="last")
        idx_end = df_end["unit_nr"].values
        X_end = scaler.fit_transform(df_end)
        
        
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(X_end)
        
        failure_mode = kmeans.predict(X_end)
        idx_failure_mode = {k: v for k, v in zip(idx_end, failure_mode)}
        
        return idx_failure_mode