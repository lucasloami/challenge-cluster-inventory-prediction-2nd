from __future__ import division

import pandas as pd
import numpy as np


class TMCMFeatureEng(object):
    def __init__(self, df, categ_cols, numeric_cols):
        self.df = df.copy()
        self.categ_cols = categ_cols
        self.numeric_cols = numeric_cols
        
    def _preprocessing(self):
        self.df[self.numeric_cols] = self.df[self.numeric_cols].apply(lambda x: 
                                                                      (x.astype(float) - min(x))/(max(x)-min(x)), 
                                                                      axis=0)
    
    def _create_matrix(self, indexes, columns):
        return(pd.DataFrame(index=indexes, columns=columns))
    
    def _get_col_name(self, val):
        for k, v in self.categ_mapping.iteritems():
            if val in self.categ_mapping[k]:
                return(k)         
    
    def set_categ(self):
        self.categ_vals = []
        self.categ_mapping = {}
        counter = 0
        for col in self.categ_cols:
            unique_vals = self.df[col].unique().tolist()
            self.categ_vals += unique_vals
            self.categ_mapping[col] = unique_vals
            if len(unique_vals) > counter:
                self.base_items = unique_vals
                self.base_col = col
            
    def fill_co_occurence_matrix(self):
        offset = 0
        for index in self.co_occurrence_matrix.index:
            offset_columns = self.co_occurrence_matrix.columns[offset:]
            offset +=1
            for column in offset_columns:
                i_col = self._get_col_name(index)
                c_col = self._get_col_name(column)
                n = len(self.df[ (self.df[i_col] == index) & (self.df[c_col] == column) ])
                self.co_occurrence_matrix.loc[index, column] = n
#         self.co_occurrence_matrix.fillna(0, inplace=True)
    
    def fill_distance_matrix(self):
        offset = 1
        for index in self.distance_matrix.index:
            offset_columns = self.distance_matrix.columns[offset:]
            offset +=1
            for column in offset_columns:
                mxy = self.co_occurrence_matrix.loc[index, column]
                mx = self.co_occurrence_matrix.loc[index, index]
                my = self.co_occurrence_matrix.loc[column, column]
                dxy = mxy/(mx + my - mxy)
                self.distance_matrix.loc[index, column] = dxy
#         self.distance_matrix.fillna(0, inplace=True)
    
    def _get_best_attr_within_group_variance(self):
        best_wgv = {'wgv': float('Inf')}
        for numeric_col in self.numeric_cols:
            ss = 0
            for base_item in self.base_items:
                sub_df = self.df[self.df[self.base_col] == base_item]
                var = sub_df[numeric_col].var()
                var = 0 if np.isnan(var) else var
                ss += var

            if  ss < best_wgv['wgv']:
                best_wgv['wgv'] = ss
                best_wgv['attr_col'] = numeric_col
                
        return(best_wgv)
    
    def calculate_value_for_categ(self):
        best_wgv = self._get_best_attr_within_group_variance()
        
        # CALCULATE FOR BASE COL
        df2 = self.df.groupby(self.base_col).agg({best_wgv['attr_col']: 'mean'}).reset_index()
        df2 = df2.rename(columns={best_wgv['attr_col']: "numeric_{}".format(self.base_col)})
#         self.teste = df2
        self.df = self.df.merge(df2, how='inner', on=self.base_col)
        
        #CALCULATE FOR REMAINING CATEG COLS
        for col, values in self.categ_mapping.iteritems():
            if col != self.base_col:
                self.df['numeric_{}'.format(col)] = np.nan
                for value in values: # unique values [A, B]
                    v = sum([self.distance_matrix.loc[value, item] * 
                             df2.loc[df2[self.base_col] == item, 'numeric_{}'.format(self.base_col)].iloc[0] 
                             for item in self.base_items])
                    self.df.loc[self.df[col] == value, 'numeric_{}'.format(col)] = v
    
    def transform(self, normalize=False):
        if normalize == True:
            self._preprocessing()
            
        self.set_categ()
        self.co_occurrence_matrix = self._create_matrix(self.categ_vals, self.categ_vals)
        self.distance_matrix = self._create_matrix(self.categ_vals, self.categ_vals)
        self.fill_co_occurence_matrix()
        self.fill_distance_matrix()
        self.calculate_value_for_categ()
        return(self.df)