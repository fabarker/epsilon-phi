import os
from epsilonPhi.core.dataModel.alchemist.SessionManager import *

nan = pd.pandas._libs.tslibs.nattype.NaTType
session = SessionMgr().getSessionFactory()

_DATA_PATH = os.path.join(os.environ.get('HOMEDRIVE'), os.environ.get('HOMEPATH'), 'Documents', 'Data')


folder_name = r'options'
info_workbook_name = 'Options Spec.xlsx'
info_sheetname = 'Underlier Mapping'

df_info = pd.read_excel(os.path.join(_DATA_PATH, folder_name, info_workbook_name), sheet_name=info_sheetname)
df_info = df_info.set_index('Symbol', drop=True)


data_folder = os.path.join(_DATA_PATH, folder_name, 'Data')
_new_data_folder = os.path.join(_DATA_PATH, folder_name, 'Data Repository')

spec = pd.DataFrame()
for symbol, row in df_info.iterrows():

    dta_path = os.path.join(data_folder, symbol.replace('.','') + '.csv')
    df_raw = pd.read_csv(dta_path, index_col=0, header=[1, 2, 3, 4, 5])
    dataypes = ['O1','OY','O2','O3','O6','ON']
    subset_raw = df_raw.iloc[:, [x in dataypes for x in df_raw.columns.get_level_values('DATATYPE')]]

    for col in subset_raw.columns:
        df_col = subset_raw.get(col).to_frame().dropna()
        df_col.columns.names = subset_raw.columns.names
        MNEM_TYPE = df_col.columns.get_level_values('MNEM')[0] + '(' + df_col.columns.get_level_values('DATATYPE')[0] + ')'
        df_col.to_csv(os.path.join(_new_data_folder, MNEM_TYPE.replace('.','') + '.csv'))
        new_info_row = row.to_frame().copy().T
        new_info_row['new_name'] = df_col.columns.get_level_values('Name')[0]
        new_info_row['symbol'] = MNEM_TYPE
        spec = pd.concat((spec, new_info_row), axis=0)