from collections import defaultdict
import matplotlib.pyplot as plt

import pandas as pd
from glob import glob
import seaborn as sns

folder_dropout_original = '/home/fcf/Downloads/resultados_dstitan/results do script com todos os treinamentos/varias simulacoes/'
folder_modifications = '/home/fcf/Downloads/resultados_dstitan/results do script com todos os treinamentos/varias simulacoes/dropout_decayed/cos_decayed/'

files_sgd = sorted(glob(folder_dropout_original + '*sgd_dropoutmethod=original*'))
files_modifications = sorted(glob(folder_modifications + '*.csv'))

sgd_tr = defaultdict(list)
modifications_tr = defaultdict(list)
sgd_val = defaultdict(list)
modifications_val = defaultdict(list)

for i, (file_sgd, file_modifications) in enumerate(zip(files_sgd, files_modifications)):
    sgd_csv = pd.read_csv(file_sgd)
    modifications_csv = pd.read_csv(file_modifications)

    sgd_tr[i] = sgd_csv['acc']
    modifications_tr[i] = modifications_csv['acc']
    sgd_val[i] = sgd_csv['val_acc']
    modifications_val[i] = modifications_csv['val_acc']

sgd_df_tr = pd.DataFrame.from_dict(sgd_tr)
modifications_df_tr = pd.DataFrame.from_dict(modifications_tr)
sgd_df_val = pd.DataFrame.from_dict(sgd_val)
modifications_df_val = pd.DataFrame.from_dict(modifications_val)

fix, ax = plt.subplots()

ax.plot(sgd_df_tr.mean(axis=1), 'b', label='original_tr')
ax.plot(modifications_df_tr.mean(axis=1), 'r', label='mod_tr')
ax.plot(sgd_df_val.mean(axis=1), 'b--', label='original_val')
ax.plot(modifications_df_val.mean(axis=1), 'r--', label='mod_val')

ax.legend()
print('finished')



