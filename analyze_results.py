from collections import defaultdict
import matplotlib.pyplot as plt

import pandas as pd
from glob import glob
import seaborn as sns

folder = '/home/fcf/Downloads/resultados_dstitan/results do script com todos os treinamentos/varias simulacoes/'

files_sgd = sorted(glob(folder + '*sgd*'))
files_dropgrads = sorted(glob(folder + '*dropgrads*'))

sgd = defaultdict(list)
dropgrads = defaultdict(list)

for i, (file_sgd, file_dropgrads) in enumerate(zip(files_sgd, files_dropgrads)):
    sgd_csv = pd.read_csv(file_sgd)
    dropgrads_csv = pd.read_csv(file_dropgrads)

    sgd[i] = sgd_csv['val_acc']
    dropgrads[i] = dropgrads_csv['val_acc']

sgd_df = pd.DataFrame.from_dict(sgd)
dropgrads_df = pd.DataFrame.from_dict(dropgrads)

