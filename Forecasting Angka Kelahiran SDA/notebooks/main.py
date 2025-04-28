import pandas as pd

path = 'data/raw/data_ahh_periode.csv'

df = pd.read_csv(path)
df = df[df['1'] == 'Sidoarjo'].reset_index(drop=True)

df = df.rename(columns={
    '0': 'Periode',
    '1': 'Kota',
    '2': 'Pria_1',
    '3': 'Wanita_1',
    '4': 'Pria_2',
    '5': 'Wanita_2',
    '6': 'Pria_3',
    '7': 'Wanita_3'
})

periode_mapping = {
    '2010-2012': [2010, 2011, 2012],
    '2013-2015': [2013, 2014, 2015],
    '2016-2018': [2016, 2017, 2018],
    '2019-2021': [2019, 2020, 2021],
    '2022-2024': [2022, 2023, 2024],
}

rows = []
for idx, row in df.iterrows():
    periode = row['Periode']
    kota = row['Kota']
    tahun_list = periode_mapping.get(periode, [])
    for i in range(3):
        if i < len(tahun_list):
            tahun = tahun_list[i]
            pria = row[f'Pria_{i+1}'].replace(',', '.') if pd.notna(row[f'Pria_{i+1}']) else None
            wanita = row[f'Wanita_{i+1}'].replace(',', '.') if pd.notna(row[f'Wanita_{i+1}']) else None
            if pria != '-' and pria is not None:
                rows.append([tahun, kota, 'Laki-laki', float(pria)])
            if wanita != '-' and wanita is not None:
                rows.append([tahun, kota, 'Perempuan', float(wanita)])

df_sda = pd.DataFrame(rows, columns=['Tahun', 'Kota', 'Jenis_Kelamin', 'Jumlah'])
df_sda = df_sda.sort_values(by=['Tahun']).reset_index(drop=True)
df_sda.to_csv('data_ahh_sidoarjo_formatted.csv', index=False)
print(df_sda)
print(df_sda.shape)