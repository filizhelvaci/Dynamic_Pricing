import pandas as pd
import numpy as np
import statsmodels.stats.api as sms
from scipy.stats import shapiro
from scipy import stats
from scipy import stats

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)
#pd.set_option('display.float_format', lambda x: '%.f' % x)

df_ = pd.read_csv("dataset/pricing.csv",sep=";")
df= df_.copy()
df.head()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def crm_data_prep(dataframe):
    dataframe.dropna(axis=0, inplace=True)
    replace_with_thresholds(dataframe, "price")

    return dataframe

df["category_id"].value_counts()
df.groupby("category_id").mean()
df.groupby("category_id")["price"].describe()
df.groupby("category_id").aggregate([np.mean,np.median])

df_p=crm_data_prep(df)

df_489=df_p.loc[df["category_id"]==489756,["price"]]
df_201=df_p.loc[df["category_id"]==201436,["price"]]
df_874=df_p.loc[df["category_id"]==874521,["price"]]
df_361=df_p.loc[df["category_id"]==361254,["price"]]
df_326=df_p.loc[df["category_id"]==326584,["price"]]
df_675=df_p.loc[df["category_id"]==675201,["price"]]


# Item'in fiyatı kategorilere göre farklılık göstermekte midir? İstatistiki olarak ifade edilmesi

#Hipotezimiz:
# H0: M1=M2   -Katogoriler arasında anlamlı bir farklılık yotur
# H1: M1!=M2    -Katogoriler arasında anlamlı bir farklılık vardır

list=[df_201,df_326,df_361,df_489,df_675,df_874]
for i in list:
    test_istatistigi, pvalue = shapiro(i["price"])
    print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

    # H0: Normal dağılım varsayımı sağlanmaktadır.
    # H1: ...sağlanmamaktadır
# Normallik varsayımı hiç bir kategori için sağlanmadı.Bu sebeple AB Testlerinde
# Bağımsız iki örneklem T Testini kullanırken nonparametrik mannwhitneyu testi kullanacağız

for i in range(0,6):
    a=5
    print("dış döngü")
    for j in range(i,a):
        #print(i,j+1,a)
        a-=1
        test_istatistigi, pvalue = stats.mannwhitneyu(list[i],list[j+1])
        print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
        print(f"döngü={i}{j+1} da bulunuyorsun")


# Fiyatlar her kategoride,o kategorinin ortalama değeri olmalıdır.
# Çünkü yapılan AB testlerinde kategoriler arasında fark vardır ağırlıklı çıkan sonuç oldu.

df_p.groupby("category_id")["price"].describe()
df_p.groupby("category_id").aggregate([np.mean,np.median])

# Fiyat konusunda "hareket edebilir olmak" istenmektedir. Fiyat stratejisi için karar destek sistemi oluşturulması
# Güven Aralığı oluşturuldu

sms.DescrStatsW(df_489["price"]).tconfint_mean()
sms.DescrStatsW(df_201["price"]).tconfint_mean()
sms.DescrStatsW(df_874["price"]).tconfint_mean()
sms.DescrStatsW(df_361["price"]).tconfint_mean()
sms.DescrStatsW(df_326["price"]).tconfint_mean()
sms.DescrStatsW(df_675["price"]).tconfint_mean()

