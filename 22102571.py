import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from sklearn import cluster
import errors as err
import cluster_tools as clust


#Read and clean the datafile
def readFile(fn):
    path = "filepath\\" + fn
    df = pd.read_csv(path)
    df = df.drop(df.columns[:2], axis=1)
    df=df.drop(columns=['Country Code'])
    # Remove the string of year from column names
    df.columns = df.columns.str.replace(' \[YR\d{4}\]', '', regex=True)
    countries=['United States','Italy']
    #Transpose the dataframe
    df = df[df['Country Name'].isin(countries)].T 
    #Rename columns
    df = df.rename({'Country Name': 'year'})
    df = df.reset_index().rename(columns={'index': 'year'})
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df = df.replace('..', np.nan)
    df = df.replace(np.nan,0)
    df["year"] = df["year"].astype(int)
    df["Italy"]=df["Italy"].astype(float)
    df["United States"]=df["United States"].astype(float)
    return df
def curve_fun(t, scale, growth):
 c = scale * np.exp(growth * (t-1990))
 return c
def gdp(df_gdp):
  plt.plot(df_gdp["year"], df_gdp["Italy"], c="green", label="Italy")
  plt.plot(df_gdp["year"], df_gdp["United States"], c="pink", label="United States")
  plt.xlim(1990,2019)
  plt.xlabel("Year")
  plt.ylabel(" GDP per capita growth (annual %)")
  plt.legend()
  plt.title("GDP per capita for Italy and United States")
  plt.show()
df_co2=readFile("Co2_Emissions.csv")
df_gdp=readFile("GDP_per_Capita.csv")
df_renew= readFile("Renewable_Energy.csv")
param, cov = opt.curve_fit(curve_fun,df_co2["year"],df_co2["Italy"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
error_range = err.error_prop(df_co2["year"], curve_fun, param, sigma)
mean = error_range.iloc[0]
up = mean + error_range.iloc[1]
low = mean - error_range.iloc[1]
df_co2["fit_value"] = curve_fun(df_co2["year"], * param)
# 1: Plotting the co2 emission values for Italy
plt.figure()
plt.title("CO2 emissions (metric tons per capita) - Italy")
plt.plot(df_co2["year"],df_co2["Italy"], c="green", label="data")
plt.plot(df_co2["year"],df_co2["fit_value"],c="pink",label="fit")
plt.fill_between(df_co2["year"],low,up,alpha=0.4)
plt.legend()
plt.xlim(1990,2019)
plt.xlabel("Year")
plt.ylabel("CO2")
plt.show()
# 2: Plotting the predicted values for Italy co2 emission
plt.figure()
plt.title("CO2 emission prediction of Italy")
pred_year = np.arange(1990,2030)
pred_ind = curve_fun(pred_year,*param)
plt.plot(df_co2["year"],df_co2["Italy"],c="green", label="data")
plt.plot(pred_year,pred_ind,c="pink", label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("CO2")
plt.show()

param, cov = opt.curve_fit(curve_fun, df_co2["year"],df_co2["United States"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
error_range = err.error_prop(df_co2["year"],curve_fun,param,sigma)
mean = error_range.iloc[0]
up = mean + error_range.iloc[1]
low = mean - error_range.iloc[1]
df_co2["fit_value"] = curve_fun(df_co2["year"], * param)
# 3: Plotting co2 emission prediction for United state
plt.figure()
plt.title("United States CO2 emission prediction For 2030")
pred_year = np.arange(1990,2030)
pred_ind = curve_fun(pred_year,*param)
plt.plot(df_co2["year"],df_co2["United States"],c="green", label="data")
plt.plot(pred_year,pred_ind, c="pink", label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("CO2")
plt.show()
# 4: Renewable energy use as a percentage of total energy - Italy
param, cov = opt.curve_fit(curve_fun,df_renew["year"],df_renew["Italy"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
error_range= err.error_prop(df_renew["year"],curve_fun,param,sigma)
mean = error_range.iloc[0]
up = mean + error_range.iloc[1]
low = mean - error_range.iloc[1]
df_renew["fit_value"] = curve_fun(df_renew["year"], * param)
plt.figure()
plt.title("Renewable energy use as a percentage of total energy - Italy")
plt.plot(df_renew["year"],df_renew["Italy"], c="green", label="data")
plt.plot(df_renew["year"],df_renew["fit_value"], c="pink", label="fit")
plt.fill_between(df_renew["year"],low,up,alpha=0.3)
plt.legend()
plt.xlim(1990,2019)
plt.xlabel("Year")
plt.ylabel("Renewable energy(% of total energy use)")
plt.show()
#5) Renewable energy prediction - Italy
plt.figure()
plt.title("Renewable energy prediction - Italy")
pred_year = np.arange(1990,2030)
pred_ind = curve_fun(pred_year,*param)
plt.plot(df_renew["year"],df_renew["Italy"],c="green", label="data")
plt.plot(pred_year,pred_ind, c="pink", label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Renewable energy(% of total energy use)")
plt.show()
#6) Renewable energy prediction - Japan
param, cov = opt.curve_fit(curve_fun,df_renew["year"],df_renew["United States"],p0=[4e8, 0.1])
sigma = np.sqrt(np.diag(cov))
error_range= err.error_prop(df_renew["year"],curve_fun,param,sigma)
mean = error_range.iloc[0]
up = mean + error_range.iloc[1]
low = mean - error_range.iloc[1]
df_renew["fit_value"] = curve_fun(df_renew["year"], * param)
plt.figure()
plt.title("Renewable energy prediction - United States")
pred_year = np.arange(1990,2030)
pred_ind = curve_fun(pred_year,*param)
plt.plot(df_renew["year"],df_renew["United States"], c="green", label="data")
plt.plot(pred_year,pred_ind, c="pink", label="predicted values")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Renewable energy(% of total energy use)")
plt.show()
gdp(df_gdp)
# 8) Japan and Italy - CO2 Emission
df_co2= df_co2.iloc[:,1:3]
#Normalize data
df_co2_norm=(df_co2 - df_co2.mean()) / df_co2.std()
df_renew_norm=(df_renew - df_renew.mean()) / df_renew.std()
kmean = cluster.KMeans(n_clusters=4).fit(df_co2_norm)
label = kmean.labels_
plt.scatter(df_co2_norm["United States"],df_co2_norm["Italy"],c=label,cmap="coolwarm")
plt.title("United States and Italy - CO2 Emission")
plt.xlabel("Co2 emission of United States")
plt.ylabel("co2 emission of Italy")
c = kmean.cluster_centers_
plt.show()
#9) co2 emission vs renewable enery usage - Italy
italy = pd.DataFrame()
italy["co2_emission"] = df_co2_norm["Italy"]
italy["renewable_energy"] = df_renew_norm["Italy"]
kmean = cluster.KMeans(n_clusters=4).fit(italy)
label = kmean.labels_
plt.scatter(italy["co2_emission"], italy["renewable_energy"],c=label,cmap="coolwarm")
plt.title("co2 emission vs renewable enery usage - Italy")
plt.xlabel("co2 emission")
plt.ylabel("Renewable energy")
c = kmean.cluster_centers_
for t in range(2):
 xc,yc = c[t,:]
 plt.plot(xc,yc,"ok",markersize=8)
plt.figure()
plt.show()