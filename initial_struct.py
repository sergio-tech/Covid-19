import pandas as pd
import numpy as np
dataset = pd.read_csv("descarga.csv")
import matplotlib.pyplot as plt


x=0
Indices_China = []
Indices_Italy = []
Indices_Spain = []
Indices_South_Korea = []
Indices_Mexico = []

for country in dataset["countriesAndTerritories"]:
   if country == "China":
      Indices_China.append(x)
   elif country == "Italy":
      Indices_Italy.append(x)
   elif country == "Spain":
      Indices_Spain.append(x)
   elif country == "South_Korea":
      Indices_South_Korea.append(x)
   elif country == "Mexico": 
      Indices_Mexico.append(x)   
   x = x +1       
china_min = min(Indices_China)
china_max = max(Indices_China)
italy_min = min(Indices_Italy)
italy_max = max(Indices_Italy)
spain_min = min(Indices_Spain)
spain_max = max(Indices_Spain)
south_Korea_min = min(Indices_South_Korea)
south_Korea_max = max(Indices_South_Korea)
mexico_min = min(Indices_Mexico)
mexico_max = max(Indices_Mexico)

china = dataset.loc[china_min:china_max]
italy = dataset.loc[italy_min:italy_max]
spain = dataset.loc[spain_min:spain_max]
korea = dataset.loc[south_Korea_min:south_Korea_max]
mexico = dataset.loc[mexico_min:mexico_max]

china_cases = []
italy_cases = []
spain_cases = []
korea_cases = []
mexico_cases = []

for dato in china["cases"]:
   china_cases.append(dato)
china_cases.reverse()
for dato in italy["cases"]:
   italy_cases.append(dato)
italy_cases.reverse()
for dato in spain["cases"]:
   spain_cases.append(dato)
spain_cases.reverse()
for dato in korea["cases"]:
   korea_cases.append(dato)
korea_cases.reverse()
for dato in mexico["cases"]:
   mexico_cases.append(dato)
mexico_cases.reverse()



without_days_mexico = 0
with_days_mexico = 0
minncases_mexico = 0
ncases_mexico = 0
virus_mexico = []
total_cases_by_day_mexico = []
for cases in mexico_cases:
   ncases_mexico += cases
   if ncases_mexico > 20:
      with_days_mexico += 1
   else:
      without_days_mexico += 1
      minncases_mexico = ncases_mexico
range_mexico = range(without_days_mexico,(with_days_mexico + without_days_mexico - 1))
days_mexico = range(1,with_days_mexico)

for cases in range_mexico:
   virus_mexico.append(mexico_cases[cases])
total = 0
for cases in virus_mexico:
   total = total + cases
   total_cases_by_day_mexico.append(total)
print(total_cases_by_day_mexico)

"""
fig, axs = plt.subplots(1, 2, figsize=(9, 9), sharey=True)
axs[0].bar(days_mexico, virus_mexico,label = "Infected by day")
axs[1].plot(days_mexico, total_cases_by_day_mexico,label = "Infected")

axs[0].set_title('Infected by day')
axs[0].set_xlabel('Days with virus 0ver 20 cases')
axs[0].set_ylabel('Cases presented by day')
axs[0].set_ylim(0,200)
#plt.show()


axs[1].set_title('Infected')
axs[1].set_xlabel('Days with virus 0ver 20 cases')
axs[1].set_ylabel('Cases in Mexico')
axs[1].set_ylim(0,800)
fig.suptitle('Mexico \n Cases to 29/03/2020')
"""


fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Days with virus 0ver 20 cases')
ax1.set_ylabel('Infected by day', color=color)
ax1.bar(days_mexico, virus_mexico, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Total cases', color=color)  # we already handled the x-label with ax1
ax2.plot(days_mexico, total_cases_by_day_mexico, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
fig.suptitle('\n Mexico \n Cases to 29/03/2020')


plt.show()

