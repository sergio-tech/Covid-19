# del histórico de casos positivos de COVID-19 
# por cada 100,000 habitantes 
# a partir de 100 casos en México.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Graficador(objective,minc,date,limitdays):
    print(objective + "   "+str(minc)+"   "+date+"   "+str(limitdays))
    dataset = pd.read_csv("data/"+date+".csv")
    x=0
    indices_pais = []
    for country in dataset["countriesAndTerritories"]:
        if country == objective:
            indices_pais.append(x)
        x = x + 1  
    for v in dataset["popData2018"]:
        poblacion = v
    country_min = min(indices_pais)
    country_max = max(indices_pais)
    country = dataset.loc[country_min:country_max]
    country_cases = []
    for dato in country["cases"]:
        country_cases.append(dato)
    country_cases.reverse()

    without_days = 0
    with_days = 0
    minncases = 0
    ncases = 0
    virus = []
    total_cases_by_day = []
    
    for cases in country_cases:
        ncases += cases
        if ncases > minc:
          with_days += 1
        else:
          without_days += 1
          minncases = ncases

    range_country = range(without_days,(with_days + without_days - 1))
    ###########################################################################
    count = 1
    if limitdays == 0:
        days = range(1,with_days)

        for cases in range_country:
            virus.append(country_cases[cases])        
        
        total = 0
        counter = 0
        for cases in virus:
            if counter == 0:
                total = total + cases + minncases
                total_cases_by_day.append(total)
                counter = 1
            else:    
                total = total + cases
                total_cases_by_day.append(total)
    else:        
        if limitdays <= with_days: 
            days = range(1,limitdays+1)
        else:
            print("Cantidad de dias muy grande")
        
        for cases in range_country:
            if count <= limitdays:
                virus.append(country_cases[cases])
                count += 1

        total = 0
        counter = 0
        for cases in virus:
            if counter == 0:
                total = total + cases + minncases
                total_cases_by_day.append(total)
                counter = 1
            else:    
                total = total + cases
                total_cases_by_day.append(total)
    for v in days:
        virus[v-1]=virus[v-1]*100000/poblacion 
        total_cases_by_day[v-1]=total_cases_by_day[v-1]*100000/poblacion

    print(days)
    print(virus)
    print(total_cases_by_day)


    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel("Days with virus 0ver " + str(minc) +" cases"+"\n Data collected from:\n European Centre for Disease Prevention and Control \n https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide")
    ax1.set_ylabel('Infected by day each 100000 people', color=color)
    ax1.bar(days, virus, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Total cases each 100000 people', color=color)  # we already handled the x-label with ax1
    ax2.plot(days, total_cases_by_day, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle("\n " + objective +"\n Cases at "+date)

    
    plt.show()

#array with coutries to evaluate
countries = ["Mexico"]
#number of the minimal cases to show in graphic
from_minimal_cases = 100
#date to evaluate 
date_csv= "08_04_2020"
#maximal day to show in graphic from the minimal cases detected
number_of_days_to_compare = 0

for paises in countries:
    Graficador(paises,from_minimal_cases,date_csv,number_of_days_to_compare)