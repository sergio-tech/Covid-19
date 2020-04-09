import pandas as pd
import numpy as np
import tensorflow as tf 
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


 

    print(days)
    print(virus)
    print(total_cases_by_day)

    x = days
    y = virus

    n = len(x) # Number of data points

    #Now we will start creating our model by defining the placeholders X and Y, so that we can feed our training examples X and Y into the optimizer during the training process.
    X = tf.placeholder("float") 
    Y = tf.placeholder("float") 
    #Now we will declare two trainable Tensorflow Variables for the Weights and Bias and initializing them randomly using np.random.randn().
    W = tf.Variable(np.random.randn(), name = "W") 
    b = tf.Variable(np.random.randn(), name = "b") 
    #Now we will define the hyperparameters of the model, the Learning Rate and the number of Epochs.
    learning_rate = 0.01
    training_epochs = 1000
    #Now, we will be building the Hypothesis, the Cost Function, and the Optimizer. We won’t be implementing the Gradient Descent Optimizer manually since it is built inside Tensorflow. After that, we will be initializing the Variables.
    # Hypothesis 
    y_pred = tf.add(tf.multiply(X, W), b) 
    # Mean Squared Error Cost Function 
    cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n) 
    # Gradient Descent Optimizer 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 
    # Global Variables Initializer 
    init = tf.global_variables_initializer() 
    #Now we will begin the training process inside a Tensorflow Session
    # Starting the Tensorflow Session 
    with tf.Session() as sess: 
        # Initializing the Variables 
        sess.run(init) 
        # Iterating through all the epochs 
        for epoch in range(training_epochs): 
            # Feeding each data point into the optimizer using Feed Dictionary 
            for (_x, _y) in zip(x, y): 
                sess.run(optimizer, feed_dict = {X : _x, Y : _y}) 
            # Displaying the result after every 50 epochs 
            if (epoch + 1) % 50 == 0: 
                # Calculating the cost a every epoch 
                c = sess.run(cost, feed_dict = {X : x, Y : y}) 
                print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b)) 
        # Storing necessary values to be used outside the Session 
        training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
        weight = sess.run(W) 
        bias = sess.run(b) 
    # Calculating the predictions 
    predictions = weight * x + bias 

    print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n') 
    #Output:
    #Training cost = 5.3110332 Weight = 1.0199214 bias = 0.02561663
    # Plotting the Results 































    
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel("Days with virus 0ver " + str(minc) +" cases"+"\n Data collected from:\n European Centre for Disease Prevention and Control \n https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide")
    ax1.set_ylabel('Infected by day', color=color)
    ax1.scatter(days, virus, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0,max(virus))
    """
    ax2 = ax1.twinx() 
    color = 'tab:red'
    ax2.set_ylabel('Total cases', color=color)
    ax2.plot(days, total_cases_by_day, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    """
    ax3 = ax1.twinx()
    color = 'tab:black'
    ax3.set_ylabel('Crecimiento', color= 'black')
    ax3.plot(x, predictions,color = 'black')
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.set_ylim(0,max(virus))

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