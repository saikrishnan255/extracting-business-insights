# --------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(path)


def visual_summary(type_, df, col):
    df[col].plot(kind = type_)
    plt.show()    
    """Summarize the Data using Visual Method.
    
    This function accepts the type of visualization, the data frame and the column to be summarized.
    It displays the chart based on the given parameters.
    
    Keyword arguments:
    type_ -- visualization method to be used
    df -- the dataframe
    col -- the column in the dataframe to be summarized
    """
    

def central_tendency(type_, df, col):
    stats = {'mean': np.mean,'median': np.median, 'mode': st.mode}
    return stats[type_](df[col])
    """Calculate the measure of central tendency.
    
    This function accepts the type of central tendency to be calculated, the data frame and the required column.
    It returns the calculated measure.
    
    Keyword arguments:
    type_ -- type of central tendency to be calculated
    df -- the dataframe
    col -- the column in the dataframe to do the calculations
    
    Returns:
    cent_tend -- the calculated measure of central tendency
    """

def ranger(df):
    return max(df) - min(df)

def mad(df):
    return(np.mean(np.absolute(df - np.mean(df))))

def cv(df):
    return(((np.std(df)/np.mean(df)))*100)

def iqr(df):
    return (np.percentile(df,75)- np.percentile(df,25))  

def measure_of_dispersion(type_, df, col):
    stats = {'Standard Deviation':np.std,'Variance':np.var,'Range':ranger,'Covariance':np.cov,'MAD':mad,'CV':cv,'IQR':iqr}
    return stats[type_](df[col])
    """Calculate the measure of dispersion.
    
    This function accepts the measure of dispersion to be calculated, the data frame and the required column(s).
    It returns the calculated measure.
    
    Keyword arguments:
    type_ -- type of central tendency to be calculated
    df -- the dataframe
    col -- the column(s) in the dataframe to do the calculations, this is a list with 2 elements if we want to calculate covariance
    
    Returns:
    disp -- the calculated measure of dispersion
    """

def calculate_correlation(type_, df, col1, col2):
    if type_ == 'Pearson':
        return (df.cov().loc[col1,col2])/(np.std(df[col1])*np.std(df[col2]))
    elif type_ == 'Spearman':
        d = df[[col1,col2]].rank(axis = 0)
        d['d^2'] = (d[col1] - d[col2])**2
        d_square = d['d^2'].sum()
        l = len(df[col1])
        spearman = 1-((6*d_square)/(l*(l**2-1)))
        return spearman        
    """Calculate the defined correlation coefficient.
    
    This function accepts the type of correlation coefficient to be calculated, the data frame and the two column.
    It returns the calculated coefficient.
    
    Keyword arguments:
    type_ -- type of correlation coefficient to be calculated
    df -- the dataframe
    col1 -- first column
    col2 -- second column
    
    Returns:
    corr -- the calculated correlation coefficient
    """
    


def calculate_probability_discrete(data, event):
    crisis = df[event].value_counts()
    return(crisis.iloc[1]/(crisis.iloc[0] + crisis.iloc[1]))
    """Calculates the probability of an event from a discrete distribution.
    
    This function accepts the distribution of a variable and the event, and returns the probability of the event.
    
    Keyword arguments:
    data -- series that contains the distribution of the discrete variable
    event -- the event for which the probability is to be calculated
    
    Returns:
    prob -- calculated probability fo the event
    """
   

def event_independence_check(prob_event1, prob_event2, prob_event1_event2):
    pa_b = prob_event1_event2/prob_event2
    if pa_b == prob_event1:
        return 'Independent'
    elif pa_b != prob_event1:
        return 'Dependent'
    """Checks if two events are independent.
    
    This function accepts the probability of 2 events and their joint probability.
    And prints if the events are independent or not.
    
    Keyword arguments:
    prob_event1 -- probability of event1
    prob_event2 -- probability of event2
    prob_event1_event2 -- probability of event1 and event2
    """
# Checking if banking crisis is independent
b_s = df[(df['systemic_crisis'] == 1) & (df['banking_crisis'] == 'crisis')]
b_i = df[(df['inflation_crises'] == 1) & (df['banking_crisis'] == 'crisis')]
b_c = df[(df['currency_crises'] == 1) & (df['banking_crisis'] == 'crisis')]

p_bank_system = b_s['case'].count()/df['case'].count()
p_bank_currency = b_c['case'].count()/df['case'].count()
p_bank_inflation = b_i['case'].count()/df['case'].count()

p_bank = calculate_probability_discrete(df,'banking_crisis')
p_system = calculate_probability_discrete(df,'systemic_crisis')
p_inflation = calculate_probability_discrete(df,'inflation_crises')
p_currency = calculate_probability_discrete(df,'currency_crises')

# System
event_independence_check(p_bank, p_system, p_bank_system)
# Currency
event_independence_check(p_bank, p_currency, p_bank_currency)
# Inflation
event_independence_check(p_bank, p_inflation, p_bank_inflation)

# Bank given system
p_b_s = p_bank_system/p_system
p_b_c = p_bank_currency/p_currency
p_b_i = p_bank_inflation/p_inflation

prob_ = [p_b_s,p_b_c,p_b_i]
    
    
def bayes_theorem(df, col1, event1, col2, event2):
    """Calculates the conditional probability using Bayes Theorem.
    
    This function accepts the dataframe, two columns along with two conditions to calculate the probability, P(B|A).
    You can call the calculate_probability_discrete() to find the basic probabilities and then use them to find the conditional probability.
    
    Keyword arguments:
    df -- the dataframe
    col1 -- the first column where the first event is recorded
    event1 -- event to define the first condition
    col2 -- the second column where the second event is recorded
    event2 -- event to define the second condition
    
    Returns:
    prob -- calculated probability for the event1 given event2 has already occured
    """
    




