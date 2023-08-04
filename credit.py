import numpy as np
import pandas as pd

def upper_outlier(df,col): # Finds the upper bound outlier of dataframe
    return df[col].describe()[6]+(df[col].describe()[6]-df[col].describe()[4])*1.5

def lower_outlier(df,col): # Finds the lower bound outlier of dataframe
    return df[col].describe()[4]-(df[col].describe()[6]-df[col].describe()[4])*1.5

def df_transform(df): # transforms the dataframe into one fit for model prediction
    # Drop unnecessary cols
    col_to_drop = ['Age','SSN','ID','Customer_ID','Name','Month','Occupation',
                   'Num_Credit_Inquiries','Monthly_Inhand_Salary','Type_of_Loan']
    df = df.drop(columns=col_to_drop)
    
    # Lower case column names and changes object types into either float or int
    df.columns = df.columns.str.lower()
    float_cols = ['annual_income','changed_credit_limit','outstanding_debt','amount_invested_monthly','monthly_balance']
    int_cols = ['num_of_loan','num_of_delayed_payment']
    df[float_cols] = df[float_cols].astype(str).replace({'_':'','':np.nan},regex=True).astype(float)
    df[int_cols] = df[int_cols].astype(str).replace({'_':'','':np.nan,'nan':np.nan},regex=True).astype(float).astype('Int64')
    df['credit_history_age'] = df['credit_history_age'].str[:2].astype('float').astype('Int64')
    
    # Drops Null rows
    df = df.dropna(subset = ['num_of_delayed_payment','changed_credit_limit','credit_history_age',
                              'amount_invested_monthly','monthly_balance'])
    
    # One hot encoding for some object columns which can be useful
    df = pd.get_dummies(df, columns=['payment_behaviour','credit_mix','payment_of_min_amount'], prefix="", prefix_sep="")
    
    # Get rid of the outliers and unrealistic values using the functions made above
    df = df[(df['num_of_delayed_payment']>=0) & (df['delay_from_due_date']>=0)]
    
    df = df[(df['num_bank_accounts']>=0) & (df['num_bank_accounts']<=upper_outlier(df,'num_bank_accounts')) &
             (df['num_credit_card']>=0) & (df['num_credit_card']<=upper_outlier(df,'num_credit_card')) &
             (df['interest_rate']>=0) & (df['interest_rate']<=upper_outlier(df,'interest_rate')) &
             (df['num_of_loan']>=0) & (df['num_of_loan']<=upper_outlier(df,'num_of_loan')) &
             (df['num_of_delayed_payment']>=0) & (df['num_of_delayed_payment']<=upper_outlier(df,'num_of_delayed_payment'))]
    return df