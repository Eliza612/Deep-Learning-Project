# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 13:31:51 2018

@author: Huaihan Chen
"""

import pandas as pd
import gzip

def back_test(name, Predict_df, data, Percent, one_slot):
    """
    name: the name of the processing future
    Predict_df: the result of prediction dataframe
    data: the price information of processing future
    This is long only (We can only enter into a position)
    """
    
    l = ['OrderType', 'FuturePrice', 'OrderPrice', 'RemainingMoney', 'Position', 'ActualMoney']
    #globals()[name + '_result'] = pd.DataFrame(index=Predict_df.index, columns=l)
    rec = pd.DataFrame(index=Predict_df.index, columns=l)
    RemainMoney = 1000000
    CumPosition = 0
    Position = 0
    ActualMoney = 1000000
    for index, row in data.iterrows():
#         if row['indicator'] == 0:
#             # this row data is not available 
#             continue
#         else:
        if Predict_df.loc[index][name] == 1:
            # There is positive signal
            FuturePrice = row['close']
            OrderPrice = FuturePrice * Percent * one_slot

            if Position != -1:
                # We currently don't take a short position
                if RemainMoney - OrderPrice > 0: # check if we have enough money
                    Position = 1
                    CumPosition += one_slot
                    RemainMoney -= OrderPrice
                    # ActualMoney is to calculate PnL
                    ActualMoney -= FuturePrice * one_slot    
                    rec.loc[index] = ['long', FuturePrice, OrderPrice, 
                                      RemainMoney, CumPosition, ActualMoney]
                else:
                    continue
            else:
                # We already have a short position
                Position = 1
                ActualMoney += CumPosition * FuturePrice # Close the short position
                RemainMoney = ActualMoney - OrderPrice   # Money we need to pay for future
                ActualMoney -= FuturePrice * one_slot    
                CumPosition = one_slot
                rec.loc[index] = ['long', FuturePrice, OrderPrice, 
                                      RemainMoney, CumPosition, ActualMoney]

        elif Predict_df.loc[index][name] == 0:
            # make no action
            FuturePrice = row['close']
            rec.loc[index] = ['hold', FuturePrice, 0, RemainMoney, CumPosition, ActualMoney]

        elif Predict_df.loc[index][name] == -1:
            # There is a negative signal
            FuturePrice = row['close']
            OrderPrice = FuturePrice * Percent * one_slot

            if Position != 1:
                # Notice that even we open a short position, we still need to pay margin
                if RemainMoney - OrderPrice > 0:
                    Position = -1
                    CumPosition -= one_slot
                    RemainMoney -= OrderPrice
                    ActualMoney += FuturePrice * one_slot
                    rec.loc[index] = ['short', FuturePrice, OrderPrice, 
                                      RemainMoney, CumPosition, ActualMoney]
                else:
                    continue
            else:                    
                # Close the long position and take a short one
                Position = -1
                ActualMoney += CumPosition * FuturePrice # Close the short position
                RemainMoney = ActualMoney - OrderPrice   # Money we need to pay for future
                ActualMoney += FuturePrice * one_slot    
                CumPosition = -one_slot
                rec.loc[index] = ['short', FuturePrice, OrderPrice, 
                                      RemainMoney, CumPosition, ActualMoney]
    
    rec['PnL'] = rec['ActualMoney'] + rec['Position'] * rec['FuturePrice']
    
    return rec


if __name__ == "__main__":
    price = pd.read_csv("../project/Data_cleaned/A_cleaned.csv", index_col = 0, parse_dates = True)
    f=pd.read_csv('../project/trial1_df/signal_1.gz',index_col = 0,  parse_dates = True)
    name = "A"
    A = back_test(name, f, price.loc[f.index], 0.05, 10)





