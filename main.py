import pandas as pd
import warnings
from pulp import *


def ground_handling_opt(dt_date, hand_funct, df_input, df_param_iter):

    # Disponiblity matrix preparation a

    df_time_table = pd.DataFrame()
    df_time_table['hour'] = list(range(6, 24))
    df_time_table['hour'] = df_time_table['hour']
    df_time_table['day'] = dt_date
    df_time_table['handling_function'] = hand_funct

    df_join = df_time_table.merge(df_input, how='left', left_on=['day', 'handling_function'],
                                  right_on=['dt_flight', 'handling_function'])

    df_join['required'] = df_join[(df_join['hour'] == df_join['ts_operation_start'].str[11:13].astype('int')) |
                         (df_join['hour'] == df_join['ts_operation_end'].str[11:13].astype('int'))]['required_employees']

    df_join = df_join.fillna(0)

    df_join_agg = df_join.groupby(['day', 'hour', 'handling_function'], as_index=False).agg({'required': sum})

    df_join_agg['hour'] = df_join_agg['hour'].astype('int')
    df_join_agg['required'] = df_join_agg['required'].astype('int')
    df_join_agg = df_join_agg.sort_values(['hour'])

    df_join_agg['FTM'] = 0
    df_join_agg['FTA'] = 0

    df_join_agg['FTM'][0:4] = 1
    df_join_agg['FTM'][5:9] = 1

    df_join_agg['FTA'][9:13] = 1
    df_join_agg['FTA'][14:18] = 1

    for i in range(1, 15):
        df_join_agg['PT' + str(i)] = 0

    for i in range(1, 15):
        df_join_agg['PT' + str(i)][i-1:i + 3] = 1

    a = df_join_agg[['FTM', 'FTA', 'PT1', 'PT2', 'PT3', 'PT4', 'PT5', 'PT6', 'PT7', 'PT8', 'PT9', 'PT10', 'PT11', 'PT12', 'PT13', 'PT14']]
    a = a.reset_index()
    a = a.drop('index', axis=1)

    # number of shifts
    n = a.shape[1]

    # number of time windows
    T = a.shape[0]

    # number of workers required per time window
    d = df_join_agg["required"].values
    d[d == 0] = 1

    # wage rate per shift
    w = pd.array([df_param_iter[0], df_param_iter[0], df_param_iter[1], df_param_iter[1], df_param_iter[1], df_param_iter[1], df_param_iter[1], df_param_iter[1], df_param_iter[1], \
                  df_param_iter[1], df_param_iter[1], df_param_iter[1], df_param_iter[1], df_param_iter[1], df_param_iter[1], df_param_iter[1]])

    # Decision variables
    y = LpVariable.dicts("num_workers", list(range(n)), lowBound=0, cat="Integer")

    # Objective: Minimize the total cost
    prob = LpProblem("scheduling_workers", LpMinimize)

    prob += lpSum([w[j] * y[j] for j in range(0, n)])

    for (i, t) in enumerate(a):
        prob += lpSum([a[t][j] * y[j] for j in range(0, n)]) >= d[i]

    prob.solve()
    print("Status:", LpStatus[prob.status])


    for (shift, name) in enumerate(a):
        for index, row in df_join_agg.iterrows():
            df_join_agg[name][index+4] = row[name] * int(y[shift].value())


    df_join_agg = df_join_agg.reset_index()
    df_join_agg = df_join_agg.drop('index', axis=1)


    df_result = df_join_agg[['day', 'hour', 'handling_function']]
    df_result['Full-time Employees'] = df_join_agg['FTM'] + df_join_agg['FTA']
    df_result['Part-time Employees'] = df_join_agg['PT1'] + df_join_agg['PT2'] + df_join_agg['PT3'] + df_join_agg['PT4'] + df_join_agg['PT5'] + df_join_agg['PT6'] + df_join_agg['PT7'] + \
                                       df_join_agg['PT8'] + df_join_agg['PT9'] + df_join_agg['PT10'] + df_join_agg['PT11'] + df_join_agg['PT12'] + df_join_agg['PT13'] + df_join_agg['PT14']

    df_result['Total Employees'] = df_result['Full-time Employees'] + df_result['Part-time Employees']

    return df_result


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    warnings.filterwarnings("ignore")

    df_input = pd.read_csv("input-ground-handling-optimizer2.csv")

    df_dates_fun = df_input[['dt_flight', 'handling_function']]
    df_dates_fun = df_dates_fun.drop_duplicates()

    df_param = pd.read_json('parameters.json')

    df_output = pd.DataFrame()

    for index, row in df_dates_fun.iterrows():

        df_param_iter = df_param[row['handling_function']]

        df = ground_handling_opt(row['dt_flight'], row['handling_function'], df_input, df_param_iter)

        df['Full-time Employees cost'] = df['Full-time Employees'] * df_param_iter[0]
        df['Full-part Employees cost'] = df['Part-time Employees'] * df_param_iter[1]

        df['Total cost'] = df['Full-time Employees cost'] + df['Full-part Employees cost']

        df_output = df_output.append(df)

    print('------------------------------------------------------------------------------------------------------------------')

    df_output.to_csv('output.csv', sep=';', index=False)



