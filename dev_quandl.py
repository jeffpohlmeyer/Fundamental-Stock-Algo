import quandl
import pandas as pd
import numpy as np
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from dateutil import parser
from pandas.io.data import DataReader                             # Module for getting data from Yahoo
import os
import time
import csv
import matplotlib
import matplotlib.pyplot as plt
from math import floor
from sklearn import preprocessing, cross_validation, svm, linear_model
import xgboost

os.system('clear')


def daterange(start_date, end_date, step):
	for n in range(0,int((end_date - start_date).days),step):
		yield start_date + timedelta(n)

def annualized_sharpe(returns,N=252):
	"""
	Calculate the annualised Sharpe ratio of a returns stream 
	based on a number of trading periods, N. N defaults to 252,
	which then assumes a stream of daily returns.

	The function assumes that the returns are the excess of 
	those compared to a benchmark.
	"""
	return np.sqrt(N) * returns.mean() / returns.std()

# Used for syntactic purposes
def single_line(df):
	for row in df.itertuples():
		if row[4] == 1:
			enter_date = row[1]
			ticker = row[2]
		if row[5] == 1:
			yield ticker, enter_date, row[1]

# Used for syntactic purposes
def transaction_convert(df,allotment):
	for row in df.itertuples():
		if row[4] == 1:
			date_val = row[1]
			price = row[3]
			quantity = floor(allotment / price)
			stock = row[2]
			trans = 'buy'
		if row[5] == 1:
			date_val = row[1]
			price = row[3]
			trans = 'sell'
		yield stock, trans, date_val, price, quantity

# Calculate the max capital to allot to one trade based on the max number of simultaneous positions in the training data
def allotment_calculator(dates,one_line):
	i = 0
	for date in dates['Date']:
		one_line['compare_date'] = date
		one_line['compare_date'] = pd.to_datetime(one_line['compare_date'])
		dates.ix[i,'Count'] = one_line[(one_line['Enter_Date'] <= one_line['compare_date']) & (one_line['Exit_Date'] >= one_line['compare_date'])]['Ticker'].count()
		i += 1
	max_simultaneous = dates['Count'].max()
	proportion = 1. / max_simultaneous
	return 35000. * proportion

# Tracker to determine if a trade is on
def date_up(df):
	in_trade = False
	Enter_Trade = 0
	for row in df.itertuples():
		if int(row[14]) == 0:
			Enter_Trade = 0
		elif int(row[14]) == 1 and Enter_Trade == 0 and in_trade == False:
			Enter_Trade = 1
		else:
			Enter_Trade = 0
		if Enter_Trade == 1:
			Trade_Date = row[1]
			Trade_Price = row[2]
			Target_Date = row[7]
			Target_Price = row[8]
			Ticker = row[3]
		elif Enter_Trade == 0 and in_trade == False:
			Trade_Date = 0
			Trade_Price = 0
			Target_Date = 0
			Target_Price = 0
			Ticker = 0
		if (row[1] == Target_Date or row[2] >= Target_Price and in_trade == True and row[3] == Ticker) or (row[1].date() == row[16].date() and (Enter_Trade == 1 or in_trade == 1)):
			Exit_Trade = 1
		else:
			Exit_Trade = 0
		if (Enter_Trade == 1 and Exit_Trade != 1) or (in_trade == True and Exit_Trade == 0):
			in_trade = True
		else:
			in_trade = False
		if Exit_Trade == 1:
			try:
				Profit_Percent = (row[2] / Trade_Price - 1) * 0.1
			except Exception as e:
				Profit_Percent = 0
		else:
			Profit_Percent = 0
		yield Enter_Trade, Trade_Date, Trade_Price, Target_Date, Target_Price, Exit_Trade, Profit_Percent, in_trade

# Tracker to calculate performance
def analysis(start_date,end_date):
	transactions = pd.read_csv('temp_file.csv')

	stocks = list(pd.unique(transactions.stock))

	portfolio_columns = ['Ticker','Quantity','Price','Value']

	prices = {}
	for stock in stocks:
		data_input = pd.read_csv('DataSets/'+stock+'.csv')
		column_name = data_input.columns[1]
		data_input = data_input.rename(columns = {column_name : 'Close'})[['Unnamed: 0','Close']]
		prices[stock] = data_input

	portfolio = pd.DataFrame(columns=portfolio_columns)

	cash = [['Cash',1.0,35000.0,35000.0]]

	portfolio = portfolio.append(pd.DataFrame(cash,columns = portfolio_columns))#,inplace=True)

	tracker = pd.DataFrame(daterange(start_date,end_date,1),columns=['Date'])
	tracker['Cash'] = 0
	tracker['Stocks'] = 0
	tracker['Total'] = 0
	tracker['sp'] = 0
	for date_index, date_row in tracker.iterrows():
		try:
			for index,row in transactions.iterrows():
				if parser.parse(row['date']).date() == date_row['Date']:
					transaction = [[row.stock,row.quantity,row.price,row.quantity * row.price]]
					if row['trans'] == 'buy':
						portfolio = portfolio.append(pd.DataFrame(transaction,columns = portfolio_columns),ignore_index=True)
						portfolio.ix[0,'Price'] = portfolio.ix[0,'Price'] - row.quantity * row.price
						portfolio.ix[0,'Value'] = portfolio.ix[0,'Price']
					if row['trans'] == 'sell':
						portfolio.drop(portfolio[portfolio['Ticker'] == row.stock].index,inplace=True)
						portfolio.ix[0,'Price'] = portfolio.ix[0,'Price'] + row.quantity * row.price
						portfolio.ix[0,'Value'] = portfolio.ix[0,'Price']
			for stock in list(portfolio['Ticker']):
				if stock != 'Cash':
					portfolio.ix[portfolio.loc[portfolio.Ticker == stock].index,'Price'] = float(prices[stock].loc[prices[stock]['Unnamed: 0'] == str(date_row['Date'])]['Close'])
					portfolio.ix[portfolio.loc[portfolio.Ticker == stock].index,'Value'] = portfolio.ix[portfolio.loc[portfolio.Ticker == stock].index,'Price'] * portfolio.ix[portfolio.loc[portfolio.Ticker == stock].index,'Quantity']
		except Exception as e:
			print date_row.Date,str(e)
		tracker.ix[date_index,'Cash'] = portfolio.ix[0,'Value']
		tracker.ix[date_index,'Stocks'] = portfolio.ix[1:,'Value'].sum()
		tracker.ix[date_index,'Total'] = tracker.ix[date_index,'Cash'] + tracker.ix[date_index,'Stocks']
	sp = pd.read_csv('sp.csv')
	sp.Date = pd.to_datetime(sp.Date)
	sp['start_date'] = start_date
	sp['end_date'] = end_date

	sp.start_date = pd.to_datetime(sp.start_date)
	sp.end_date = pd.to_datetime(sp.end_date)
	sp = sp[(sp['Date'] >= sp['start_date']) & (sp['Date'] < sp['end_date'])][['Date','Adj Close']]
	tracker['sp'] = sp['Adj Close']
	return tracker

start_date = date(2014,1,1)
end_date = date(2016,12,31)

# Parameter vectors for xgboost
eta_vec = [0.01, 0.05, 0.1, 0.15, 0.2]
m_depth_vec = [3, 5, 7, 9]
subsample_vec = [0.5, 0.75, 1]
colsample_vec = [0.5, 0.75, 1]

data_import = pd.read_csv('SF1_ZEP_SP500_Tickers.csv')

data_import.dropna(axis=0,inplace=True)

data_import.DateInSP500 = pd.to_datetime(data_import.DateInSP500)
data_import.DateOutSP500 = pd.to_datetime(data_import.DateOutSP500)

sectors = list(data_import.Sector.unique())

summary_vals = []
trade_summary_vals = []
for eta in eta_vec:
	for m_depth in m_depth_vec:
		for subsample in subsample_vec:
			for colsample in colsample_vec:
				print 'eta:',str(eta),'m_depth:',str(m_depth),'subsample:',str(subsample),'colsample:',str(colsample)
				for day_count in range(35,40,5):	# 35 days was one of the better performing periods
					for percent_vals in range(15,20,5):	# 15% was one of the better performing values
						trade_data = pd.DataFrame(columns=['Unnamed: 0','Ticker','CLOSE','Enter_Trade','Exit_Trade','Profit_Percent'])
						for sector in sectors:
							df = data_import[data_import.Sector == sector]
							values = []
							values.append('Unnamed: 0')
							values.append('CLOSE')
							values.append('HIGH')
							values.append('LOW')
							values.append('VOLUME')
							values.append('EPS_ART')
							values.append('EPSGROWTH1YR_ART')
							values.append('ASSETS_ARY')
							values.append('CASHNEQ_ARY')
							values.append('CURRENTRATIO_ARY')
							values.append('DE_ARY')
							values.append('DIVYIELD')
							values.append('EBITDA_ART')
							values.append('EQUITY_ARY')
							values.append('EV')
							values.append('FCF_ART')
							values.append('GROSSMARGIN_ART')
							values.append('INTEXP_ART')
							values.append('INVENTORY_ARY')
							values.append('NCF_ARY')
							values.append('NCFOGROWTH1YR_ART')
							values.append('NETINCCMN_ART')
							values.append('NETINCGROWTH1YR_ART')
							# values.append('PB_ARY')
							# values.append('PE_ART')
							values.append('PPNENET_ARY')
							# values.append('PS_ART')
							values.append('REVENUE_ART')
							values.append('REVENUEGROWTH1YR_ART')
							# values.append('ROA_ART')
							# values.append('ROE_ART')
							values.append('ROIC_ART')
							values.append('ROS_ART')
							values.append('SGNA_ARY')
							values.append('WORKINGCAPITAL_ARY')
							values.append('FXUSD')
							values.append('SHARESWA_ART')
							values.append('SHARESBAS')
							values.append('In')
							values.append('Out')
							values.append('Rolling_Max')
							values.append('Rolling_Min')
							values.append('Average_Volume')
							values.append('Up_Move')
							values.append('Down_Move')
							values.append('AdjPrice')
							values.append('AdjDate')
							values.append('Up_Target_Price')
							values.append('Up_Target')
							values.append('Down_Target')
							values.append('Ticker')

							mydata = pd.DataFrame(columns=values)

							target_value = percent_vals / 100.

							for row, SF_row, ZEP_row in zip(df.SP500,df.SF1,df.ZEP):
								try:
									file_name = 'DataSets/' + row + '.csv'
									csv_import = pd.read_csv(file_name)
									in_date = df[df.SP500 == row].DateInSP500.values[0]
									out_date = df[df.SP500 == row].DateOutSP500.values[0]

									csv_import['In'] = in_date
									csv_import['Out'] = out_date
									csv_import['Unnamed: 0'] = pd.to_datetime(csv_import['Unnamed: 0'])

									num_days = day_count

									csv_import['Rolling_Max_'+ZEP_row] = csv_import['ZEP/'+ZEP_row+' - HIGH'][::-1].shift(1).rolling(window=num_days,min_periods=num_days).max()[::-1]
									csv_import['Rolling_Min_'+ZEP_row] = csv_import['ZEP/'+ZEP_row+' - LOW'][::-1].shift(1).rolling(window=num_days,min_periods=num_days).min()[::-1]
									csv_import['Average_Volume_'+ZEP_row] = csv_import['ZEP/'+ZEP_row+' - VOLUME'].rolling(window=num_days,min_periods=num_days).mean()
									csv_import['Up_Move_'+ZEP_row] = csv_import['Rolling_Max_'+ZEP_row] / csv_import['ZEP/'+ZEP_row+' - CLOSE'].shift(-1) - 1
									csv_import['Down_Move_'+ZEP_row] = csv_import['Rolling_Min_'+ZEP_row] / csv_import['ZEP/'+ZEP_row+' - CLOSE'].shift(-1) - 1
									csv_import['AdjPrice_'+ZEP_row] = csv_import['ZEP/'+ZEP_row+' - CLOSE'].shift(-num_days)
									csv_import['AdjDate_'+ZEP_row] = csv_import['Unnamed: 0'].shift(-num_days)
									csv_import['Up_Target_Price'+ZEP_row] = (1+target_value) * csv_import['ZEP/'+ZEP_row+' - CLOSE'].shift(-1)
									csv_import = csv_import[(csv_import['Unnamed: 0'].values > csv_import['In'].values) & (csv_import['Unnamed: 0'].values < csv_import['Out'].values)]

									csv_import['Up_Target_'+row] = np.array(np.where(csv_import['Up_Move_'+ZEP_row]>=target_value,1,0))
									csv_import['Down_Target_'+row] = np.array(np.where(csv_import['Down_Move_'+ZEP_row]<=-target_value,1,0))
									csv_import['Ticker'] = row

									keys = csv_import.columns

									dictionary = dict(zip(keys,values))

									mydata_temp = csv_import
									mydata_temp.fillna(method='ffill',axis=0,inplace=True)
									mydata_temp.dropna(axis=0,inplace=True)
									mydata_temp = mydata_temp.rename(columns = dictionary)

									mydata_temp['ASSETS_ARY'] = mydata_temp['ASSETS_ARY'] / mydata_temp['SHARESWA_ART']
									mydata_temp['CASHNEQ_ARY'] = mydata_temp['CASHNEQ_ARY'] / mydata_temp['SHARESWA_ART']
									mydata_temp['EBITDA_ART'] = mydata_temp['EBITDA_ART'] / mydata_temp['SHARESWA_ART']
									mydata_temp['EQUITY_ARY'] = mydata_temp['EQUITY_ARY'] / mydata_temp['SHARESWA_ART']
									mydata_temp['EV'] = mydata_temp['EV'] / mydata_temp['SHARESWA_ART']
									mydata_temp['FCF_ART'] = mydata_temp['FCF_ART'] / mydata_temp['SHARESWA_ART']
									mydata_temp['NCF_ARY'] = mydata_temp['NCF_ARY'] / mydata_temp['SHARESWA_ART']
									mydata_temp['NETINCCMN_ART'] = mydata_temp['NETINCCMN_ART'] / mydata_temp['SHARESWA_ART']
									mydata_temp['PPNENET_ARY'] = mydata_temp['PPNENET_ARY'] / mydata_temp['SHARESWA_ART']
									mydata_temp['REVENUE_ART'] = mydata_temp['REVENUE_ART'] / mydata_temp['SHARESWA_ART']
									mydata_temp['SGNA_ARY'] = mydata_temp['SGNA_ARY'] / mydata_temp['SHARESWA_ART']
									mydata_temp['WORKINGCAPITAL_ARY'] = mydata_temp['WORKINGCAPITAL_ARY'] / mydata_temp['SHARESWA_ART']

									mydata_temp['EPS_ART'] = mydata_temp['EPS_ART'] / mydata_temp['CLOSE']
									mydata_temp['EPSGROWTH1YR_ART'] = mydata_temp['EPSGROWTH1YR_ART'] / mydata_temp['CLOSE']
									mydata_temp['ASSETS_ARY'] = mydata_temp['ASSETS_ARY'] / mydata_temp['CLOSE']
									mydata_temp['CASHNEQ_ARY'] = mydata_temp['CASHNEQ_ARY'] / mydata_temp['CLOSE']
									mydata_temp['CURRENTRATIO_ARY'] = mydata_temp['CURRENTRATIO_ARY'] / mydata_temp['CLOSE']
									mydata_temp['DE_ARY'] = mydata_temp['DE_ARY'] / mydata_temp['CLOSE']
									mydata_temp['DIVYIELD'] = mydata_temp['DIVYIELD'] / mydata_temp['CLOSE']
									mydata_temp['EBITDA_ART'] = mydata_temp['EBITDA_ART'] / mydata_temp['CLOSE']
									mydata_temp['EQUITY_ARY'] = mydata_temp['EQUITY_ARY'] / mydata_temp['CLOSE']
									mydata_temp['EV'] = mydata_temp['EV'] / mydata_temp['CLOSE']
									mydata_temp['FCF_ART'] = mydata_temp['FCF_ART'] / mydata_temp['CLOSE']
									mydata_temp['GROSSMARGIN_ART'] = mydata_temp['GROSSMARGIN_ART'] / mydata_temp['CLOSE']
									mydata_temp['INTEXP_ART'] = mydata_temp['INTEXP_ART'] / mydata_temp['CLOSE']
									mydata_temp['INVENTORY_ARY'] = mydata_temp['INVENTORY_ARY'] / mydata_temp['CLOSE']
									mydata_temp['NCF_ARY'] = mydata_temp['NCF_ARY'] / mydata_temp['CLOSE']
									mydata_temp['NCFOGROWTH1YR_ART'] = mydata_temp['NCFOGROWTH1YR_ART'] / mydata_temp['CLOSE']
									mydata_temp['NETINCCMN_ART'] = mydata_temp['NETINCCMN_ART'] / mydata_temp['CLOSE']
									mydata_temp['NETINCGROWTH1YR_ART'] = mydata_temp['NETINCGROWTH1YR_ART'] / mydata_temp['CLOSE']
									mydata_temp['PPNENET_ARY'] = mydata_temp['PPNENET_ARY'] / mydata_temp['CLOSE']
									mydata_temp['REVENUE_ART'] = mydata_temp['REVENUE_ART'] / mydata_temp['CLOSE']
									mydata_temp['REVENUEGROWTH1YR_ART'] = mydata_temp['REVENUEGROWTH1YR_ART'] / mydata_temp['CLOSE']
									mydata_temp['ROIC_ART'] = mydata_temp['ROIC_ART'] / mydata_temp['CLOSE']
									mydata_temp['ROS_ART'] = mydata_temp['ROS_ART'] / mydata_temp['CLOSE']
									mydata_temp['SGNA_ARY'] = mydata_temp['SGNA_ARY'] / mydata_temp['CLOSE']
									mydata_temp['WORKINGCAPITAL_ARY'] = mydata_temp['WORKINGCAPITAL_ARY'] / mydata_temp['CLOSE']

									mydata = mydata.append(mydata_temp,ignore_index=True)
									
								except Exception as e:
									print row, str(e)

							mydata['Train_Date'] = start_date
							mydata['Train_Date'] = pd.to_datetime(mydata['Train_Date'])

							try:
								train_data = mydata[(mydata['Unnamed: 0'].values < mydata['Train_Date'].values)]
								test_data = mydata[(mydata['Unnamed: 0'].values >= mydata['Train_Date'].values)]

								train_output_up = train_data['Up_Target']
								train_output_down = train_data['Down_Target']
								output_df = test_data[['Unnamed: 0','CLOSE','Ticker','Average_Volume','Up_Target','Down_Target','AdjDate','Up_Target_Price','Rolling_Max','Rolling_Min','Up_Move','Down_Move','Out']]

								output_df['Unnamed: 0'] = pd.to_datetime(output_df['Unnamed: 0'])
								output_df['AdjDate'] = pd.to_datetime(output_df['AdjDate'])
								train_data.drop(['HIGH','LOW','VOLUME','FXUSD','SHARESWA_ART','SHARESBAS','In','Out','Rolling_Min','Rolling_Max','Average_Volume','Up_Move','Down_Move','AdjDate','AdjPrice','Up_Target_Price'],axis=1,inplace=True)
								test_data.drop(['HIGH','LOW','VOLUME','FXUSD','SHARESWA_ART','SHARESBAS','In','Out','Rolling_Min','Rolling_Max','Average_Volume','Up_Move','Down_Move','AdjDate','AdjPrice','Up_Target_Price'],axis=1,inplace=True)

								test_output_up = test_data['Up_Target']
								test_output_down = test_data['Down_Target']
								train_data.drop(['Unnamed: 0','CLOSE','Train_Date','Ticker','Up_Target','Down_Target'],axis=1,inplace=True)
								test_data.drop(['Unnamed: 0','CLOSE','Train_Date','Ticker','Up_Target','Down_Target'],axis=1,inplace=True)

								scaler = preprocessing.StandardScaler().fit(train_data)
								train_data = scaler.transform(train_data)

								clf_up = xgboost.XGBClassifier(max_depth=m_depth,subsample=subsample,learning_rate=eta,colsample_bytree=colsample)
								clf_up.fit(train_data,train_output_up)
								up_result = clf_up.predict(scaler.transform(test_data))
								up_result = [round(value) for value in up_result]

								tickers = list(output_df.Ticker.unique())
								last_date = []
								for ticker in tickers:
									try:
										last_date.append(output_df[output_df['Ticker'] == ticker]['Unnamed: 0'].iloc[-1])
									except Exception as e:
										print str(e)

								date_dict = dict(zip(tickers,last_date))


								output_df['Up Result'] = up_result

								output_df['Down Result'] = 0

								output_df.reset_index(inplace=True)
								output_df.drop(['index'],inplace=True,axis=1)

								output_df['Last_Date'] = output_df['Ticker'].map(date_dict)

								trade_df_columns = ['Enter_Trade', 'Trade_Date', 'Trade_Price', 'Target_Date', 'Target_Price', 'Exit_Trade', 'Profit_Percent', 'In_Trade']

								trade_df = pd.concat([output_df,pd.DataFrame(date_up(output_df),columns=trade_df_columns)],axis=1,join='outer')

								profit = trade_df['Profit_Percent'].sum()
								max_win = trade_df['Profit_Percent'].max()
								max_loss = trade_df['Profit_Percent'].min()
								print 'Sector:',sector,'Days:',num_days,'Target:',target_value,'Profit:','%.5f' % profit,'Max Win:','%.5f' % max_win,'Max Loss:','%.5f' % max_loss
								trade_data = trade_data.append(trade_df[(trade_df['Enter_Trade'] == 1) | (trade_df['Exit_Trade'] == 1)][['Unnamed: 0','Ticker','CLOSE','Enter_Trade','Exit_Trade','Profit_Percent']],ignore_index=True)
						
								summary_vals.append([sector,num_days,target_value,profit,max_win,max_loss])
							except Exception as e:
								print str(e)

						dates = pd.DataFrame(daterange(start_date,end_date,1),columns=['Date'])
						dates['Count'] = 0
						one_line = pd.DataFrame(single_line(trade_data),columns = ['Ticker','Enter_Date','Exit_Date'])
						allotment = allotment_calculator(dates,one_line)
						del dates, one_line

						transactions = pd.DataFrame(transaction_convert(trade_data,allotment),columns=['stock','trans','date','price','quantity'])

						transactions.to_csv('temp_file.csv',sep=',',index=False)	# Done to get the dates in text format
						del transactions

						tracker = analysis(start_date,end_date)
						tracker['port_daily_ret'] = tracker['Total'].pct_change()
						tracker['port_excess_daily_ret'] = tracker['port_daily_ret'] - 0.005/252
						port_sharpe = annualized_sharpe(tracker['port_excess_daily_ret'])
						tracker['sp_daily_ret'] = tracker['sp'].pct_change()
						tracker['sp_excess_daily_ret'] = tracker['sp_daily_ret'] - 0.005/252
						sp_sharpe = annualized_sharpe(tracker['sp_excess_daily_ret'])
						window = 252
						roll_max = pd.rolling_max(tracker['Total'],window,min_periods=1)
						daily_drawdown = tracker['Total']/roll_max - 1.0
						max_daily_drawdown = pd.rolling_min(daily_drawdown,window,min_periods=1)
						tracker['port_dd'] = daily_drawdown
						tracker['port_max_dd'] = max_daily_drawdown
						port_return = tracker.iloc[-1]['Total'] / tracker.iloc[0]['Total']
						sp_return = tracker.iloc[-1]['sp'] / tracker.iloc[0]['sp']

						trade_summary_val = [eta,m_depth,subsample,colsample,target_value,num_days,port_return,sp_return,port_sharpe,sp_sharpe,tracker['port_dd'].min()]
						# trade_data.to_csv('Target_Percent_'+str(target_value)+'_Number_of_Days_'+str(num_days)+'.csv')
						trade_summary_vals.append(trade_summary_val)
						print trade_summary_val
						# trade_df.to_csv('output_file_sector_'+sector+'.csv',sep=',')

	# summary_data = pd.DataFrame(summary_vals,columns=['Sector','Day_Count','Target_Percent','Profit','Max_Win','Max_Loss'])
	# summary_data.to_csv('Summary Data.csv',sep=',')
trade_summary_data = pd.DataFrame(trade_summary_vals,columns=['eta','m_depth','subsample','colsample','Target_Percent','Num_Days','Portfolio_Return','SP_Return','Portfolio_Sharpe','SP_Sharpe','Max_DD'])
trade_summary_data.to_csv('Trade_Summary_Data.csv',sep=',')
print trade_summary_data