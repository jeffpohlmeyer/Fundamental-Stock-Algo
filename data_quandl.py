import quandl
import pandas as pd
import numpy as np
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import os
import time
import csv

os.system('clear')

# quandl.ApiConfig.api_key = *******

df = pd.read_csv('SF1_ZEP_SP500_Tickers.csv')
# df = pd.read_csv('error_list.csv')

df.dropna(axis=0,inplace=True)

error_list = []

for SP500_row, SF_row, ZEP_row in zip(df['SP500'],df['SF1'], df['ZEP']):
	print 'S&P 500 Ticker:',SP500_row+',','SF1 Ticker:',SF_row+',',"Zack's Ticker:",ZEP_row
	try:
		data_list = []
		column_list = []
		data_list.append('ZEP/'+ZEP_row)
		data_list.append('SF1/'+SF_row+'_EPS_ART')
		data_list.append('SF1/'+SF_row+'_EPSGROWTH1YR_ART')
		data_list.append('SF1/'+SF_row+'_ASSETS_ARY')
		data_list.append('SF1/'+SF_row+'_CASHNEQ_ARY')
		data_list.append('SF1/'+SF_row+'_CURRENTRATIO_ARY')
		data_list.append('SF1/'+SF_row+'_DE_ARY')
		data_list.append('SF1/'+SF_row+'_DIVYIELD')
		data_list.append('SF1/'+SF_row+'_EBITDA_ART')
		data_list.append('SF1/'+SF_row+'_EQUITY_ARY')
		data_list.append('SF1/'+SF_row+'_EV')
		data_list.append('SF1/'+SF_row+'_FCF_ART')
		data_list.append('SF1/'+SF_row+'_GROSSMARGIN_ART')
		data_list.append('SF1/'+SF_row+'_INTEXP_ART')
		data_list.append('SF1/'+SF_row+'_INVENTORY_ARY')
		data_list.append('SF1/'+SF_row+'_NCF_ARY')
		data_list.append('SF1/'+SF_row+'_NCFOGROWTH1YR_ART')
		data_list.append('SF1/'+SF_row+'_NETINCCMN_ART')
		data_list.append('SF1/'+SF_row+'_NETINCGROWTH1YR_ART')
		data_list.append('SF1/'+SF_row+'_NETMARGIN_ART')
		# data_list.append('SF1/'+SF_row+'_PB_ARY')
		# data_list.append('SF1/'+SF_row+'_PE_ART')
		data_list.append('SF1/'+SF_row+'_PPNENET_ARY')
		# data_list.append('SF1/'+SF_row+'_PS_ART')
		data_list.append('SF1/'+SF_row+'_REVENUE_ART')
		data_list.append('SF1/'+SF_row+'_REVENUEGROWTH1YR_ART')
		# data_list.append('SF1/'+SF_row+'_ROA_ART')
		# data_list.append('SF1/'+SF_row+'_ROE_ART')
		data_list.append('SF1/'+SF_row+'_ROIC_ART')
		data_list.append('SF1/'+SF_row+'_ROS_ART')
		data_list.append('SF1/'+SF_row+'_SGNA_ARY')
		data_list.append('SF1/'+SF_row+'_WORKINGCAPITAL_ARY')
		data_list.append('SF1/'+SF_row+'_FXUSD')
		data_list.append('SF1/'+SF_row+'_SHARESWA_ART')
		data_list.append('SF1/'+SF_row+'_SHARESBAS')

		column_list.append('ZEP/'+ZEP_row+' - CLOSE')
		column_list.append('ZEP/'+ZEP_row+' - HIGH')
		column_list.append('ZEP/'+ZEP_row+' - LOW')
		column_list.append('ZEP/'+ZEP_row+' - VOLUME')
		column_list.append('SF1/'+SF_row+'_EPS_ART - Value')
		column_list.append('SF1/'+SF_row+'_EPSGROWTH1YR_ART - Value')
		column_list.append('SF1/'+SF_row+'_ASSETS_ARY - Value')
		column_list.append('SF1/'+SF_row+'_CASHNEQ_ARY - Value')
		column_list.append('SF1/'+SF_row+'_CURRENTRATIO_ARY - Value')
		column_list.append('SF1/'+SF_row+'_DE_ARY - Value')
		column_list.append('SF1/'+SF_row+'_DIVYIELD - Value')
		column_list.append('SF1/'+SF_row+'_EBITDA_ART - Value')
		column_list.append('SF1/'+SF_row+'_EQUITY_ARY - Value')
		column_list.append('SF1/'+SF_row+'_EV - Value')
		column_list.append('SF1/'+SF_row+'_FCF_ART - Value')
		column_list.append('SF1/'+SF_row+'_GROSSMARGIN_ART - Value')
		column_list.append('SF1/'+SF_row+'_INTEXP_ART - Value')
		column_list.append('SF1/'+SF_row+'_INVENTORY_ARY - Value')
		column_list.append('SF1/'+SF_row+'_NCF_ARY - Value')
		column_list.append('SF1/'+SF_row+'_NCFOGROWTH1YR_ART - Value')
		column_list.append('SF1/'+SF_row+'_NETINCCMN_ART - Value')
		column_list.append('SF1/'+SF_row+'_NETINCGROWTH1YR_ART - Value')
		# column_list.append('SF1/'+SF_row+'_PB_ARY - Value')
		# column_list.append('SF1/'+SF_row+'_PE_ART - Value')
		column_list.append('SF1/'+SF_row+'_PPNENET_ARY - Value')
		# column_list.append('SF1/'+SF_row+'_PS_ART - Value')
		column_list.append('SF1/'+SF_row+'_REVENUE_ART - Value')
		column_list.append('SF1/'+SF_row+'_REVENUEGROWTH1YR_ART - Value')
		# column_list.append('SF1/'+SF_row+'_ROA_ART - Value')
		# column_list.append('SF1/'+SF_row+'_ROE_ART - Value')
		column_list.append('SF1/'+SF_row+'_ROIC_ART - Value')
		column_list.append('SF1/'+SF_row+'_ROS_ART - Value')
		column_list.append('SF1/'+SF_row+'_SGNA_ARY - Value')
		column_list.append('SF1/'+SF_row+'_WORKINGCAPITAL_ARY - Value')
		column_list.append('SF1/'+SF_row+'_FXUSD - Value')
		column_list.append('SF1/'+SF_row+'_SHARESWA_ART - Value')
		column_list.append('SF1/'+SF_row+'_SHARESBAS - Value')
		

		mydata = quandl.get(data_list, start_date='2006-1-1')

		mydata = mydata[column_list]

		mydata.fillna(method='ffill',axis=0,inplace=True)

		mydata.to_csv('DataSets/'+SP500_row+'.csv',sep=',')
	except Exception as e:
		print str(e)
		error_list.append(SP500_row)

resultFile = open('error_list.csv','wb')
wr = csv.writer(resultFile,dialect='excel')
wr.writerow(error_list)
resultFile.close()