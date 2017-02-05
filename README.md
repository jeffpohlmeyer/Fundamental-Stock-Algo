# Fundamental-Stock-Algo

This project is a basic attempt at creating a fundamental data based stock algo using scikit-learn and xgboost.

Data is pulled for as many historical S&P500 constituent stocks back to 2006 as was available in Quandl.  A number of fundamental values was used proportional to price to determine when a stock is undervalued and thus is worthy of a buy.

If a stock that was marked as a buy hits a given percent profit before another given number of days then the stock was sold.  If the target was not hit before the number of days chosen then the stock was sold at the end of the period.  For loops were created to optimize these parameters.

Then a decent combination was determined and separate For loops were used to optimize parameters for use of xgboost's classifier.  This is the version that is uploaded.  Previous versions were not committed in the highly unlikely case that the model significantly outperformed the SPX, which it did not.

This was done over the month of January while on paternity leave and will likely not see significant development in the near future as I am now back at work.  Stock data is not uploaded as it is premium data on Quandl.
