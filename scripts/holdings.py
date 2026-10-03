# from yfinance import ETFQuery, screen, EquityQuery
import yfinance as yf
import pprint
import csv 
import json 

# print('Data Query Start')
# etf =  ETFQuery('and', [
#     ETFQuery('gt', ['intradayprice', 10]),
#     ETFQuery('is-in', ['performanceratingoverall', 4, 5]),
#     ETFQuery('eq', ['categoryname', 'Technology'])
# ])
# data = screen(etf)

# with open('data.json', 'a') as f:
#   json.dump(data, f, indent=1)

# equ = EquityQuery('and', [
#     EquityQuery('is-in', ['exchange', 'NMS', 'NYQ']),
#     EquityQuery('gt', ['intradayprice', 10]),
#     EquityQuery('lt', ["epsgrowth.lasttwelvemonths", 15]),
#     EquityQuery('eq', ['sector', 'Technology'])
# ])

# data = screen(equ)

# with open('data2.json', 'a') as file:
#   json.dump(data, file, indent=1)


ts = yf.Ticker('UFOX')
data = ts.funds_data
pprint.pprint(data.top_holdings)