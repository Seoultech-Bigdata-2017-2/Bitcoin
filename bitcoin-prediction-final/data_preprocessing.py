import json
import csv

from datetime import datetime

data = json.load(open('./bitcoin_hour.json'))


fieldnames = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Currency', 'Weighted Price']
with open('./bitcoin-hour.csv', 'w', newline='') as csvfile:
	writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
	writer.writeheader()

	for item in data:
		if not item[1] == 1.7e+308:
			convert = datetime.fromtimestamp(item[0]).strftime('%Y-%m-%d')

			writer.writerow({'Date':convert, 'Open':item[1], 'High':item[2], 'Low':item[3], 'Close':item[4], 'Volume':item[5], 'Currency':item[6], 'Weighted Price':item[7]})