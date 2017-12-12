import os
import csv
import time
import requests

from datetime import datetime


DATA_30M = 'https://bitcoincharts.com/charts/chart.json?m=bitstampUSD&SubmitButton=Draw&r=10&i=30-min&c=1&s=2017-11-01' + \
        '&e=2017-12-13&Prev=&Next=&t=S&b=&a1=&m1=10&a2=&m2=25&x=0&i1=&i2=&i3=&i4=&v=1&cv=0&ps=0&l=0&p=0&'
TICKER_URL = 'https://www.bitstamp.net/api/ticker/'

def main():
    # Create
    # create()

    try:
        while True:
            req = requests.get(TICKER_URL)
            body = req.json()
            writeCSV(body)

            # sleep(1800)
            time.sleep(10)
    except:
        pass

def convertTime(timestamp):
    d = datetime.fromtimestamp(timestamp)
    return d.strftime('%Y-%m-%d %H:%M')

def create():
    if not os.path.exists('./bitcoin.txt'):
        req = requests.get(DATA_30M)
        body = req.json()

        with open('./bitcoin.txt', 'w') as f:
            fieldnames = ['DATE', 'LAST']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for line in body[:-1]:
                writer.writerow({'DATE': convertTime(line[0]), 'LAST': line[1]})


def writeCSV(data):
    try:
        with open('./bitcoin.txt', 'a') as f:
            fieldnames = ['DATE', 'LAST']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'DATE': convertTime(datetime.now().timestamp()), 'LAST': data['last']})

    except:
        print("Error: Failed to write csv file.")

if __name__ == '__main__':
    main()
