import os
import time
import requests

DATA_30M = 'https://bitcoincharts.com/charts/chart.json?m=bitstampUSD&SubmitButton=Draw&r=10&i=30-min&c=0&' + \
           's=&e=&Prev=&Next=&t=S&b=&a1=&m1=10&a2=&m2=25&x=0&i1=&i2=&i3=&i4=&v=1&cv=0&ps=0&l=0&p=0&'
TICKER_URL = 'https://www.bitstamp.net/api/ticker/'

def main():
    # Create
    create()

    try:
        while True:
            req = requests.get(TICKER_URL)
            body = req.json()
            writeCSV(body['last'])

            # sleep(1800)
            time.sleep(1800)
    except:
        pass

def create():
    if not os.path.exists('./bitcoin.txt'):
        req = requests.get(DATA_30M)
        body = req.json()

        with open('./bitcoin.txt', 'w') as f:
            f.write('LAST\n')

            for line in body:
                f.write("{}\n".format(line[1]))


def writeCSV(data):
    try:
        with open('./test.txt', 'a') as f:
            f.write("{}\n".format(data))
    except:
        print("Error: Failed to write csv file.")

if __name__ == '__main__':
    main()