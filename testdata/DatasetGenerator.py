import csv
import numpy as np

NUMBER_OF_PREVIOUS_PRICES = 10

years = []
months = []
days = []
prices = []
temperature = []
rainfall = []
snow = []

def readSpotPrices():
    with open('elspotprices.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            date = row[0].split("/")
            years.append(date[2]);
            months.append(date[1]);
            days.append(date[0]);
            prices.append(float(row[1].replace(',','.')))
        
def readTemperature(): 
    with open('temperature.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            oslo = float(row[0])
            bergen = float(row[1])
            trondheim = float(row[2])
            tromso = float(row[3])
            temperature.append([oslo, bergen,trondheim, tromso])

def readRainfall():
    with open('rain.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            tronstad_rain = float(row[0])
            kvildal_rain = float(row[1])
            rainfall.append([tronstad_rain, kvildal_rain])
            
def readSnow():
    with open('snow.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            tronstad_snow = float(row[0])
            kvildal_snow = float(row[1])
            snow.append([tronstad_snow, kvildal_snow])
            
def getAccumulatedRainfallFromIndex(i):
    accumulated = []
    for x in range(0, len(rainfall[i])):
        single_location = [row[x] for row in rainfall]
        accumulateLastWeek = np.sum(single_location[i-7:i])
        accumulateLastMonth = np.sum(single_location[i-30:i])
        accumulated.append(accumulateLastWeek)
        accumulated.append(accumulateLastMonth)
    return accumulated
    
def getPreviousPricesFromIndex(i):
    previous_prices = []
    for x in range(1, NUMBER_OF_PREVIOUS_PRICES+1):
        previous_prices.append(prices[i-x])
    return reversed(previous_prices)

def generateRow(index):
    row = []
    previous_prices = getPreviousPricesFromIndex(index)
    accumulatedRainfall = getAccumulatedRainfallFromIndex(index)
    row.append(years[index])
    row.append(months[index])
    row.append(days[index])
    row.extend(temperature[index])
    row.extend(rainfall[index])
    row.extend(accumulatedRainfall)
    row.extend(snow[index])
    row.extend(previous_prices)
    row.append(prices[index])
    row.append(prices[index+1])
    return row


def generateNormalizedRow(index):
    row = []
    previous_prices = getPreviousPricesFromIndex(index)
    accumulatedRainfall = getAccumulatedRainfallFromIndex(index)

    year = (float(years[index])-13)/(16-13)*2 -1;
    month = float(months[index])/6-1
    day = float(days[index])/31*2-1
    
    maxprice = np.amax(prices)
    minprice = np.amin(prices)
    price = (prices[index] - minprice)/(maxprice-minprice)*2-1
    next_price = (prices[index+1] - minprice)/(maxprice-minprice)*2-1
    previous_prices_scaled =[]
    for pp in previous_prices:
        previous_prices_scaled.append((pp - minprice)/(maxprice-minprice)*2-1)
    
    lokasjoner = temperature[index]
    maxtemperature = np.amax(temperature)
    mintemperature = np.amin(temperature)
    lokasjoner_scaled = []
    for t in lokasjoner:
           lokasjoner_scaled.append((t - mintemperature)/(maxtemperature-mintemperature)*2 -1)
    
    lokasjoner_snow = snow[index]
    maxsnow = np.amax(snow)
    minsnow = np.amin(snow)
    snow_scaled = []
    for t in lokasjoner_snow:
            snow_scaled.append((t-minsnow)/(maxsnow-minsnow)*2 -1)
            
    lokasjoner_rainfall = rainfall[index]
    maxrainfall = np.amax(rainfall)
    minrainfall = np.amin(rainfall)
    rainfall_scaled = []
    for t in lokasjoner_rainfall:
            rainfall_scaled.append((t-minrainfall)/(maxrainfall-minrainfall)*2 -1)
            
    max_accumulated_rainfall = np.amax(accumulatedRainfall)
    min_accumulated_rainfall = np.amin(accumulatedRainfall)
    accumulated_rainfall_scaled = []
    for t in accumulatedRainfall:
            accumulated_rainfall_scaled.append((t-min_accumulated_rainfall)/(max_accumulated_rainfall-min_accumulated_rainfall)*2 -1)
            
    row.append(year)
    row.append(month)
    row.append(day)
    row.extend(lokasjoner_scaled)
    row.extend(rainfall_scaled)
    row.extend(accumulated_rainfall_scaled)
    row.extend(snow_scaled)
    row.extend(previous_prices_scaled)
    row.append(price)
    row.append(next_price)
    return row

def writeDataSet(name, start, end, writelabel):
    with open(name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if writelabel: 
            writer.writerow(["year", "month", "day", 
                            "temperature oslo", "temperature bergen", "temperature trondheim", "temperature tromso",
                             "tronstad rain", "kvildal rain", "tronstad rain last week","tronstad rain last month",
                             "kvildal rain last week", "kvildal rain last month", "tronstad snow", "kvildal snow",
                            "10 days ago", "9 days ago","8 days ago", "7 days ago", "6 days ago", "5 days ago",
                            "4 days ago", "3 days ago", "2 days ago", "1 day ago", "today", "tomorrow"])
        for x in range(NUMBER_OF_PREVIOUS_PRICES, NUMBER_OF_ROWS):
            row = generateRow(x)
            writer.writerow(row)
                
                
def writeDataSetScaled(name,start, end, writelabel):
    with open(name, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        if writelabel: 
            writer.writerow(["year", "month", "day", 
                            "temperature oslo", "temperature bergen", "temperature trondheim", "temperature tromso",
                              "tronstad rain", "kvildal rain", "tronstad rain last week","tronstad rain last month",
                             "kvildal rain last week", "kvildal rain last month", "tronstad snow", "kvildal snow",
                            "10 days ago", "9 days ago","8 days ago", "7 days ago", "6 days ago", "5 days ago",
                            "4 days ago", "3 days ago", "2 days ago", "1 day ago", "today", "tomorrow"])
        for x in range(start,end ):
            row = generateNormalizedRow(x)
            writer.writerow(row)
                
            
readSpotPrices()
readTemperature()
readRainfall()
readSnow()
NUMBER_OF_ROWS= len(years)-1
print NUMBER_OF_ROWS
print len(temperature)

writeDataSetScaled("test_scaled.csv", 2099 ,NUMBER_OF_ROWS,False )
    