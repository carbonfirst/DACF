import csv
import math
import sys
from datetime import datetime as dt
from datetime import timezone as tz

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz as pytz
from numpy.lib.utils import source
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import utility

LOCAL_TIMEZONES = {"BPAT": "US/Pacific", "CISO": "US/Pacific", "ERCO": "US/Central", 
                    "SOCO" :"US/Central", "SWPP": "US/Central", "FPL": "US/Eastern", 
                    "ISNE": "US/Eastern", "NYIS": "US/Eastern", "PJM": "US/Eastern", 
                    "MISO": "US/Eastern"}

CARBON_INTENSITY_COLUMN = 1 # column for real-time carbon intensity

# Operational carbon emission factors
# Carbon rate used by electricityMap. Checkout this link:
# https://github.com/electricitymap/electricitymap-contrib/blob/master/config/co2eq_parameters_direct.json

# Median direct emission factors
carbonRateDirect = {"coal": 760, "biomass": 0, "nat_gas": 370, "geothermal": 0, "hydro": 0,
                "nuclear": 0, "oil": 406, "solar": 0, "unknown": 575, 
                "other": 575, "wind": 0} # g/kWh # check for biomass. it is > 0
forcast_carbonRateDirect = {"avg_coal_production_forecast": 760, "avg_biomass_production_forecast": 0, 
                "avg_nat_gas_production_forecast": 370, "avg_geothermal_production_forecast": 0, 
                "avg_hydro_production_forecast": 0, "avg_nuclear_production_forecast": 0, 
                "avg_oil_production_forecast": 406, "avg_solar_production_forecast": 0, 
                "avg_unknown_production_forecast": 575, "avg_other_production_forecast": 575, 
                "avg_wind_production_forecast": 0} # g/kWh # correct biomass


def initialize(inFileName):
    print("FILE: ", inFileName)
    dataset = pd.read_csv(inFileName, header=0, infer_datetime_format=True, 
                            parse_dates=["UTC time"]) #, index_col=["Local time"]
    print(dataset.head(2))
    print(dataset.tail(2))
    dataset.replace(np.nan, 0, inplace=True) # replace NaN with 0.0
    num = dataset._get_numeric_data()
    num[num<0] = 0
    
    print(dataset.columns)
    # print("UTC time", dataset["UTC time"].dtype)
    return dataset

def createHourlyTimeCol(dataset, datetime, startDate):
    modifiedDataset = pd.DataFrame(np.empty((17544, len(dataset.columns.values))) * np.nan,
                    columns=dataset.columns.values)
    startDateTime = np.datetime64(startDate)
    hourlyDateTime = []
    hourlyDateTime.append(startDateTime)
    idx = 0
    modifiedDataset.iloc[0] = dataset.iloc[0]
    for i in range(17544-1):
        hourlyDateTime.append(hourlyDateTime[i] +np.timedelta64(1, 'h'))
        # # print(datetime[i+1], datetime[i], (datetime[i+1]-datetime[i]).total_seconds())
        # # if (hourlyDateTime[-1] != datetime[i+1]):
        # if ((pd.Timestamp(datetime[i+1]).hour-pd.Timestamp(datetime[i]).hour) != 1):
        #     if (pd.Timestamp(datetime[i]).hour == 23 and pd.Timestamp(datetime[i+1]).hour == 0):
        #         pass
        #     else:
        #     # print(i, hourlyDateTime[-1], datetime[i+1])
        #         print(i, datetime[i], datetime[i+1], pd.Timestamp(datetime[i]).hour, pd.Timestamp(datetime[i+1]).hour)
        #     # print(dataset.iloc[i-1])
        #     # print(dataset.iloc[i])
    # exit(0)
    return hourlyDateTime


def calculateCarbonIntensity(dataset, carbonRate, numSources):
    global CARBON_INTENSITY_COLUMN
    carbonIntensity = 0
    carbonCol = []
    miniDataset = dataset.iloc[:, CARBON_INTENSITY_COLUMN:CARBON_INTENSITY_COLUMN+numSources]
    print("**", miniDataset.columns.values)
    rowSum = miniDataset.sum(axis=1).to_list()
    for i in range(len(miniDataset)):
        if(rowSum[i] == 0):
            # basic algorithm to fill missing values if all sources are missing
            # just using the previous hour's value
            # same as electricityMap
            for j in range(1, len(dataset.columns.values)):
                if(dataset.iloc[i, j] == 0):
                    dataset.iloc[i, j] = dataset.iloc[i-1, j]
                miniDataset.iloc[i] = dataset.iloc[i, CARBON_INTENSITY_COLUMN:CARBON_INTENSITY_COLUMN+numSources]
                # print(miniDataset.iloc[i])
            rowSum[i] = rowSum[i-1]
        carbonIntensity = 0
        for j in range(len(miniDataset.columns.values)):
            source = miniDataset.columns.values[j]
            sourceContribFrac = miniDataset.iloc[i, j]/rowSum[i]
            # print(sourceContribFrac, carbonRate[source])
            carbonIntensity += (sourceContribFrac * carbonRate[source])
        if (carbonIntensity == 0):
            print(miniDataset.iloc[i])
        carbonCol.append(round(carbonIntensity, 2)) # rounding to 2 values after decimal place
    dataset.insert(loc=CARBON_INTENSITY_COLUMN, column="carbon_intensity", value=carbonCol)
    return dataset

def calculateCarbonIntensityFromSourceForecasts(dataset, carbonRate, numSources):
    global CARBON_INTENSITY_COLUMN
    carbonCol = []
    miniDataset = dataset.iloc[:, CARBON_INTENSITY_COLUMN+1:CARBON_INTENSITY_COLUMN+1+numSources]
    print("**", miniDataset.columns.values)
    rowSum = miniDataset.sum(axis=1).to_list()
    for i in range(len(miniDataset)):
        if(rowSum[i] == 0):
            # basic algorithm to fill missing values if all sources are missing
            # just using the previous hour's value
            # same as electricityMap
            for j in range(1, len(dataset.columns.values)):
                if(dataset.iloc[i, j] == 0):
                    dataset.iloc[i, j] = dataset.iloc[i-1, j]
                miniDataset.iloc[i] = dataset.iloc[i, CARBON_INTENSITY_COLUMN+1:CARBON_INTENSITY_COLUMN+1+numSources]
                # print(miniDataset.iloc[i])
            rowSum[i] = rowSum[i-1]
        carbonIntensity = 0
        for j in range(len(miniDataset.columns.values)):
            source = miniDataset.columns.values[j]
            sourceContribFrac = miniDataset.iloc[i, j]/rowSum[i]
            # print(sourceContribFrac, carbonRate[source])
            carbonIntensity += (sourceContribFrac * carbonRate[source])
        if (carbonIntensity == 0):
            print(miniDataset.iloc[i])
        carbonCol.append(round(carbonIntensity, 2)) # rounding to 2 values after decimal place
    dataset.insert(loc=CARBON_INTENSITY_COLUMN+1, column="carbon_from_src_forecasts", value=carbonCol)
    return dataset


def getDatesInLocalTimeZone(dateTime, localTimezone):
    dates = []
    fromZone = pytz.timezone("UTC")
    for i in range(0, len(dateTime), 24):
        day = pd.to_datetime(dateTime[i]).replace(tzinfo=fromZone)
        day = day.astimezone(localTimezone)
        dates.append(day)    
    return dates

def runProgram(iso, isForecast, numSources):
    IN_FILE_NAME = None
    OUT_FILE_NAME = None

    if (isForecast is True):
        IN_FILE_NAME = "../data/"+iso+"/"+iso+"_src_prod_forecasts_test_period.csv"
        OUT_FILE_NAME = "../data/"+iso+"/"+iso+"_carbon_from_src_prod_forecasts_direct.csv"
    else:
        IN_FILE_NAME = "../data/"+iso+"/"+iso+".csv"
        OUT_FILE_NAME = "../data/"+iso+"/"+iso+"_direct_emissions.csv"        
    
    dataset = initialize(IN_FILE_NAME)

    if (isForecast is True):
        print("Calculating carbon intensity from src prod forecasts using direct emission factors...")
        dataset = calculateCarbonIntensityFromSourceForecasts(dataset, forcast_carbonRateDirect, 
                    numSources)

        dailyAvgMape, avgMape = utility.getMape(dataset["UTC time"].values, dataset["carbon_intensity"].values, 
                        dataset["carbon_from_src_forecasts"].values)
        print("Mean MAPE: ", avgMape)
        print("Median MAPE: ", np.percentile(dailyAvgMape, 50))
        print("90th percentile MAPE: ", np.percentile(dailyAvgMape, 90))
        print("95th percentile MAPE: ", np.percentile(dailyAvgMape, 95))
    else:
        print("Calculating real time carbon intensity using direct emission factors...")
        dataset = calculateCarbonIntensity(dataset, carbonRateDirect, numSources)

    dataset.to_csv(OUT_FILE_NAME)
    
    return


if __name__ == "__main__":
    if (len(sys.argv) !=4):
        print("Usage: python3 carbonIntensityCalculator.py <region> <f/r> <num_sources>")
        print("Refer github repo for regions.")
        print("f - forecast, r - real time")
        print("num_sources - no. of sources producing electricity in the region")
        # print("carbon_intensity_col - column no. where carbon_intensity should be inserted")
        exit(0)
    print("DACF: Calculating carbon intensity for region: ", sys.argv[1])
    region = sys.argv[1]
    isForecast = False
    if (sys.argv[2].lower() == "f"):
        isForecast = True
    numSources = int(sys.argv[3])
    runProgram(region, isForecast, numSources)
    print("Calculating carbon intensity for region: ", sys.argv[1], " done.")









  

    
    

