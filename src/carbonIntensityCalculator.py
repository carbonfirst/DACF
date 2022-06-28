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
import seaborn as sns
from numpy.lib.utils import source
from pandas.core.frame import DataFrame
from pandas.io.formats import style
from scipy.sparse import data
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import validation
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf

# ISO_LIST = ["CISO", "ERCO", "ISNE", "PJM"]
LOCAL_TIMEZONES = {"BPAT": "US/Pacific", "CISO": "US/Pacific", "ERCO": "US/Central", 
                    "SOCO" :"US/Central", "SWPP": "US/Central", "FPL": "US/Eastern", 
                    "ISNE": "US/Eastern", "NYIS": "US/Eastern", "PJM": "US/Eastern", 
                    "MISO": "US/Eastern"}
# START_ROW = {"CISO": 30712, "ERCO": 30714, "ISNE": 30715, "PJM": 30715}
IN_FILE_NAME = None
OUT_FILE_NAME = None

DAY_INTERVAL = 1
MONTH_INTERVAL = 1

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



def initialize(inFileName, startRow):
    print("FILE: ", inFileName)
    dataset = pd.read_csv(inFileName, header=0, infer_datetime_format=True, 
                            parse_dates=["UTC time"]) #, index_col=["Local time"]
    print(dataset.head(2))
    print(dataset.tail(2))
    dataset.replace(np.nan, 0, inplace=True) # replace NaN with 0.0
    num = dataset._get_numeric_data()
    num[num<0] = 0
    
    print(dataset.columns)
    print("UTC time", dataset["UTC time"].dtype)
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

def fillMissingHours(dataset, datetime, hourlyDateTime):
    modifiedDataset = pd.DataFrame(index=hourlyDateTime, columns= dataset.columns.values)
    idx = 0
    for i in range(len(hourlyDateTime)):
        if(datetime[idx]==hourlyDateTime[i]):
            for j in range(len(dataset.columns.values)):
                modifiedDataset.iloc[i,j] = dataset.iloc[idx,j]
            idx +=1
        else:
            print(idx, i, datetime[idx], hourlyDateTime[i])
            modifiedDataset.iloc[i,0] = hourlyDateTime[i]
            for j in range(1, len(dataset.columns.values)):
                modifiedDataset.iloc[i,j] = dataset.iloc[idx,j]
    return modifiedDataset



def calculateCarbonIntensity(dataset, carbonRate, numSources):
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
        carbonCol.append(round(carbonIntensity, 2))
    dataset.insert(loc=CARBON_INTENSITY_COLUMN, column="carbon_intensity", value=carbonCol)
    return dataset

def calculateCarbonIntensityFromSourceForecasts(dataset, carbonRate, carbonIntensityCol):
    carbonIntensity = 0
    carbonCol = []
    miniDataset = dataset.iloc[:, carbonIntensityCol:]
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
                miniDataset.iloc[i] = dataset.iloc[i, carbonIntensityCol:]
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
        carbonCol.append(round(carbonIntensity, 2))
    dataset.insert(loc=carbonIntensityCol, column="carbon_from_src_forecasts", value=carbonCol)
    return dataset


def readFile(inFileName):
    # load the new file
    dataset = pd.read_csv(inFileName, header=0, infer_datetime_format=True, 
                            parse_dates=["UTC time"], index_col=["UTC time"])    
    numRowsInYear = 8784
    print(dataset.head())
    columns = dataset.columns
    print(columns)
    dateTime = dataset.index.values
    return dataset, dateTime

def showDailyAverageCarbon(dataset, dateTime, localTimezone, iso):
    carbon = np.array(dataset["carbon_intensity"].values)
    carbon = np.resize(carbon, (carbon.shape[0]//24, 24))
    dailyAvgCarbon = np.mean(carbon, axis = 1)
    dates = getDatesInLocalTimeZone(dateTime, localTimezone)    
    
    fig, ax = plt.subplots()
    ax.plot(dates, dailyAvgCarbon)
    # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d, %H:%M"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1, tz=localTimezone))    
    plt.xlabel("Local time")
    plt.ylabel("Daily avg Carbon Intensity (g/kWh)")
    plt.title(iso)
    plt.grid(axis="x")
    plt.xticks(rotation=90)
    plt.legend()
    return

def getDatesInLocalTimeZone(dateTime, localTimezone):
    dates = []
    fromZone = pytz.timezone("UTC")
    for i in range(0, len(dateTime), 24):
        day = pd.to_datetime(dateTime[i]).replace(tzinfo=fromZone)
        day = day.astimezone(localTimezone)
        dates.append(day)    
    return dates

def aggregateHalfHourlyData(origDataset):
    dataset = pd.DataFrame(np.empty((8760*2, len(origDataset.columns.values))) * np.nan,
                    columns=origDataset.columns.values)
    idx = 1
    dataset.iloc[0] = origDataset.iloc[0]
    for i in range(1, len(origDataset)):
        if((origDataset.iloc[i-1,0].minute == 30 and origDataset.iloc[i,0].minute == 0)
            or (origDataset.iloc[i-1,0].minute == 0 and origDataset.iloc[i,0].minute == 30)):
            dataset.iloc[idx] = origDataset.iloc[i]
        else:
            if((origDataset.iloc[i-1,0].minute == 30 and origDataset.iloc[i,0].minute == 30)):
                dataset.iloc[idx] = origDataset.iloc[i-1]
                dataset.iloc[idx, 0] = origDataset.iloc[i-1, 0] + pd.DateOffset(minutes=30)
                idx += 1
                dataset.iloc[idx] = origDataset.iloc[i]
            elif((origDataset.iloc[i-1,0].minute == 0 and origDataset.iloc[i,0].minute == 0)):
                dataset.iloc[idx] = origDataset.iloc[i]
                dataset.iloc[idx, 0] = origDataset.iloc[i-1, 0] + pd.DateOffset(minutes=30)
                idx += 1
                dataset.iloc[idx] = origDataset.iloc[i]
        idx += 1

    print(idx)

    idx = 0
    modifiedDataset = pd.DataFrame(np.empty((len(dataset)//2, len(dataset.columns.values))) * np.nan,
                    columns=dataset.columns.values)
    for i in range(0, len(dataset), 2):
        modifiedDataset.iloc[idx,0] = dataset.iloc[i+1, 0]
        for j in range(1, len(dataset.columns)):
                modifiedDataset.iloc[idx,j] = dataset.iloc[i,j]+dataset.iloc[i+1, j]
        idx += 1
    print(modifiedDataset.head())
    print(len(modifiedDataset))
    return modifiedDataset

idx=0
CARBON_INTENSITY_COLUMN = 2
ISO_LIST = ["ERCO"]

if(len(sys.argv) < 4):
    print("Parameters missing. Usage: python3 carbonDataCleaner.py <iso> <emissionFactorType (-l/-d)> <# electricity sources>")
    exit(0)
ISO_LIST = [sys.argv[1]]
emissionFactorType = "lifecycle" #default
if(sys.argv[2]=="-l"):
    emissionFactorType = "lifecycle" # default
else:
    emissionFactorType = "direct"
numSources = int(sys.argv[3])


for iso in ISO_LIST:
    IN_FILE_NAME = "../data/"+iso+"/"+iso+"_src_prod_forecasts_lifecycle.csv"
    # IN_FILE_NAME = "../data/"+iso+"/"+iso+"_source_mix.csv"
    # IN_FILE_NAME = iso+"/"+iso+"_solar_wind_fcst_final.csv"
    OUT_FILE_NAME = "../data/"+iso+"/"+iso+"_forecast_carbon_lifecycle.csv"
    CARBON_FROM_SRC_FORECASTS_OUT_FILE_NAME = "../data/"+iso+"/"+iso+"_carbon_from_src_forecasts_lifecycle.csv"
    if (emissionFactorType == "direct"):
        IN_FILE_NAME = "../data/"+iso+"/"+iso+"_src_prod_forecasts_direct.csv"
        OUT_FILE_NAME = "../data/"+iso+"/"+iso+"_forecast_carbon_direct.csv"
        CARBON_FROM_SRC_FORECASTS_OUT_FILE_NAME = "../data/"+iso+"/"+iso+"_carbon_from_src_forecasts_direct.csv"
    # IN_FILE_NAME = iso+"/"+iso+"_2019.csv"
    # OUT_FILE_NAME = iso+"/fuel_forecast/"+iso+"_2019_clean2.csv"
    startRow = 0 #START_ROW[iso]
    dataset = initialize(IN_FILE_NAME, startRow)





    # hourlyDateTime = createHourlyTimeCol(dataset, dataset["UTC time"].values, "2020-01-01T00:00")
    # dataset["UTC time"] = hourlyDateTime
    # # modifiedDataset = aggregateHalfHourlyData(dataset)
    # modifiedDataset.to_csv(OUT_FILE_NAME)
    # # print(len(modifiedDataset))
    # # modifiedDataset = fillMissingHours(modifiedDataset, modifiedDataset["UTC time"].values, hourlyDateTime)
    # # print(len(modifiedDataset))
    # exit(0)
    # print(len(modifiedDataset))
    # hourlyDateTime = createHourlyTimeCol(dataset, dataset["Local time"].values, "2019-01-01T00:00")
    # localTime = createHourlyTimeCol(dataset, dataset["Local time"].values, "2019-01-01T01:00")
    # modifiedDataset.insert(0, "datetime", hourlyDateTime)
    # modifiedDataset["Local time"] = localTime
    # modifiedDataset.index = hourlyDateTime
    # print(modifiedDataset.head())
    # print(modifiedDataset.tail())
    # modifiedDataset.to_csv(IN_FILE_NAME)
    # exit(0)

    # if(emissionFactorType == "lifecycle"):
    #     print("Calculating carbon intensity using lifecycle emission factors...")
    #     dataset = calculateCarbonIntensity(dataset, carbonRateLifecycle, numSources)
    # else:
    #     print("Calculating carbon intensity using direct emission factors...")
    #     dataset = calculateCarbonIntensity(dataset, carbonRateDirect, numSources)

    
    print("Calculating carbon intensity from forecasts using direct emission factors...")
    dataset = calculateCarbonIntensityFromSourceForecasts(dataset, forcast_carbonRateDirect, CARBON_INTENSITY_COLUMN)

    dailyAvgMape, avgMape = getMape(dataset["UTC time"].values, dataset["carbon_intensity"].values, 
                    dataset["carbon_from_src_forecasts"].values)
    for item in dailyAvgMape:
        print(item)
    print("Mean MAPE: ", avgMape)
    print("Median MAPE: ", np.percentile(dailyAvgMape, 50))
    print("90th percentile MAPE: ", np.percentile(dailyAvgMape, 90))
    print("95th percentile MAPE: ", np.percentile(dailyAvgMape, 95))
    # print("99th percentile MAPE: ", np.percentile(dailyAvgMape, 99))

    # dataset = addDayAheadForecastColumns(dataset, 0.1)
    # print(dataset.head())
    # dataset.to_csv(OUT_FILE_NAME)
    # dataset.to_csv(CARBON_FROM_SRC_FORECASTS_OUT_FILE_NAME)
    
    # inFileName = "dataset/"+iso+"_clean.csv"
    # dataset, dateTime = readFile(inFileName)
    # print("ISO: ", iso)
    # # analyzeTimeSeries(dataset, None, None, dateTime)
    # showDailyAverageCarbon(dataset, dateTime, pytz.timezone(LOCAL_TIMEZONES[idx]), iso)
    # print(iso, " Analyzed")
    idx+=1
# showPlots()


