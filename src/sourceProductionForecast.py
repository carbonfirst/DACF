import csv
import math
import sys
from datetime import datetime as dt
from datetime import timezone as tz

import numpy as np
import pandas as pd
import pytz as pytz
from keras.layers import Dense, Flatten
from keras.models import Sequential
from scipy.sparse import data
from sklearn.utils import validation
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import utility

############################# MACRO START #######################################
LOCAL_TIMEZONES = {"CISO": "US/Pacific", "ERCO": "US/Central", "ISNE": "US/Eastern", 
                    "PJM": "US/Eastern", "SE": "CET", "DE": "CET"}
IN_FILE_NAME = None
OUT_FILE_NAME = None
LOCAL_TIMEZONE = None

COAL = 3
NAT_GAS = 4
NUCLEAR = 5
OIL = 6
HYDRO = 7
SOLAR = 8
WIND = 9
UNKNOWN = 10
GEOTHERMAL = 11
BIOMASS = 12
SOURCE_COL = None

NUM_VAL_DAYS = 30
NUM_TEST_DAYS = 184
TRAINING_WINDOW_HOURS = 24
PREDICTION_WINDOW_HOURS = 24
MODEL_SLIDING_WINDOW_LEN = 24
DAY_INTERVAL = 1
MONTH_INTERVAL = 1
NUMBER_OF_EXPERIMENTS = 1

NUM_FEATURES_DICT = {"coal":6, "nat_gas":6, "nuclear":6, "oil":6, "hydro":11, "solar": 11,
                    "wind":11, "unknown": 6, "biomass": 6, "geothermal":6}

NUM_FEATURES = 6
############################# MACRO END #########################################

def initDataset(inFileName, sourceCol):
    dataset = pd.read_csv(inFileName, header=0, infer_datetime_format=True, 
                            parse_dates=['UTC time'], index_col=['UTC time'])

    print(dataset.head())
    print(dataset.columns)
    dateTime = dataset.index.values
    
    print("\nAdding features related to date & time...")
    modifiedDataset = utility.addDateTimeFeatures(dataset, dateTime, sourceCol)
    dataset = modifiedDataset
    print("Features related to date & time added")
    
    for i in range(sourceCol, len(dataset.columns.values)):
        col = dataset.columns.values[i]
        dataset[col] = dataset[col].astype(np.float64)
        # print(col, dataset[col].dtype)

    return dataset, dateTime

# convert training data into inputs and outputs (labels)
def manipulateTrainingDataShape(data, trainWindowHours, labelWindowHours): 
    print("Data shape: ", data.shape)
    X, y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)-(trainWindowHours+labelWindowHours)+1):
        # define the end of the input sequence
        trainWindow = i + trainWindowHours
        labelWindow = trainWindow + labelWindowHours
        xInput = data[i:trainWindow, :]
        # xInput = xInput.reshape((len(xInput), 1))
        X.append(xInput)
        y.append(data[trainWindow:labelWindow, 0])
        # print(data[trainWindow:labelWindow, 0])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)

def manipulateTestDataShape(data, slidingWindowLen, predictionWindowHours, isDates=False): 
    X = list()
    # step over the entire history one time step at a time
    for i in range(0, len(data)-(predictionWindowHours)+1, slidingWindowLen):
        # define the end of the input sequence
        predictionWindow = i + predictionWindowHours
        X.append(data[i:predictionWindow])
    if (isDates is False):
        X = np.array(X, dtype=np.float64)
    else:
        X = np.array(X)
    return X


def trainANN(trainX, trainY, valX, valY, hyperParams):
    n_timesteps, n_features, nOutputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]
    epochs = 1 #hyperParams['epoch']
    batchSize = hyperParams['batchsize']
    activationFunc = hyperParams['actv']
    lossFunc = hyperParams['loss']
    optimizer = hyperParams['optim']
    hiddenDims = hyperParams['hidden']
    learningRates = hyperParams['lr']
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(hiddenDims[0], input_shape=(n_timesteps, n_features), activation=activationFunc)) # 20 for coal, nat_gas, nuclear
    model.add(Dense(hiddenDims[1], activation='relu')) # 50 for coal, nat_gas, nuclear
    model.add(Dense(nOutputs))

    opt = tf.keras.optimizers.Adam(learning_rate = learningRates)
    model.compile(loss=lossFunc, optimizer=optimizer[0],
                    metrics=['mean_absolute_error'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint('best_model_ann.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # fit network
    hist = model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize[0], verbose=2,
                        validation_data=(valX, valY), callbacks=[es, mc])
    model = load_model("best_model_ann.h5")
    utility.showModelSummary(hist, model)
    return model, n_features

def getDayAheadForecasts(trainX, trainY, model, history, testData, 
                            trainWindowHours, numFeatures, depVarColumn):
    global MODEL_SLIDING_WINDOW_LEN
    global PREDICTION_WINDOW_HOURS
    # walk-forward validation over each day
    print("Testing...")
    predictions = list()
    for i in range(0, len(testData)//24):
        dayAheadPredictions = list()
        tempHistory = history.copy()
        currentDayHours = i* MODEL_SLIDING_WINDOW_LEN
        for j in range(0, PREDICTION_WINDOW_HOURS, 24):
            yhat_sequence, newTrainingData = getForecasts(model, tempHistory, trainWindowHours, numFeatures)
            dayAheadPredictions.extend(yhat_sequence)
            # add current prediction to history for predicting the next day
            # following 3 lines are redundant currently. Will become useful if
            # prediction period goes beyond 24 hours.
            latestHistory = testData[currentDayHours+j:currentDayHours+j+24, :].tolist()
            for k in range(24):
                latestHistory[k][depVarColumn] = yhat_sequence[k]
            tempHistory.extend(latestHistory)

        # get real observation and add to history for predicting the next day
        history.extend(testData[currentDayHours:currentDayHours+MODEL_SLIDING_WINDOW_LEN, :].tolist())
        predictions.append(dayAheadPredictions)

    # evaluate predictions days for each day
    predictedData = np.array(predictions, dtype=np.float64)
    return predictedData


def getForecasts(model, history, trainWindowHours, numFeatures):
    # flatten data
    data = np.array(history, dtype=np.float64)
    # retrieve last observations for input data
    input_x = data[-trainWindowHours:]
    # reshape into [1, n_input, num_features]
    input_x = input_x.reshape((1, len(input_x), numFeatures))
    # print("ip_x shape: ", input_x.shape)
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat, input_x

def getANNHyperParams():
    hyperParams = {}
    hyperParams['epoch'] = 100 
    hyperParams['batchsize'] = [10] 
    hyperParams['actv'] = "relu"
    hyperParams['loss'] = "mse"
    hyperParams['optim'] = ["adam"] #, "rmsprop"]
    hyperParams['lr'] = 1e-2 #, 1e-3
    hyperParams['hidden'] = [20, 50] #, [50, 50]]#, [20, 50]] #, [50, 50]]
    return hyperParams

def runProgram(ISO, source):
    global NUCLEAR, COAL, SOLAR, WIND, NAT_GAS, GEOTHERMAL, HYDRO, UNKNOWN, BIOMASS, OIL

    periodRMSE = []
    bestRMSE = []
    predictedData = None
    
    LOCAL_TIMEZONE = pytz.timezone(LOCAL_TIMEZONES[ISO])
    if ISO == "SE":
        NUCLEAR = 3
        UNKNOWN = 4
        HYDRO = 6
        FUEL = {3:"nuclear", 4:"unknown", 5:"wind", 6:"hydro"} # SE
    elif ISO == "DE":
        BIOMASS = 2
        GEOTHERMAL = 5
        HYDRO = 6
        NUCLEAR = 7
        OIL = 8
        SOLAR = 9
        WIND = 10
        UNKNOWN = 11
        FUEL = {2:"biomass", 3:"coal", 4:"nat_gas", 5:"geothermal", 6:"hydro", 7:"nuclear",
                    8:"oil", 9:"solar", 10:"wind", 11:"unknown"} # DE
    else:
        FUEL = {3:"coal", 4:"nat_gas", 5:"nuclear", 6:"oil", 7:"hydro", 8:"solar",
                    9:"wind", 10:"unknown"}


    SOURCE_TO_SOURCE_COL_MAP = {"coal": COAL, "nat_gas" : NAT_GAS, "nuclear" : NUCLEAR,
        "oil" : OIL, "hydro" : HYDRO, "solar" : SOLAR, "wind" : WIND, 
        "geothermal" : GEOTHERMAL, "biomass" : BIOMASS}
    SOURCE_COL = SOURCE_TO_SOURCE_COL_MAP[source]
    NUM_FEATURES = NUM_FEATURES_DICT[FUEL[SOURCE_COL]]
    print("Source: ", source, ", source col: ", SOURCE_COL, ", no. features: ", NUM_FEATURES)
    IN_FILE_NAME = "../data/"+ISO+"/fuel_forecast/"+ISO+"_"+FUEL[SOURCE_COL]+"_2019_clean.csv"
    OUT_FILE_NAME_PREFIX = "../data/"+ISO+"/fuel_forecast/"+ISO+"_src_prod_forecast"
    
    for period in range(4):

        ########################################################################
        #### Train - Jan - Dec 2019, Test - Jan - Jun 2020 ####
        if (period == 0):
            DATASET_LIMITER = 13128
            OUT_FILE_SUFFIX = "_h1_2020"
            NUM_TEST_DAYS = 182
        #### Train - Jan 2019 - Jun 2020, Test - Jul - Dec 2020 ####
        if (period == 1):
            DATASET_LIMITER = 17544
            OUT_FILE_SUFFIX = "_h2_2020"
            NUM_TEST_DAYS = 184
        #### Train - Jan 2020 - Dec 2020, Test - Jan - Jun 2021 ####
        if (period == 2):
            DATASET_LIMITER = 21888
            OUT_FILE_SUFFIX = "_h1_2021"
            NUM_TEST_DAYS = 181
        #### Train - Jan 2020 - Jun 2021, Test - Jul - Dec 2021 ####
        if (period == 3):
            DATASET_LIMITER = 26304
            OUT_FILE_SUFFIX = "_h2_2021"
            NUM_TEST_DAYS = 184
        ########################################################################
        

        print("Initializing...")
        dataset, dateTime = initDataset(IN_FILE_NAME, SOURCE_COL)
        print("***** Initialization done *****")

        # split into train and test
        print("Spliting dataset into train/test...")
        trainData, valData, testData, fullTrainData = utility.splitDataset(dataset.values, NUM_TEST_DAYS, 
                                                NUM_VAL_DAYS)
        trainDates = dateTime[: -(NUM_TEST_DAYS*24)]
        fullTrainDates = np.copy(trainDates)
        trainDates, validationDates = trainDates[: -(NUM_VAL_DAYS*24)], trainDates[-(NUM_VAL_DAYS*24):]
        testDates = dateTime[-(NUM_TEST_DAYS*24):]
        trainData = trainData[:, SOURCE_COL: SOURCE_COL+NUM_FEATURES]
        valData = valData[:, SOURCE_COL: SOURCE_COL+NUM_FEATURES]
        testData = testData[:, SOURCE_COL: SOURCE_COL+NUM_FEATURES]

        print("TrainData shape: ", trainData.shape) # (days x hour) x features
        print("ValData shape: ", valData.shape) # (days x hour) x features
        print("TestData shape: ", testData.shape) # (days x hour) x features
        print("***** Dataset split done *****")

        for i in range(trainData.shape[0]):
            for j in range(trainData.shape[1]):
                if(np.isnan(trainData[i, j])):
                    trainData[i, j] = trainData[i-1, j]

        for i in range(valData.shape[0]):
            for j in range(valData.shape[1]):
                if(np.isnan(valData[i, j])):
                    valData[i, j] = valData[i-1, j]

        for i in range(testData.shape[0]):
            for j in range(testData.shape[1]):
                if(np.isnan(testData[i, j])):
                    testData[i, j] = testData[i-1, j]

        featureList = dataset.columns.values[SOURCE_COL:SOURCE_COL+NUM_FEATURES]
        print("Features: ", featureList)

        print("Scaling data...")
        # unscaledTestData = np.zeros(testData.shape[0])
        # for i in range(testData.shape[0]):
        #     unscaledTestData[i] = testData[i, 0]
        trainData, valData, testData, ftMin, ftMax = utility.scaleDataset(trainData, valData, testData)
        print("***** Data scaling done *****")
        print(trainData.shape, valData.shape, testData.shape)


        print("\nManipulating training data...")
        X, y = manipulateTrainingDataShape(trainData, TRAINING_WINDOW_HOURS, TRAINING_WINDOW_HOURS)
        # Next line actually labels validation data
        valX, valY = manipulateTrainingDataShape(valData, TRAINING_WINDOW_HOURS, TRAINING_WINDOW_HOURS)
        print("***** Training data manipulation done *****")
        print("X.shape, y.shape: ", X.shape, y.shape)

        ######################## START #####################
        
        hyperParams = getANNHyperParams()

        for xx in range(NUMBER_OF_EXPERIMENTS):
            OUT_FILE_NAME = OUT_FILE_NAME_PREFIX + "_" + featureList[0] + OUT_FILE_SUFFIX + "_expt_"+str(xx)+".csv"
            print("\nStarting training (iteration ", str(xx), ")...")
            bestModel, numFeatures = trainANN(X, y, valX, valY, hyperParams)
            print("***** Training done *****")
            history = valData[-TRAINING_WINDOW_HOURS:, :].tolist()
            predictedData = getDayAheadForecasts(X, y, bestModel, history, testData, 
                            TRAINING_WINDOW_HOURS, numFeatures, 0)            
            actualData = manipulateTestDataShape(testData[:, 0], 
                    MODEL_SLIDING_WINDOW_LEN, PREDICTION_WINDOW_HOURS, False)
            formattedTestDates = manipulateTestDataShape(testDates, 
                    MODEL_SLIDING_WINDOW_LEN, PREDICTION_WINDOW_HOURS, True)
            formattedTestDates = np.reshape(formattedTestDates, 
                    formattedTestDates.shape[0]*formattedTestDates.shape[1])
            actualData = actualData.astype(np.float64)
            print("ActualData shape: ", actualData.shape)
            actual = np.reshape(actualData, actualData.shape[0]*actualData.shape[1])
            print("actual.shape: ", actual.shape)
            unscaledTestData = utility.inverseDataScaling(actual, ftMax[0], 
                                ftMin[0])
            predictedData = predictedData.astype(np.float64)
            print("PredictedData shape: ", predictedData.shape)
            predicted = np.reshape(predictedData, predictedData.shape[0]*predictedData.shape[1])
            print("predicted.shape: ", predicted.shape)
            unScaledPredictedData = utility.inverseDataScaling(predicted, 
                        ftMax[0], ftMin[0])
            rmseScore, mapeScore = utility.getScores(actualData, predictedData, 
                                        unscaledTestData, unScaledPredictedData)
            print("***** Forecast done *****")
            print("Overall RMSE score: ", rmseScore)
            bestRMSE.append(rmseScore)

            data = []
            for i in range(len(unScaledPredictedData)):
                row = []
                row.append(str(formattedTestDates[i]))
                row.append(str(unscaledTestData[i]))
                row.append(str(unScaledPredictedData[i]))
                data.append(row)
            utility.writeOutFuelForecastFile(OUT_FILE_NAME, data, featureList[0])

            
        print("Average RMSE after ", NUMBER_OF_EXPERIMENTS, " expts: ", np.mean(bestRMSE))
        print(bestRMSE)

    periodRMSE.append(bestRMSE)
    ######################## END #####################

    # actual = np.reshape(actualData, actualData.shape[0]*actualData.shape[1])
    # predicted = np.reshape(predictedData, predictedData.shape[0]*predictedData.shape[1])
    # unScaledPredictedData = inverseDataNormalization(predicted, ftMax[0], 
    #                         ftMin[0])

    print("RMSE: ", periodRMSE)
    return



if __name__ == "__main__":
    if (len(sys.argv) !=3):
        print("Usage: python3 sourceProductionForecast.py <region> <source>")
        print("Refer github repo for regions & sources")
        exit(0)
    print("DACF: ANN model for region: ", sys.argv[1], " and source: ", sys.argv[2])
    region = sys.argv[1]
    source = sys.argv[2]
    runProgram(region, source)
    print("Source production forecast for region: ", sys.argv[1], " and source: ", sys.argv[2], " done.")