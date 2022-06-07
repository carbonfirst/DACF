# DACF
DACF: Day-ahead Carbon Intensity Forecasting of Power Grids using Machine Learning. <br>
(Refer DACF.pdf for the paperS)

Version: 1.0 <br>
Authors: Diptyaroop Maji, Ramesh K Sitaraman, Prashant Shenoy <br>
Affiliation: University of Massachusetts, Amherst


<!-- ## CarbonCast Architecture
### First tier
### Second Tier
#### CarbonCastCNN
#### CarbonCastLR -->

## 1. Regions covered 
* US: 
    * California ([CISO]())
    * Pennsylvania-Jersey-Maryland Interconnection ([PJM]())
    * Texas ([ERCOT]())
    * New England ([ISO-NE]())
* Europe (European regions are monitored by [ENTSOE]()):
    * Sweden
    * Germany

## 2. Data Sources
US ISO electricity generation by source: [EIA hourly grid monitor](https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48)

European regions electricity generation by source: [ENTSOE]()

Weather forecasts: [GFS weather forecast archive]()

Solar/Wind Forecasts:
* CISO: [OASIS]()
* European regions: [ENTSOE]()
* We currently do not have solar/wind forecasts for other regions, Hence, we generate them using ANN models along with pther source production forecasts.

## 3. Usage
### 3.1 Installing dependencies:
DACF requires Python 3, Keras & Tensorflow 2.x <br>
Other required packages:
* Numpy, Pandas, MatplotLib, SKLearn, Pytz, Datetime
<!-- * ``` pip3 install numpy, matplotlib, sklearn, datetime, matplotlib ``` -->

<!-- ### 3.2 Getting Weather data:
The aggregated & cleaned weather forecasts that we have used for our regions are provided in ```data/```. If you need weather forecasts for other regions, or even for the same regions (eg. if you want to use a different aggregation method), the procedure is as follows:<br>
* GitHub repo of script to fetch weather data can be found [here]().
* Once you have obtained the grib2 files, use the following files to aggregate & clean the data:<br>
```python3 dataCollectionScript.py```<br>
```python3 cleanWeatherData.py```<br> -->

### 3.2 Getting source production forecasts:
For getting source production forecasts, run the following file:<br>
```python3 sourceProductionForecast.py <region> <source>```<br>
<!-- Note that you need to change the config.json file to get a particular source production forecast for a specific region. Example:
``` <example> ```<br>
A detailed description of how to configure is given in Section 3.5 -->

<!-- ### 3.5 Configuring CarbonCast:
Change the config.json file for desired configurations. Below are the fields used in the file along with their meaning:<br>
PREDICTION_WINDOW_HOURS: Prediction window in hours. (Default: 24, for day-ahead forecasting) -->

<!-- ### 3.6 Output (forecasts): -->

## 4. Developer mode

We welcome users to suggest modifications to improve CarbonCast and/or add new features or models to the existing codebase. 
<!-- Use the developer branch to make edits and submit a change. -->

## 5. Citing DACF
If you use DACF, please consider citing our paper. The BibTex format is as follows: <br>
``` [To appear, e-Energy 2022] ```

## 7. Acknowledgements
This work is part of the [CarbonFirst](http://carbonfirst.org/) project, supported by NSF grants 2105494, 2021693, and 2020888, and a grant from VMware.
