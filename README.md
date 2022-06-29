# DACF
DACF: Day-ahead Carbon Intensity Forecasting of Power Grids using Machine Learning. <br>
(Refer <b><i>```DACF.pdf```</i></b> for the paper)

<b>Version:</b> 1.0 <br>
<b>Authors:</b> Diptyaroop Maji, Ramesh K Sitaraman, Prashant Shenoy <br>
<b>Affiliation:</b> University of Massachusetts, Amherst


## 0. Citing DACF
If you use DACF, please consider citing our paper. The BibTex format is as follows: <br>

&nbsp; &nbsp; &nbsp; &nbsp;@inproceedings{maji2022dacf,<br>
&nbsp; &nbsp; &nbsp; &nbsp;  title={DACF: Day-ahead Carbon Intensity Forecasting of Power Grids using Machine Learning},<br>
&nbsp; &nbsp; &nbsp; &nbsp;  author={Maji, Diptyaroop and Sitaraman, Ramesh K and Shenoy, Prashant},<br>
&nbsp; &nbsp; &nbsp; &nbsp;  booktitle={Proceedings of the Thirteenth ACM International Conference on Future Energy Systems},<br>
&nbsp; &nbsp; &nbsp; &nbsp;  year={2022}<br>
&nbsp; &nbsp; &nbsp; &nbsp;}<br>


## 1. Regions covered 
* US: 
    * California ([CISO](https://www.caiso.com/Pages/default.aspx))
    * Pennsylvania-Jersey-Maryland Interconnection ([PJM](https://www.pjm.com/))
    * Texas ([ERCOT](https://www.ercot.com/))
    * New England ([ISO-NE](https://www.iso-ne.com/))
* Europe (European regions are monitored by [ENTSOE](https://transparency.entsoe.eu/)):
    * Sweden
    * Germany

## 2. Data Sources
US ISO electricity generation by source: [EIA hourly grid monitor](https://www.eia.gov/electricity/gridmonitor/dashboard/electric_overview/US48/US48)

European regions electricity generation by source: [ENTSOE](https://transparency.entsoe.eu/)

Weather forecasts: [GFS weather forecast archive](https://rda.ucar.edu/datasets/ds084.1/)

Solar/Wind Forecasts:
* CISO: [OASIS](http://oasis.caiso.com/mrioasis/logon.do)
* European regions: [ENTSOE](https://transparency.entsoe.eu/)
* We currently do not have solar/wind forecasts for other regions, Hence, we generate them using ANN models along with pther source production forecasts.

## 3. Usage
### 3.1 Installing dependencies:
DACF requires Python 3, Keras & Tensorflow 2.x <br>
Other required packages:
* ```Numpy, Pandas, MatplotLib, SKLearn, Pytz, Datetime```
<!-- * ``` pip3 install numpy, matplotlib, sklearn, datetime, matplotlib ``` -->

<!-- ### 3.2 Getting Weather data:
The aggregated & cleaned weather forecasts that we have used for our regions are provided in ```data/```. If you need weather forecasts for other regions, or even for the same regions (eg. if you want to use a different aggregation method), the procedure is as follows:<br>
* GitHub repo of script to fetch weather data can be found [here]().
* Once you have obtained the grib2 files, use the following files to aggregate & clean the data:<br>
```python3 dataCollectionScript.py```<br>
```python3 cleanWeatherData.py```<br> -->

### 3.2 Getting source production forecasts:
For getting source production forecasts, run the following file:<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ```python3 sourceProductionForecast.py <region> <source>```<br>
<b>Example:</b> ```python3 sourceProductionForecast.py CISO nat_gas```<br>
<b>Regions:</b> <i>CISO, PJM, ERCO, ISNE, SE, DE</i> <br>
<b>Sources:</b> <i>coal, nat_gas, oil, solar, wind, hydro, unknown, geothermal, biomass, nuclear</i>
<!-- Note that you need to change the config.json file to get a particular source production forecast for a specific region. Example:
``` <example> ```<br>
A detailed description of how to configure is given in Section 3.5 -->

### 3.3 Calculating average carbon intensity:
We use the following formula for calculating both real-time and forecasted avg carbon intensity:<br>
<img src="images/ci_avg.png">    , where <br>
<br>
<i>CI<sub>avg</sub></i> = Average carbon intensity (real-time or forecast) of a region <br>
<i>E<sub>i</sub></i> = Electricity produced by source i, when we are calculating real-time avg. carbon intensity, & day-ahead predicted
electricity produced by source i, when we are calculating day-ahead carbon intensity forecasts. <br>
<i>CR<sub>i</sub></i> = Median operational (direct) carbon emission rate (also known as carbon emission factor) of source i. <br><br>

To calculate carbon intensity, run the following file:<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ```python3 carbonIntensityCalculateor.py <region> <f/r> <num_sources>```<br>
<b>Example:</b> ```python3 carbonIntensityCalculateor.py CISO r 8```<br>
<b>Regions:</b> <i>CISO, PJM, ERCO, ISNE, SE, DE</i> <br>
<b><i>f</i> :</b> forecast (based on source production forecasts), <b><i>r</i> :</b> real-time (based on historical electricity production data)<br>
<b>No. of sources producting electricity:</b> <i>CISO: 8, PJM: 8, ERCO: 7, ISNE: 8, SE: 4, DE: 10</i> <br>

<!-- ### 3.6 Output (forecasts): -->

## 4. Developer mode
DACF is a working prototype. However, we understand that it still needs a lot of improvements. We will be updating the codebase periodically
to add new things (features, regions, improved models etc.). In addition to that, we welcome users to suggest modifications 
to improve DACF and/or add new features or models to the existing codebase. 
<!-- Use the developer branch to make edits and submit a change. -->

## 5. Acknowledgements
This work is part of the [CarbonFirst](http://carbonfirst.org/) project, supported by NSF grants 2105494, 2021693, and 2020888, and a grant from VMware.
