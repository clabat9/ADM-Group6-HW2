# ADM-Group6-HW2
# Analyze how and where taxi moved in NYC in the first semester.

The proposed scripts allow you to explore some features about taxis' behaviour in NYC.

## Get data to analyze

### Download sets of taxi movments

The data in usage are available on [Taxi in NYC](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml). 
They're .csv files well organized with so many interesting features. You can check the data dictionary present on the same page of downloads.


### Additional data

To get a better visualization we need to know which borough a certain zone relates to  . So another useful dataset is taxi_zone_lookup.csv, available on the same page.

## Script and Other files descriptions

1. __`Hw2_libf.py`__: 
	This script contains all the useful functions to get the proposed analysis deeply commented to have a clear view of the logic behind.

2. __`zones.htm`__: 
	This html files contains a NY map that higlights the zone where data were recorded.
	
3. __`picks.html`__: 
	This html file contains a Ny Choroplet map that takes in account the number of taxi picks.
	
4. __`drops.html`__: 
	This html file contains a Ny Choroplet map that takes in account the number of taxi drops.

## `IPython Notebook: Homework2.ipynb`
The goal of the `Notebook` is to provide a storytelling friendly format that shows how to use the implemented code and to carry out the analysis.
I tried to organize it in a way the reader can follow our logic and conlcusions.
