
# coding: utf-8

# In[ ]:


import pandas
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import pickle
import re
import scipy.stats as pstat
import time
import datetime as dt
import itertools as it
import json
import folium


# RQ1

def RQ1 (months) :
    nyc = [] # These two list will be useful to get the overall average in NY
    idx = []
    months_length = [31,28,31,30,31,30] #A vector with months length
    
    for month in range(1, months + 1): # For every moths, do ..
        
        with open("taxi-2018-0" + str(month), "rb") as f : #Unserializes the byte-stream created above for the current month dataset
            taxi = pickle.load(f)                        
            
        
        
        tmpny = taxi["tpep_pickup_datetime"].count() / months_length[month - 1] # Counts how many trips there were in NY in the month and divides by the length of the month
        tmpb = (taxi.groupby("Borough").count()["tpep_pickup_datetime"] / months_length[month  - 1]).to_frame() #Makes the same thing for each borough
        tmpb = tmpb.rename( columns = {"tpep_pickup_datetime" : "2018-0" + str(month)})             # Just renaming columns to have a better visualization
        
        # The following conditional statements create the final df by merging months together
        if month > 1 :
            
            mean_taxi = mean_taxi.merge(tmpb, on = "Borough")
        
        else :
            
            mean_taxi = tmpb
            
        nyc.append(tmpny) 
        idx.append("2018-0" + str(month)) # Update the lists containg info about the entire city for every month
        
    mean_taxi.loc["NYC"] = pandas.Series(nyc, index = idx) # Update final df with overall infos
    
    return mean_taxi.round(decimals = 1)




# RQ2

def RQ2 (slots,months) :

    pass_taxi = pandas.DataFrame() #Initialize resulting df
      
    for month in range(1,months + 1): # For every month, do...
        tmp = pandas.DataFrame() 
        with open("taxi-2018-0" + str(month), "rb") as f : #Unserializes the byte-stream created above for the current month dataset
            taxi = pickle.load(f) 
        for slot in slots : # for every time slot
            
            # The following conditional statements update the number of runs registered in slots each month
            if month >1 :
                
                # To check a slot, we setted a Regex that works on the pickup time -> ex: "2018-01-[0-9]{2,2} 0[0-5]:[0-9]{2,2}:[0-9]{2,2}"
                # is matched for trips started from 00:00 to 05:59 in January. Obviously we used apply to check the regex for all records.
                
                tmp["from " + slot[0] + slot[2] + ":00 " + "to " + slot[0] + slot[4] +":59"] = taxi.loc[taxi["tpep_pickup_datetime"].apply(lambda x: bool(re.match(r"2018-0"+str(month)+"-[0-9]{2,2} "+ slot + ":[0-9]{2,2}:[0-9]{2,2}",x))) == True ].groupby("Borough")["passenger_count"].agg('sum')
                
                pass_taxi.loc[:,"from " + slot[0] + slot[2] + ":00 " + "to " + slot[0] + slot[4] +":59"] = tmp["from " + slot[0] + slot[2] + ":00 " + "to " + slot[0] + slot[4] +":59"] + pass_taxi.loc[:,"from " + slot[0] + slot[2] + ":00 " + "to " + slot[0] + slot[4] +":59"]
               
            else :
                
                pass_taxi["from " + slot[0] + slot[2] + ":00 " + "to " + slot[0] + slot[4] +":59"]=taxi.loc[taxi["tpep_pickup_datetime"].apply(lambda x: bool(re.match(r"2018-0"+str(month)+"-[0-9]{2,2} "+ slot +  ":[0-9]{2,2}:[0-9]{2,2}",x))) == True ].groupby("Borough")["passenger_count"].agg('sum').fillna(0)
                
                pass_taxi.fillna(0, inplace = True)
                
                
    pass_taxi.loc['NYC'] = pandas.Series(pass_taxi.sum()) # Update final df with overall infos
    

    return pass_taxi




# RQ3

# The following function takes a df as argument and returns the difference, in minutes, between the pickup and dropoff time.
# It uses simply instructions derived from time and datetime libs
def delta_min(x) :
        delta = int(time.mktime(dt.datetime.strptime(x["tpep_dropoff_datetime"], '%Y-%m-%d %H:%M:%S').timetuple())-
                    time.mktime(dt.datetime.strptime(x["tpep_pickup_datetime"], '%Y-%m-%d %H:%M:%S').timetuple()))/60
        if delta > 0 :
            return delta
        else :
            return 0
        
        
def RQ3(deltas,months) :

    time_distro = pandas.DataFrame() #Initialize resulting df
    
    for month in range(1,months+1): # For every month, do...
        
        with open("taxi-2018-0"+str(month),"rb") as f : #Unserializes the byte-stream created above for the current month dataset
            taxi = pickle.load(f)   
        
        taxi["Delta"] = taxi.loc[:,["tpep_pickup_datetime","tpep_dropoff_datetime"]].apply(delta_min,axis = 1) # Adds a column "Delta" with trip durations
        
        tmp = taxi.loc[taxi["trip_distance"] > 0] # Filter weird data choosing runs with a trip distance > 0
        
        for delta in range(1,len(deltas)) : # for every choosen duration
            
            # The following conditional statements update the number of runs registered in durations slots each month
            if month > 1 :
                
                #To classify a trip, we simply use loc to check in which duration slot the trip will be placed
                time_distro["Between " + str(deltas[delta-1]) + " and " + str(deltas[delta]) + " mins"] = taxi.loc[(taxi["Delta"]<= deltas[delta]) & (taxi["Delta"] > deltas[delta - 1]) ].groupby("Borough").count()["Delta"] + time_distro["Between " + str(deltas[delta - 1]) + " and " + str(deltas[delta]) + " mins"]
            
            else :
                
                time_distro["Between " + str(deltas[delta-1]) + " and " + str(deltas[delta]) + " mins"] = taxi.loc[(taxi["Delta"]<= deltas[delta]) & (taxi["Delta"] > deltas[delta - 1]) ].groupby("Borough").count()["Delta"]
        
        time_distro.fillna(0, inplace = True)
        
    time_distro.loc['NYC'] = pandas.Series(time_distro.sum()) # Update final df with overall infos
    
    return time_distro




# RQ4

def Chi_squared(data) :
    
    tmp = data.iloc[:-1,:] # Not interested in NYC data but only boroughs
    
    rows = len(tmp)
    cols = len(tmp.columns)
    
    col_tot = pandas.Series(tmp.sum(axis = 0 ))
    row_tot = pandas.Series(tmp.sum(axis = 1))
    
    chi_sq = 0
    for i in range(cols):
        for j in range(rows):
            
            exp = (row_tot.iloc[j] * col_tot.iloc[i])/(tmp.values.sum()) # expected number
            if exp > 0 :
                chi_sq = ((tmp.iloc[j,i] - exp)**2)/exp + chi_sq #Chi Square updating
            
    return chi_sq


def RQ4 (months) :
    pay_taxi = pandas.DataFrame() #Initialize resulting df
    
    for month in range(1,months+1): # For every month, do...
        
        with open("taxi-2018-0"+str(month),"rb") as f : #Unserializes the byte-stream created above for the current month dataset
            taxi = pickle.load(f)
        taxi = taxi.loc[:,["payment_type", "Borough"]] # Take only columns of interest
        
        # The following conditional statements update the number of payments registered for each payment type each month
        if month > 1:
            
            # We used a list comprehension to build the final df (sequentially checks each payment type)
            pay = pandas.concat([taxi.loc[taxi["payment_type"] == i].groupby("Borough").count().rename(columns = {"payment_type": "type" + str(i)}) for i in range(1,5)], axis = 1, sort = True).fillna(0)  + pay
        
        else :
            
            pay = pandas.concat([taxi.loc[taxi["payment_type"] == i].groupby("Borough").count().rename(columns = {"payment_type": "type"+str(i)}) for i in range(1,5)], axis = 1, sort = True).fillna(0) 
        
    pay.loc['NYC'] = pandas.Series(pay.sum()) # Update final df with overall infos
    pay.rename(columns = {"type1" : "Credit Card", "type2" : "Cash", "type3" : "No Charge", "type4" : "Dispute"}, inplace = True)
    chi = Chi_squared(pay) # Compute the Chi squared test
    chit = pstat.chi2_contingency(pay)
   
    return pay, chi,chit




# RQ5

def RQ5 (months) :
    
    for month in range(1,months+1): # For every month, do ...
        
        with open("taxi-2018-0"+str(month),"rb") as f :  # Unserializes the byte-stream created above for the current month dataset
            taxi = pickle.load(f)
                    
        taxi["Delta"] = taxi.loc[:,["tpep_pickup_datetime","tpep_dropoff_datetime"]].apply(delta_min,axis = 1) # Add trips durations to df
        
        if month > 1 :
            
            corr_df =pandas.concat([corr_df,taxi.loc[taxi["trip_distance"] > 0][["Delta","trip_distance"]]], axis=0) # Update df for every month with trips distance and duration, filtering negative distance
        else :
            corr_df = taxi.loc[taxi["trip_distance"] > 0][["Delta","trip_distance"]]
   
    corr_df.reset_index(drop = True, inplace = True)    # Reset indices
    
    pearson = corr_df["Delta"].corr(corr_df["trip_distance"]) # Get pearson Coefficient
    
    return pearson, corr_df
    
    

# CRQ1a

def dol_per_mile(x) :  # Function that computes $/mile for every trip
     return(x["fare_amount"]/(x["trip_distance"]))
    
def CR1a(months) :
    
    trip_fare = pandas.DataFrame() # Initializing data Frames
    stats = pandas.DataFrame()
    
    for month in range(1,months+1): # For every month, do...
        
        with open("taxi-2018-0"+str(month),"rb") as f : # Unserializes the byte-stream created above for the current month data
            taxi = pickle.load(f)
            
        # Drops columns of not interest
        taxi = taxi.drop(["tpep_pickup_datetime","tpep_dropoff_datetime","passenger_count","PULocationID","DOLocationID","payment_type"], axis = 1)
       
        if month > 1: #Updates df for every month
            trip_fare = pandas.concat([trip_fare,taxi], axis = 0)
        else :
            trip_fare = taxi
    
    trip_fare = trip_fare.loc[(trip_fare["trip_distance"]>0) & (trip_fare["fare_amount"]>0) ] # Filters negative trip distance and fares
    
    trip_fare["$/mile"] = trip_fare.apply(dol_per_mile, axis = 1) # Adds the column of $/mile
    
    trip_fare.reset_index(drop=True, inplace = True)
    
    mean = trip_fare.loc[:,"$/mile"].mean() # Gets the global summaries
    std = trip_fare.loc[:,"$/mile"].std()
    
    stats["Mean"]= trip_fare.groupby("Borough").mean().loc[:,"$/mile"] # Gets boroughs summaries
    stats["Std. Dev"]= trip_fare.groupby("Borough").std().loc[:,"$/mile"]
    
    stats.loc['NYC'] = pandas.Series([mean,std], index = ["Mean","Std. Dev"]) # Finishes building stats
    
    t_test = pandas.DataFrame(columns = ["t-score","p-value"]) # Initializing t-test df
    
    boroughs = list(set(trip_fare["Borough"])) # List of boroughs
    
    comb = list(it.combinations(boroughs, 2)) # All possible not ordered combinations of two boroughs
    
    trip_fare = trip_fare.loc[(trip_fare["$/mile"] > 0) & (trip_fare["$/mile"] <30)] # Filters data in a reasonable way (0 $/mile or >30 $/mile is so strange)
    
    for bor in comb : # For each of them, it makes the t-test and updates the df
                t_test.loc[bor[0] + "-" + bor[1] ] = pandas.Series(pstat.ttest_ind(trip_fare.loc[trip_fare["Borough"]==bor[0]]["$/mile"],trip_fare.loc[trip_fare["Borough"]==bor[1]]["$/mile"]), index = ["t-score","p-value"])
     
    return stats, t_test, trip_fare




# CRQ1b

def wdol_per_mile(x) : # We just weight the $/mile with trip duration
        return((x["fare_amount"])/(x["trip_distance"]*x["Delta"]))
    
# The function doesn't need comments, it very very similiar to the CR1a one

def CR1b (months) :
    
    trip_fare = pandas.DataFrame()
    stats = pandas.DataFrame()
    
    for month in range(1,months+1):
        
        with open("taxi-2018-0"+str(month),"rb") as f :
            taxi = pickle.load(f)
            
        taxi = taxi.drop(["passenger_count","PULocationID","DOLocationID","payment_type"], axis = 1)
        
        taxi["Delta"] = taxi.loc[:,["tpep_pickup_datetime","tpep_dropoff_datetime"]].apply(delta_min,axis = 1)
        
        if month > 1:
            trip_fare = pandas.concat([trip_fare,taxi], axis = 0)
        else :
            trip_fare = taxi
            
    trip_fare = trip_fare.loc[(trip_fare["trip_distance"]>0) & (trip_fare["fare_amount"]>0) & (trip_fare["Delta"]>0) ]
    
    trip_fare["w$/mile"] = trip_fare.apply(wdol_per_mile, axis = 1)
    
    trip_fare.reset_index(drop=True, inplace = True)
    
    mean = (trip_fare.loc[:,"w$/mile"]).mean()
    std = (trip_fare.loc[:,"w$/mile"]).std()
    
    trip_fare_groups = trip_fare.groupby("Borough")
    
    t_test = pandas.DataFrame(columns = ["t-score","p-value"])
    
    boroughs = list(set(trip_fare["Borough"]))
    
    display(boroughs)
    
    comb = list(it.combinations(boroughs, 2))
    
    for bor in comb :
                t_test.loc[bor[0] + "-" + bor[1] ] = pandas.Series(pstat.ttest_ind(trip_fare.loc[trip_fare["Borough"]==bor[0]]["w$/mile"],trip_fare.loc[trip_fare["Borough"]==bor[1]]["w$/mile"]), index = ["t-score","p-value"])
    
    stats["Mean"]= trip_fare_groups.mean().loc[:,"w$/mile"]
    
    stats["Std. Dev"]= trip_fare_groups.std().loc[:,"w$/mile"]
    
    stats.loc['NYC'] = pandas.Series([mean,std], index = ["Mean","Std. Dev"])
    
    return stats,t_test,trip_fare


# CRQ2

def taxi_visualization (months) :
    
    taxi = pandas.DataFrame()
    
    for month in range(1,months+1): # We need to concatenate all month dfs
        
               taxi = taxi.append(pandas.DataFrame(pandas.read_csv(r"D:\Claudio\Uni\M 1° anno Sapienza\AMDS\Homeworks\Hw 2\Yellow Cab Data\yellow_tripdata_2018-0" + str(month) + ".csv",usecols=['PULocationID', 'DOLocationID'], engine = "python")), ignore_index = True)

    geo_data = json.load(open(r"D:\Claudio\Uni\M 1° anno Sapienza\AMDS\Homeworks\Hw 2\Homework_2\taxi_zones.json")) # Geo data
    
    
    # First, empty map of NY
    NY = folium.Map(
    location = [40.742054, -73.769417],   #coordinates of new York
    zoom_start = 13,                        
    tiles = "CartoDB positron"              #style 
    )
    
    # fill it with the zone where data are recorded 
    folium.GeoJson(
    geo_data,
    style_function = lambda feature: {
        'fillColor':'lightgreen',
        'color' : 'orange',
        'weight' : 1,
        'fillOpacity' : 0.3,
        'colorOpacity' : 0.5
    }
    ).add_to(NY)
    
    # The following section is made by simple manipulation on initial df in order to get a df indexed by zone_ID that contains number of picks and drops 
    
    loc_id_pick = taxi.groupby('PULocationID').count()["DOLocationID"]
    loc_id_drop = taxi.groupby('DOLocationID').count()["PULocationID"]


    zone_pick_drop = pandas.DataFrame(index=list(range(1,266)),columns=[])
    zone_pick_drop['ZoneID']=list(range(1,266))

    zone_idx_drop = []
    zone_idx_pick = []

    for idx in range(1,266):
        if idx in loc_id_pick:
            zone_idx_pick.append(loc_id_pick[idx])
        else:
            zone_idx_pick.append(0)    
        
        if idx in loc_id_drop:
            zone_idx_drop.append(loc_id_drop[idx])
        else:
            zone_idx_drop.append(0)     #we need to do this check becouse some zones are missing (maybe 0 taxi taken in that zone)

    zone_pick_drop['taxi_pickups'] = zone_idx_pick
    zone_pick_drop['taxi_dropoff'] = zone_idx_drop
    
    
    
    # Now that we have well organized data, we can create a first choropleth map takes in account the number of picks
    
    NY2 = folium.Map(
    location = [40.7142700, -74.0059700],   #coordinates of new York
    zoom_start = 11,                        
    tiles = 'CartoDB positron'              #style of our map
    )

    NY2.choropleth(
        geo_data = geo_data,  #our geojson datas
        data = zone_pick_drop,    #our dataframe
        columns = ['ZoneID', 'taxi_pickups'],
        key_on = 'feature.properties.LocationID', #the key in geojson file that way want to take as zone
        fill_color = 'YlGnBu',   #the color scale that we want
        fill_opacity = 0.8,
        line_opacity = 0.2,
        legend_name = 'Number of taxi picks in 2018',
        highlight = True    #enable the highlight function, to enable highlight functionality when you hover over each area.
    )

    folium.Marker(
        location=[40.7730135746, -73.8702298524],
        popup='Airport LaGuardia',
        icon=folium.Icon(icon='plane')
    ).add_to(NY2)

    folium.Marker(
        location=[40.6413111, -73.7781391],
        popup='John F. Kennedy International Airport',
        icon=folium.Icon(icon='plane')
    ).add_to(NY2)
    
    folium.Marker(
        location=[40.7828647, -73.9653551],
        popup = 'Central Park',
        icon=folium.Icon(color='red')
    ).add_to(NY2)

    folium.Marker(
        location=[40.758895, -73.985131],
        popup = 'Times Square',
        icon=folium.Icon(color='red')
    ).add_to(NY2)

    folium.Marker(
        location=[40.7061927, -74.0091604],
        popup = 'Wall Street',
        icon=folium.Icon(color='red')
    ).add_to(NY2)

    folium.Marker(
        location=[40.692013, -74.181557],
        popup='Newark Liberty International Airport',
        icon=folium.Icon(icon='plane')
    ).add_to(NY2)

    
    
    
    
    
    
    
    
    # Another choropleth map for taxi drops 
    
    NY3 = folium.Map(
    location=[40.7142700, -74.0059700],   #coordinates of new York
    zoom_start=11,                        
    tiles='CartoDB positron'              #style of our map
    )

    NY3.choropleth(
        geo_data=geo_data,
        data=zone_pick_drop,
        columns=['ZoneID', 'taxi_dropoff'],
        key_on='feature.properties.LocationID',
        fill_color='YlGnBu',
        fill_opacity=0.8,
        line_opacity=0.2,
        legend_name='Number of taxi drops in 2018',
        highlight=True    
    )
    
    folium.Marker(
        location=[40.7730135746, -73.8702298524],
        popup='LaGuardia Airport',
        icon=folium.Icon(icon='plane')
    ).add_to(NY3)

    folium.Marker(
        location=[40.6413111, -73.7781391],
        popup='John F. Kennedy International Airport',
        icon=folium.Icon(icon='plane')
    ).add_to(NY3)

    folium.Marker(
        location=[40.7828647, -73.9653551],
        popup = 'Central Park',
        icon=folium.Icon(color='red')
    ).add_to(NY3)

    folium.Marker(
        location=[40.758895, -73.985131],
        popup = 'Times Square',
        icon=folium.Icon(color='red')
    ).add_to(NY3)

    folium.Marker(
        location=[40.7061927, -74.0091604],
        popup = 'Wall Street',
        icon=folium.Icon(color='red')
    ).add_to(NY3)

    folium.Marker(
        location=[40.692013, -74.181557],
        popup='Newark Liberty International Airport',
        icon=folium.Icon(icon='plane')
    ).add_to(NY3)

    
    return NY, NY2, NY3