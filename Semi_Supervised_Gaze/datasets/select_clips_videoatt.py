import os

import numpy as np

clip_select = {"Hell's Kitchen": ["209_270", "52807_52838"], 
               "Titanic": ['1738_1919'],
               "West World": ['33264_33408'],
               "Jamie Oliver": ['1225_1300', '4200_4225 '],
               "CBS This Morning":['1948_2128', '4165_4256'],
               "It's Always Sunny in Philadelphia": ['25440_25530'],
               "MLB Interview": ['1918_2098'], # this has 178 inside frames
				"I Wanna Marry Harry":["5706_5779", "22729_22778 "],
    			"Downton Abby": ["2307_2458", "2967_3177"],
       			"Survivor": ["2517_2578", "13336_13397 "]}

model_name_map = {
    "Hell's Kitchen": "Kitchen", 
    "Titanic": "Titanic",
    "West World": "West_world",
    "Jamie Oliver": "JamieOliver",
    "CBS This Morning":"CBS",
    "It's Always Sunny in Philadelphia": "Philadelphia",
    "MLB Interview": "MLBInterview", # this has 178 inside frames
    "I Wanna Marry Harry":"Harry",
    "Downton Abby": "Downton",
    "Survivor": "Survivor"
    
}