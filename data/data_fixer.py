# -*- coding: utf-8 -*-
"""
Created on Fri May 28 19:30:17 2021

@author: mstan
"""

# %%

import json
import os
import ast
import pandas as pd

os.chdir(r'G:\Users\mstan\Downloads\beer\beer_json')
# %%

dic_list = []
index = 0
#data = json.load(f)
newstring = ''
with open('beer.json','r') as f:
    for i, line in enumerate(f):
      #  print(line)
    #  {"review/appearance": 2.5, "beer/style": "Hefeweizen", "review/palate": 1.5, "review/taste": 1.5, "beer/name": "Sausa Weizen", "review/timeUnix": 1234817823, "beer/ABV": 5.0, "beer/beerId": "47986", "beer/brewerId": "10325", "review/timeStruct": {"isdst": 0, "mday": 16, "hour": 20, "min": 57, "sec": 3, "mon": 2, "year": 2009, "yday": 47, "wday": 0}, "review/overall": 1.5, "review/text": "A lot of foam. But a lot.\tIn the smell some banana, and then lactic and tart. Not a good start.\tQuite dark orange in color, with a lively carbonation (now visible, under the foam).\tAgain tending to lactic sourness.\tSame for the taste. With some yeast and banana.", "user/profileName": "stcules", "review/aroma": 2.0}
#'review/appearance', 'beer/style': 'Herbed / Spiced Beer', 'review/palate': 4.0, 'review/taste': 4.0, 'beer/name': 'Caldera Ginger Beer', 'review/timeUnix': 1285632924, 'user/gender': 'Male', 'beer/ABV': 4.7, 'beer/beerId': '52159', 'beer/brewerId': '1075', 'review/timeStruct': {'isdst': 0, 'mday': 28, 'hour': 0, 'min': 15, 'sec': 24, 'mon': 9, 'year': 2010, 'yday': 271, 'wday': 1}, 'review/overall': 4.5, 'review/text': 'Poured from a 22oz bomber into my Drie Fonteinen tumbler. \t\tHazy titanium yellow body (which catches the shadows forming a beautiful mysterious gradient) with an incredibly dense pillow of magnolia cream. Heavy persistent head and rich creamy lacing.\t\tPale malt, asian pear, and a hint of citrus in the nose. A vaguely tropical lager...\t\tTastes very much like a well done APA, with a nice balance of pale malt and low hop bitterness. The ginger adds to the refreshing character, but isn\'t readily detectable at first (lacks any "bite"). Medium-dry finish - very clean and extremely quaffable. I can imagine hibiscus and beets working in small quantities, though I think they omitted those for this version...\t\tLight bodied, pillowy, smooth and moderately carbonated.\t\tDon\'t go into this expecting a ginger beer (despite its name) as it has little in common with that spicy soft drink. This is a wonderful session ale though, and worth seeking out if you are a fan of light yet flavorful lagers. Would obviously go perfectly with sushi.', 'user/profileName': 'augustgarage', 'review/aroma': 3.5}
        dic_list.append(ast.literal_eval(line)) 

        # index += 1
        # if index == 10:
        #     break
print('writing data')     
with open('beer_fixed.json','w') as f_fixed:
    for dic in dic_list:
        json.dump(dic,f_fixed)
        f_fixed.write("\n")

#with open('beer_fixed.json', "r") as read_file:
#    data = json.load(read_file)
#print(data,"\n\n")
# %%
dF = pd.read_json(r'beer_fixed.json',lines=True)

# %%

import pandas as pd

import codecs
dF = pd.read_json(codecs.open(r'beer.json','r','utf-8'))

# %%
