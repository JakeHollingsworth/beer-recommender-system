from utils import read_config, write_dictionary
import requests
import time
import gzip
import simplejson
import pandas as pd
import os
import numpy as np


def request_data(url,data_path,force_download=False):
    if not(os.path.exists(data_path+ 'raw_data.txt.gz')) or force_download:
        print('Downloading Dataset from %s to %s'%(url,data_path+'raw_data.txt.gz'))
        r = requests.get(url)
        open(data_path + 'raw_data.txt.gz', 'wb').write(r.content)
    else:
        print('Dataset Already Exists. Skipping Download')

def format_data(data_path,max_rows=-1):
    #loading code from the website with minor modifications
    def parse(filename):
        f = gzip.open(filename, 'rb')
        entry = {}
        for l in f:
            l = l.strip()
            colonPos = l.find(b":")
            if colonPos == -1:
                yield entry
                entry = {}
                continue
            eName = l[:colonPos]
            rest = l[colonPos+2:]
            entry[eName] = rest
        yield entry


    data_list = []
    counter = 0
    for e in parse(data_path + "raw_data.txt.gz"):
        try:
            e[b'review/appearance'] = float(e[b'review/appearance'])

            e[b'review/taste'] = float(e[b'review/taste'])
            e[b'review/overall'] = float(e[b'review/overall'])
            e[b'review/palate'] = float(e[b'review/palate'])
            e[b'review/aroma'] = float(e[b'review/aroma'])
            e[b'review/timeUnix'] = int(e[b'review/time'])
            e.pop(b'review/time', None)
            try:
                e[b'beer/ABV'] = float(e[b'beer/ABV'])
            except Exception as q:
                e.pop(b'beer/ABV', None)
            e[b'user/profileName'] = e[b'review/profileName']
            e.pop(b'review/profileName', None)

            e[b'beer/beerId'] = int(e[b'beer/beerId'])
            e[b'beer/brewerId'] = int(e[b'beer/brewerId'])
         #   e[b'beer/brewerID'] = int(e[b'beer/brewerID'])

            timeStruct = time.gmtime(e[b'review/timeUnix'])
           # e[b'review/timeStruct'] = dict(zip(["year", "mon", "mday", "hour", "min", "sec", "wday", "yday", "isdst"], list(timeStruct)))
            data_list.append(e)
        except Exception as q:
           # print(q)
            pass
        counter += 1
        if counter == max_rows:
            break

    pass
    dF = pd.DataFrame(data_list)
    decodedDF = pd.DataFrame()
    for key in dF.keys():
        if type(dF[key][0]) == type(b'a'):
            decodedDF[key.decode("utf-8").replace('/','_')] = dF[key].str.decode("utf-8")
        else:
            decodedDF[key.decode("utf-8").replace('/','_')] = dF[key]


    decodedDF['user_index'] = np.zeros((len(decodedDF),),dtype=np.int)
    decodedDF['item_index'] = np.zeros((len(decodedDF),),dtype=np.int)
    user_dict = {}
    item_dict = {}

    print('user_profileName')
    for i, name in enumerate(decodedDF['user_profileName'].unique()):
        user_dict[name] = i

    #    print('beer_beerId')
    for i, name in enumerate(decodedDF['beer_beerId'].unique()):
        item_dict[name] = i

    decodedDF['user_index']= decodedDF['user_profileName'].map(user_dict)
    decodedDF['item_index']= decodedDF['beer_beerId'].map(item_dict)

    return decodedDF

def write_data(dF,data_path,data_name='data.csv',dropped_keys = []):
    for key in dropped_keys:
        if key in dF.keys():
            dF = dF.drop(key,axis=1)
    dF.to_csv(data_path + data_name,index=False)

def main():
    config_dict = read_config()
    print('Requesting Data')
    request_data(url=config_dict['data_url'],data_path=config_dict['data_path'],force_download=False)
    print('Formatting Data')
    dF = format_data(data_path = config_dict['data_path'])
    print('Saving Data as CSV')
    print('Head Before Removing')
    print(dF.head())
    write_data(dF=dF, data_path = config_dict['data_path'],data_name = config_dict['data_name'], dropped_keys=config_dict['dropped_keys'])
    if config_dict['remove_raw_data']:
        print('Removing Raw Data')
        os.remove(config_dict['data_path'] + 'raw_data.txt.gz')
    write_dictionary(dF)


if __name__ == '__main__':
    config_dict = read_config()

    #main()
    dF = pd.read_csv(config_dict['data_path'] + config_dict['data_name'])
