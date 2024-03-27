#!/usr/bin/env python3
"""
script that displays the number of launches per rocket,
using the (unofficial) SpaceX API
"""
import requests

if __name__ == '__main__':

    url = 'https://api.spacexdata.com/v3/launches'
    try:
        res = requests.get(url).json()
    except:
        print("ok")