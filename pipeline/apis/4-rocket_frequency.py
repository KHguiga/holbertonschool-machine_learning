#!/usr/bin/env python3
"""
script that displays the number of launches per rocket,
using the (unofficial) SpaceX API
"""
import requests


if __name__ == '__main__':

    url = 'https://api.spacexdata.com/v3/launches'
    # user doesn’t exist
    if requests.get(url).status_code == 404:
        print('Not found')