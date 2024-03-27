#!/usr/bin/env python3


"""
Useless
"""


import requests


BASE_LAUNCH_URL = 'https://api.spacexdata.com/v4/launches'
BASE_ROCKET_URL = "https://api.spacexdata.com/v4/rockets/"

if __name__ == '__main__':

    rockets = {}

    launches = requests.get(BASE_LAUNCH_URL).json()

    for launch in launches:
        rocket_id = launch.get('rocket')
        print(rocket_id)
