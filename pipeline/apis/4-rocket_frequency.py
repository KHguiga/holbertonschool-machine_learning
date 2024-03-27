#!/usr/bin/env python3
"""
script that displays the number of launches per rocket,
using the (unofficial) SpaceX API
"""
import requests


if __name__ == '__main__':

    url = 'https://api.spacexdata.com/v3/launches'
    res = requests.get(url).json()
    rockets = {}
    # user doesn’t exist
    if requests.get(url).status_code == 404:
        print('Not found')
    for launch in res:
        rocket_name = launch['rocket']['rocket_name']
        if rocket_name in rockets:
            rockets[rocket_name] += 1
        else:
            rockets[rocket_name] = 1

    rockets = sorted(rockets.items(),
                     key=lambda x: x[1],
                     reverse=True)

    for rocket, nb_launches in rockets:
        print('{}: {}'.format(rocket, nb_launches))