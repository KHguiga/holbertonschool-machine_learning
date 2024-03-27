import requests


if __name__ == '__main__':

    url = 'https://api.spacexdata.com/v3/launches'
    res = requests.get(url).json()
    print("ok")