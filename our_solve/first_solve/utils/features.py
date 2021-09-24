from geopy.geocoders import Nominatim
from geopy.distance import lonlat, distance
from typing import Tuple, List
import numpy as np
from tqdm import tqdm
from typing import Optional
import pandas as pd


def process_floor(smt: object):
    if pd.isna(smt):
        return None
    try:
        return int(smt)
    except(ValueError):
        try:
            numbers = [str(i) for i in range(10)]
            result = []
            index = 0
            indexes = [smt.find(elem) for elem in numbers if smt.find(elem) != -1] + [99999]
            l_min = min(indexes)
            if l_min == 99999:
                return None
            else:
                while True:
                    try:
                        result += [int(smt[l_min])]
                        l_min += 1
                    except(ValueError):
                        break
                    except(IndexError):
                        break
                return int("".join(str(elem) for elem in result))
        except(ValueError):
            return None


def get_distances(place_coords: List[Tuple[float, float]],
                  citys: List[str],
                  regions: List[str]) -> List[float]:
    geolocator = Nominatim(user_agent="example app")
    #     locations_place = [geolocator.reverse(*coords) for coord in coords]
    city_coords = [geolocator.geocode(f"{citys[i]}, {str(regions[i])}, Russia").point for i in tqdm(range(len(citys)))]
    distances = [distance(lonlat(*place_coords[i]), lonlat(*city_coords[i])) for i in tqdm(range(len(city_coords)))]
    return distances
