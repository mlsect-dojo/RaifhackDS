from geopy.geocoders import Nominatim
from geopy.distance import lonlat, distance
from typing import Tuple, List
import numpy as np
from tqdm import tqdm


def get_distances(place_coords: List[Tuple[float, float]],
                  citys: List[str],
                  regions: List[str]) -> List[float]:
    geolocator = Nominatim(user_agent="example app")
    #     locations_place = [geolocator.reverse(*coords) for coord in coords]
    city_coords = [geolocator.geocode(f"{citys[i]}, {str(regions[i])}, Russia").point for i in tqdm(range(len(citys)))]
    distances = [distance(lonlat(*place_coords[i]), lonlat(*city_coords[i])) for i in tqdm(range(len(city_coords)))]
    return distances
