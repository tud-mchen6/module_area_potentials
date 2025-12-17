"""
This script collects all technology areas of the given shape, use assumed technology
cost data, renewables density and discount rate to calculate the cost of each pixel,
then produce renewables cost curves and land curves to account for the accumulated
cost and land needed for a given amount of yearly vRES production.
"""

import os
import pandas as pd
import geopandas as gpd
import rioxarray as rxr