import rockhound as rh
import matplotlib.pyplot as plt
from pdb import set_trace as bp

def collect_region(options):

    # Load a version of the topography grid
    grid = rh.fetch_etopo1(version="bedrock")
    
    # Select a subset that corresponds to 
    region = grid.sel(latitude=slice(options['region']['lat-min'], options['region']['lat-max']), longitude=slice(options['region']['lon-min'], options['region']['lon-max']))
    
    topography = {
        'topo': region.variables['bedrock'].values,
        'latitude': region.variables['latitude'].values,
        'longitude': region.variables['longitude'].values,
        }
    
    return topography

##########################
if __name__ == '__main__':

    ## Options
    options = {}
    options['region'] = {
        'lat-min': 11.5,
        'lat-max': 14.75,
        'lon-min': 39.25, 
        'lon-max': 43.25,
        }

    topography = collect_region(options)

    plt.figure(); plt.pcolormesh(topography['longitude'], topography['latitude'], topography['topo'], shading='auto'); plt.show()

    bp()