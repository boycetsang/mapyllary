# mapyllary
Unofficial API to request image and segmentation from Mapillary.   
This package is hosted at https://pypi.org/project/mapyllary/  
Note: The OSM-related functionality of this package is imported from `overpass`
https://github.com/mvexel/overpass-api-python-wrapper


## Install
This package requires python version >= 3.5
```
pip install mapyllary
```
Note that installation will take a while due to dependency on ML & CV libraries

## How to obtain a Mapillary client ID
Requests to Mapillary server must be identified by a **client id**, and you can obtain one for free (as of now) by registering an account here.
https://www.mapillary.com/
Then, go to your Dashboard > Developer to generate your client ID.

## Test run
You can run the `main.py` as a command line tool to download images from Mapillary around a certain GPS coordinates, or a collection of locations obtained by OpenStreetMap filters.
### Instructions
1.  git clone https://github.com/boycetsang/mapyllary.git
2.  Download images from a GPS coorindate (long, lat): `python main.py -c <your_client_id> -p -122.872700 45.543663`
3.  Download images from a filter (OSM style; e.g. all the highway nodes that are motorway_junction within Washington state): `python main.py -c <your_client_id> -n highway motorway_junction Seattle`
4.  Bonus: Applying a segmentation model to downloaded images
`python main.py -c "dURwZ0J4TE1kaVZKd1lQbkhmaHNqajozMzgxNjFkNDBmMjk0ZWFk" -n highway motorway_junction "North Plains" -m model/enet-model.net -l model/enet-label.txt -w 1024 -t 512`


## Using the API
Here is a colab notebook that illustrate how the API can be used to request images and segmentation data from Mapillary.

https://colab.research.google.com/drive/15V9A7Z7oiOoZlNB4_9upXnPauOxPvmif?usp=sharing

