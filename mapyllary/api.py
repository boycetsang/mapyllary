import requests
import numpy as np
import cv2, imutils
import os
from bs4 import BeautifulSoup 
import overpy
import logging
import urllib
from colorlog import ColoredFormatter
import pandas as pd
op = overpy.Overpass()

# Pre-defined colors for segmentation
COLOR_CYCLE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), 
               (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), 
               (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), 
               (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), 
               (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), 
               (255, 255, 255), (0, 0, 0)]

# setting up logging
formatter = ColoredFormatter(
    "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger('log')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class overpass:
    def __init__(self):
        pass
    def area_id_by_name(self, area_name):
        # helper function to get area id for overpass 
        url = "https://www.openstreetmap.org/geocoder/search_osm_nominatim?query=" + urllib.parse.quote(area_name)
        r = requests.get(url) 
        soup = BeautifulSoup(r.content, 'html5lib')
        osm_link = soup.find('a', attrs={'class':'set_position'})
        relation_id = osm_link.get('data-id').strip()
        logger.info("Relation ID retrieved for "
                    "{area} is {r_id}".format(
                    area=area_name, r_id=relation_id))
        # 3600000000 + osm relation id is the overpass area id
        return int(relation_id) + 3600000000

    def find_features_in_area(self, feature_type, feature_key, area):
        query = '''
[out:json][timeout:25];
area({area_id})->.searchArea;
(
  node["{feature_type}"="{feature_key}"](area.searchArea);
  way["{feature_type}"="{feature_key}"](area.searchArea);
  relation["{feature_type}"="{feature_key}"](area.searchArea);
);
out body;
>;
out skel qt;
'''
        if not isinstance(area, int):
            area = self.area_id_by_name(area)
        query = query.format(feature_type=feature_type, feature_key=feature_key,
                             area_id=area)
        return op.query(query)

    def node_to_pos(self, node_list):
        return zip([n.lon for n in node_list], [n.lat for n in node_list])

    def find_ways_by_coord(self, coords, around=3):
        query = ("way(around:{around}"
                ",{lat}"
                ",{long}"
                ");out;")
        query = query.format(around=around, long=coords[0], lat=coords[1])
        ret = op.query(query).get_ways()
        return [way.tags for way in ret]

# main class for URL query to mapillary
class mapyllary:
    def __init__(self):
        self.client_id = ''
        self.uuid = ''
        # setup some constant URLs
        self.IMAGE_PATH_URL = "https://images.mapillary.com/{key}/thumb-{res}.jpg"
        self.IMAGE_SEARCH_URL = 'https://a.mapillary.com/v3/images?'
        self.SEG_SEARCH_URL = 'https://a.mapillary.com/v3/object_detections/segmentations?'

        # set up history for reference
        self.last_url = ''
        self.last_req = ''
        self.last_ret = ''

        # default file path
        self.resources_path = 'resources'
        if not os.path.exists(self.resources_path):
            os.mkdir(self.resources_path)

        # setup pandas for db
        self.db = None

    def set_clinet_id(self, client_id):
        '''
            client ID should be obtained from Mapillary to use any query function
        '''
        self.client_id = client_id

    def get_last_json_if_undefined(self, input_json):
        if input_json is None:
            return self.last_ret

    def clean_values(self, kwargs):
        # utility function for input data type tolerance
        for k, v in kwargs.items():
            if isinstance(v, list) or isinstance(v, tuple):
                kwargs[k] = ','.join([str(e) for e in v])
        if 'closeto' in kwargs:
            if isinstance(kwargs['closeto'], dict):
                kwargs['closeto'] = kwargs['closeto']['long'] + ',' + kwargs['closeto']['lat']      

       
        return kwargs

    def search_images(self, unique_users=False, **kwargs):
        '''
        Use this function to find all image keys correspond to the defined filter.
        See this page for reference.
        https://www.mapillary.com/developer/api-documentation/#images
        Use unique_users to avoid getting similar images.
        '''

        allowed_kwarg = ['bbox', 'closeto', 'end_time', 'image_keys', 
                         'lookat', 'max_quality_score', 'min_quality_score', 
                         'organization_keys', 'pano', 'per_page', 
                         'private', 'radius', 'sequence_keys', 
                         'start_time', 'userkeys', 'usernames']
        forbidden = [k for k in kwargs if k not in allowed_kwarg]
        if forbidden:
            logger.critical(forbidden + ' is not allowed in a image query.')
            raise ValueError()

        kwargs = self.clean_values(kwargs)
        url = self.IMAGE_SEARCH_URL
        url += 'client_id=' + self.client_id
        query_param = '&'.join([arg + '=' + str(value) for (arg, value) in kwargs.items()])
        if query_param:
            url += '&' + query_param

        logger.debug('Sending Image Request to Mapillary ...')
        self.last_url = url
        self.last_req = requests.get(url)
        logger.debug('Mapillary Returned with {}...'.format(str(self.last_req)))
        self.last_ret = self.last_req.json()

        if unique_users:
            user_list = []
            new_list = []
            for f in self.last_ret['features']:
                prop = f['properties']
                if prop['user_key'] in user_list:
                    pass
                else:
                    new_list.append(f)
                    user_list.append(prop['user_key'])
            self.last_ret['features'] = new_list
        logger.info('Returned {} features'.format(len(self.last_ret['features'])))
        return self.last_ret

    def download_images(self, input_json=None, res=None, show=False):
        '''
            Download images from Mapillary server. If input_json is not used, then the last query return is used.
            The downloaded images are in the "resources" directory
        '''
        input_json = self.get_last_json_if_undefined(input_json)
        db_dict = dict()
        for f in input_json['features']:
            key = f['properties']['key']
            db_dict[key] = dict(f['properties'])
            db_dict[key]['lon'] = f['geometry']['coordinates'][0]
            db_dict[key]['lat'] = f['geometry']['coordinates'][1]
            if res is None:
                res = 320
            if res not in [320, 640, 1024, 2048]:
                logger.critical(("{res} is an unsupported resolution.".format(res=res)))
                raise ValueError

            pic_url = (self.IMAGE_PATH_URL.format(key=key, res=res))
            des_dir = os.path.join(self.resources_path, key)
            if not os.path.exists(des_dir):
                os.mkdir(des_dir)
            des = '{}/orig.jpg'.format(des_dir)
            db_dict[key]['image_path'] = os.path.realpath(des)
            with open(des, 'wb') as handle:
                response = requests.get(pic_url, stream=True)
                if not response.ok:
                    logger.critical(key, response)
                for block in response.iter_content(1024):
                    handle.write(block)
            if show:
                img = cv2.imread(des)
                cv2.imshow(key, img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()              

        if self.db is None:
            self.db = pd.DataFrame(db_dict).T
        else:
            self.db = pd.concat([self.db, pd.DataFrame(db_dict).T], 
                                sort=False)

    def add_info(self, dict_list=None, input_json=None):
        input_json = self.get_last_json_if_undefined(input_json)
        image_keys = [f['properties']['key'] for f in input_json['features']]
        if dict_list is not None:
            dict_cnt = 0
            for d in dict_list:
                d = {k + '_' + str(dict_cnt): v for k, v in d.items()}
                for k, v in d.items():
                    self.db.loc[self.db.index.isin(image_keys), k] = v
                dict_cnt += 1

    def store_info(self):
        store_path = os.path.join(self.resources_path, 'db.csv')
        logger.info('Storing metadata at %s' % store_path)
        self.db.to_csv(store_path)

    def search_seg(self, input_json=None, **kwargs):
        '''
            Request the segmentation data, then directly print them on the image.
        '''
        input_json = self.get_last_json_if_undefined(input_json)
        allowed_kwarg = ['bbox', 'closeto', 'max_score', 
                         'min_score', 'organization_keys', 'per_page', 
                         'private', 'radius', 'tile', 'userkeys', 
                         'usernames', 'values']
        forbidden = [k for k in kwargs if k not in allowed_kwarg]
        if forbidden:
            raise ValueError(forbidden + ' is not allowed in a segmentation query.')
        
        image_keys = [f['properties']['key'] for f in input_json['features']]
        kwargs['image_keys'] = image_keys
        kwargs = self.clean_values(kwargs)
        url = self.SEG_SEARCH_URL
        url += 'client_id=' + self.client_id
        query_param = '&'.join([arg + '=' + str(value) for (arg, value) in kwargs.items()])
        if query_param:
            url += '&' + query_param

        logger.debug('Sending Segmentation Request to Mapillary ...')
        self.last_url = url
        self.last_req = requests.get(url)
        logger.debug('Mapillary Returned with {}...'.format(str(self.last_req)))
        self.last_ret = self.last_req.json()

    def gen_seg_images(self, input_json=None):
        input_json = self.get_last_json_if_undefined(input_json)
        image_keys = [f['properties']['image_key'] for f in input_json['features']]
        value_dict = dict()
        for image_key in image_keys:
            img_dir = os.path.join(self.resources_path, image_key)
            img_path = '{}/orig.jpg'.format(img_dir)
            des_path = '{}/wseg.jpg'.format(img_dir)
            img = cv2.imread(img_path)
            ft_cnt = 0
            for f_ret in self.last_ret['features']:
                if not f_ret['properties']['image_key'] == image_key:
                    continue
                polygon_coords = []
                value = f_ret['properties']['value']
                if value not in value_dict:
                    value_dict[value] = COLOR_CYCLE[ft_cnt % len(COLOR_CYCLE)]
                    ft_cnt += 1 
                for coords in f_ret['properties']['shape']['coordinates']:
                    polygon_coords = (np.array(coords)  * np.flip(img.shape[0:2])).astype(np.int32)
                polygon_coords = polygon_coords.reshape((-1,1,2))
                cv2.polylines(img,polygon_coords,True, value_dict[value], 10)            
                ft_cnt += 1
            cv2.imwrite(des_path, img)

    def apply_model(self, input_json=None, model=None, labels=None, colors=None,
                    width=1024, height=512):
        # apply a cv2 dnn model, with supplied labels and model definition.
        CLASSES = open(labels).read().strip().split("\n")
        if colors:
            COLORS = open(colors).read().strip().split("\n")
            COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
            COLORS = np.array(COLORS, dtype="uint8")
        else:
            COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),
                dtype="uint8")
            COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        input_json = self.get_last_json_if_undefined(input_json)

        net = cv2.dnn.readNetFromTorch(model)
        value_dict = dict()
        ft_cnt = 0
        for f in input_json['features']:
            image_key = f['properties']['key']
            img_dir = os.path.join(self.resources_path, image_key)
            img_path = '{}/orig.jpg'.format(img_dir)
            des_path = '{}/wmodel.jpg'.format(img_dir)
            img = cv2.imread(img_path)

            image = imutils.resize(img, width=width, height=height)
            
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (1024, 512), 0,
                swapRB=True, crop=False)
            net.setInput(blob)
            output = net.forward()
            (numClasses, height, width) = output.shape[1:4]
            
            legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8")
            for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
                color = [int(c) for c in color]
                cv2.putText(legend, className, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),
                    tuple(color), -1)
                classMap = np.argmax(output[0], axis=0)
            mask = COLORS[classMap]
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            classMap = cv2.resize(classMap, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            output = ((0.4 * image) + (0.6 * mask)).astype("uint8")
            cv2.imwrite(des_path, output)



