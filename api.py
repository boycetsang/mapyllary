import requests
import numpy as np
import cv2, imutils
import os

import logging
from colorlog import ColoredFormatter

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

# main class for URL query to mapillary
class mapyllary:
    def __init__(self, cmd=False):
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

    def set_clinet_id(self, client_id):
        '''
            client ID should be obtained from Mapillary to use any query function
        '''
        self.client_id = client_id

    def clean_values(self, kwargs):
        # utility function for input data type tolerance
        if 'closeto' in kwargs:
            if isinstance(kwargs['closeto'], list) or isinstance(kwargs['closeto'], tuple):
                kwargs['closeto'] = kwargs['closeto'][0] + ',' + kwargs['closeto'][1]
            if isinstance(kwargs['closeto'], dict):
                kwargs['closeto'] = kwargs['closeto']['long'] + ',' + kwargs['closeto']['lat']      
        if 'values' in kwargs:
            if isinstance(kwargs['values'], list) or isinstance(kwargs['values'], tuple):
                kwargs['values'] = ','.join(kwargs['values'])
       
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
        if input_json is None:
            input_json = self.last_ret
        for f in input_json['features']:
            key = f['properties']['key']

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


    def search_seg(self, input_json=None, **kwargs):
        '''
            Request the segmentation data, then directly print them on the image.
        '''
        if input_json is None:
            input_json = self.last_ret
        allowed_kwarg = ['bbox', 'closeto', 'max_score', 
                         'min_score', 'organization_keys', 'per_page', 
                         'private', 'radius', 'tile', 'userkeys', 
                         'usernames', 'values']
        forbidden = [k for k in kwargs if k not in allowed_kwarg]
        if forbidden:
            raise ValueError(forbidden + ' is not allowed in a segmentation query.')

        kwargs = self.clean_values(kwargs)
        value_dict = dict()
        ft_cnt = 0
        for f in input_json['features']:
            image_key = f['properties']['key']
            kwargs['image_keys'] = image_key
            url = self.SEG_SEARCH_URL
            url += 'client_id=' + self.client_id
            query_param = '&'.join([arg + '=' + str(value) for (arg, value) in kwargs.items()])
            if query_param:
                url += '&' + query_param

            logger.info('Sending Segmentation Request to Mapillary ...')
            self.last_url = url
            self.last_req = requests.get(url)
            logger.info('Mapillary Returned with {}...'.format(str(self.last_req)))
            self.last_ret = self.last_req.json()
            img_dir = os.path.join(self.resources_path, image_key)
            img_path = '{}/orig.jpg'.format(img_dir)
            des_path = '{}/wseg.jpg'.format(img_dir)
            img = cv2.imread(img_path)

            for f_ret in self.last_ret['features']:
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
        if input_json is None:
            input_json = self.last_ret

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



