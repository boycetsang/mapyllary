import api
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interface with Mapillary.")
    parser.add_argument("-c", "--client_id", required=True,
        help="mapillary client id")
    parser.add_argument("-p", "--position", nargs=2,
        help="long, lat pair to define desired location") 
    parser.add_argument("-n", "--name", nargs=3,
        help="instead of providing a definite location, "
             "you can define a filter to search in OSM instead.") 
    parser.add_argument("-s", "--segmentation", type=str,
        help="labels to request from Mapillary", 
        default=['marking--continuous--solid', 
                 'marking--continuous--dashed'])
    parser.add_argument("-m", "--model",
        help="path to segmentation model; optional")
    parser.add_argument("-l", "--labels",
        help="path to .txt file containing labels, when a model is specified")  
    parser.add_argument("-r", "--colors", type=str,
        help="path to .txt file containing colors for labels, when a model is specified")
    parser.add_argument("-w", "--width", type=int, default=500,
        help="desired width (in pixels) of the model, when a model is specified")
    parser.add_argument("-t", "--height", type=int, default=500,
        help="desired height (in pixels) of the model, when a model is specified")
    args = vars(parser.parse_args())

    mp = api.mapyllary()
    logger = api.logger
    op = api.overpass()
    if args['position'] is not None:
        pos = [args['position']]
    elif args['name'] is not None:
        features = op.find_features_in_area(args['name'][0], args['name'][1], args['name'][2])
        pos = op.node_to_pos(features.get_nodes())
    else:
        logger.critical("Either position or name must be defined.")
        raise ValueError
    for p in pos:
        logger.info("Processing position lat: {lat}, long: {long}".format(lat=p[0], long=p[1]))
        mp.set_clinet_id(args['client_id'])
        
        mp.search_images(closeto=p, radius=50, unique_users=True)
        mp.download_images(res=2048)
        mp.add_info(op.find_ways_by_coord(p))
        mp.store_info()
        if args['segmentation'] is not None:
            mp.search_seg(values=args['segmentation'])
            mp.gen_seg_images()
        if args['model'] is not None:
            mp.apply_model(model=args['model'],
                           labels=args['labels'],
                           colors=args['colors'],
                           width=args['width'],
                           height=args['height'])