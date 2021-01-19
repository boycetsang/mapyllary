import api
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interface with Mapillary.")
    parser.add_argument("-c", "--client_id", required=True,
        help="mapillary client id")
    parser.add_argument("-p", "--position", required=True, nargs=2,
        help="lat, long pair to define desired location") 
    parser.add_argument("-m", "--model",
        help="path to deep learning segmentation model")
    parser.add_argument("-l", "--labels",
        help="path to .txt file containing labels")  
    parser.add_argument("-r", "--colors", type=str,
        help="path to .txt file containing colors for labels")
    parser.add_argument("-w", "--width", type=int, default=500,
        help="desired width (in pixels) of the model")
    args = vars(parser.parse_args())

    mp = api.mapyllary()
    mp.set_clinet_id(args['client_id'])
    ret = mp.search_images(closeto=args['position'], radius=50, unique_users=True)
    mp.download_images(res=2048)
    mp.search_seg(values='marking--continuous--solid,marking--continuous--dashed')
    if args['model'] is not None:
        mp.apply_model(model=args['model'],
                       labels=args['labels'],
                       colors=args['colors'],
                       width=args['width'],
                       height=args['height'])