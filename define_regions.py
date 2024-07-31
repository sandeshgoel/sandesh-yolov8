import cv2
import numpy as np
import time
import sys
import argparse
import json
import os

# draw text alongwith a rectangle in the background
def draw_text(img, text, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), font_scale=1.5,
        font_thickness=2, text_color=(0, 255, 0), text_color_bg=(0, 0, 0), padding=5):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w+2*padding, y + text_h+2*padding), text_color_bg, -1)
    cv2.putText(img, text, (x+padding, y + text_h + padding), font, font_scale, text_color, font_thickness)

    return text_size

# callback to handle mouse events
def mouse_call_back(event, x, y, flags, param):
    global region
    global lastTime
    global currentPos

    pos = [x, y]
    if event == cv2.EVENT_MOUSEMOVE:
        currentPos = pos

    if (event == cv2.EVENT_LBUTTONDOWN):
        delta = time.time() - lastTime
        if delta < 0.5:
            # treat this as a double click
            region = []
        else:
            region.append(pos)
        print(region)
        lastTime = time.time()

# Argument parser
def arg_parser():
    P = argparse.ArgumentParser()
    P.add_argument('-v', '--verbose', action='store_true',
                   help='Verbose log output')
    P.add_argument('-f', '--file', default='data/wfl.mp4', type=str,
                   help='name of file to process')
    P.add_argument('-r', '--region', type=str,
                   help='name of file which contains region of interest coordinates')
    P.add_argument('-c', '--camera', type=str,
                   help='camera id for which region is being defined')
    return P.parse_args()

# global variables
region = []
lastTime = time.time()
currentPos = (0,0)

def get_latest_video_file(camera):
    return args.file

def get_region_list_for_camera(camera):
    if args.region is None:
        region_file = 'regions/region-' + camera
    else:
        region_file = args.region

    if not os.path.isfile(region_file):
        return []

    with open(region_file, "r") as ifile:
        region_list = json.loads(ifile.read())
    print('[%s] Read region list from file %s (%d regions)\n' % 
          (camera, region_file, len(region_list)))
    
    return region_list

def set_region_list_for_camera(camera, region_list):
    if args.region is None:
        region_file = 'regions/region-' + camera
    else:
        region_file = args.region

    with open(region_file, 'w') as ofile:
        ofile.write(json.dumps(region_list))
    print('[%s] Written region list to file %s\n' % (camera, region_file))

def validate_region_list(region_list, w, h):
    ret = True
    for r in region_list:
        for v in r['region']:
            if v[0] > w or v[1] > h:
                print('ERROR: Region %s not within frame (%d,%d)' % 
                      (r['region'], w, h))
                ret = False
                break
    return ret

def define_regions_for_camera(camera): 
    global region

    if camera is None:
        print('Camera id must be provided. Exiting.')
        return -1
    
    # get the latest video file for the camera
    video_path = get_latest_video_file(camera)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Failed to open video:', video_path)
        return -1

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('Total frames: %d (%dx%d fps %d)' % (fc, h, w, fps))
    print('')

    # get the region list for this camera
    region_list = get_region_list_for_camera(camera)
    if not validate_region_list(region_list, w, h):
        print("Invalid regions for camera %s" % camera)
        return -1

    title = 'Viewer'    
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, mouse_call_back)

    # read next frame
    ret, orig_frame = cap.read()
    if not ret: 
        print('Failure trying to read frame')
        return -1

    region = []
    while True:
        frame = orig_frame.copy()

        draw_text(frame, 'Define region using mouse, single click to add vertex, double click to reset', pos=(100, 20))
        draw_text(frame, 'Press "s" when done, to save and continue', pos=(100, 50))
        draw_text(frame, 'Press "d" to delete a region, "q" to quit', pos=(100, 80))

        for r in region_list:
            pl = r['region']
            cv2.polylines(frame,[np.array(pl,np.int32)],True,(255,0,255),2)        
            draw_text(frame, r['description'], pos=pl[0], font_scale=1, font_thickness=1) 
        if len(region) > 0: 
            cv2.polylines(frame,[np.array(region+[currentPos],np.int32)],True,(255,0,255),2)
        cv2.imshow(title, frame)

        keyp = cv2.waitKey(200) & 0xFF
        if keyp == ord('q'):
            break        
        elif keyp == ord('d'):
                desc = input('Enter description for region to delete: ')
                region_list = [r for r in region_list if r['description'] != desc]
                set_region_list_for_camera(camera, region_list)
        elif keyp == ord('s'):
            if len(region) < 3: 
                print('Too few points (%d), discarding region\n' % len(region))
                region = []
            else:
                print('New final region:', region)
                desc = input('Enter a description for this region: ')
                rd = {
                    'region_id': 1, 'creation_time': int(time.time()),
                    'description': desc, 'region': region, 
                    'frame_height':h, 'frame_width': w
                    }
                region_list.append(rd)
                set_region_list_for_camera(camera, region_list)
                region = []

    # release resources
    cap.release()
    cv2.destroyAllWindows()

# main function
def main():
    global args
    args = arg_parser()

    print('\nInput file  :', args.file) 
    define_regions_for_camera(args.camera)

if __name__ == "__main__":
    main()
