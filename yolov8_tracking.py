from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import time
import sys
import argparse
import signal
import collections
import operator
import math

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

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
    global regionDefine
    global region
    global lastTime
    global currentPos

    pos = [x, y]
    if event == cv2.EVENT_MOUSEMOVE:
        currentPos = pos

    if event == cv2.EVENT_LBUTTONDOWN:
        print('Mouse click event, regionDefine', regionDefine)
    if regionDefine and (event == cv2.EVENT_LBUTTONDOWN):
        delta = time.time() - lastTime
        if delta < 0.5:
            # treat this as a double click
            region = []
        else:
            region.append(pos)
        print(region)
        lastTime = time.time()

# function to define a region
def define_region(title, frame):
    global region
    global regionDefine
    global area

    regionDefine = True
    region = []

    draw_text(frame, 'Define region using mouse, single click to add vertex, double click to reset', pos=(100, 20))
    draw_text(frame, 'Press "s" when done, to save and continue', pos=(100, 50))
    cv2.imshow(title, frame)

    # wait here until s is pressed
    while cv2.waitKey(200) & 0xFF != ord('s'):
        cache = frame.copy()
        if len(region) > 0: 
            cv2.polylines(frame,[np.array(region+[currentPos],np.int32)],True,(255,0,255),2)        
        cv2.imshow(title, frame)
        frame  = cache.copy()

    if len(region) < 3: 
        print('Too few points (%d), discarding region\n' % len(region))
        region = []
    else:
        print('New final region:', region)
        area = region
        ofilename = 'region_new.txt'
        with open(ofilename, 'w') as ofile:
            ofile.write(str(area))
        print('Written new region to file %s\n' % ofilename)
        region = []

    regionDefine = False
    return frame

import os
def get_latest_file(dir):
    fnames = sorted([fname for fname in os.listdir(dir) if fname.endswith('.mp4')])
    return dir+'/'+fnames[-2]

OUT_DIR = 'output'
def get_output_path(video_path):
    fname = video_path.split('/')[-1]
    if not os.path.exists(OUT_DIR):
        print('Creating output directory:', OUT_DIR)
        os.makedirs(OUT_DIR)
    out_path = OUT_DIR + '/out-' + fname 
    return out_path


# Argument parser
def arg_parser():
    P = argparse.ArgumentParser()
    P.add_argument('-n', '--numframes', default=0, type=int,
                   help='Maximum number of frames to process')
    P.add_argument('-s', '--skipframes', default=0, type=int,
                   help='Number of frames to skip')
    P.add_argument('-w', '--wait', default=1, type=int,
                   help='Miliseconds to wait between each frame processing')
    P.add_argument('-b', '--batch', action='store_true',
                   help='Run in batch mode (no video output)')
    P.add_argument('-v', '--verbose', action='store_true',
                   help='Verbose log output')
    P.add_argument('-l', '--live', type=str,
                   help='name of directory which contains live streaming files')
    P.add_argument('-f', '--file', default='data/wfl.mp4', type=str,
                   help='name of file to process')
    P.add_argument('-m', '--model', default='../models/yolov8s-pt-20.pt', type=str,
                   help='model to be used')
    P.add_argument('-r', '--region', default='region.txt', type=str,
                   help='name of file which contains region of interest coordinates')
    P.add_argument('-d', '--distthresh', default=2, type=int,
                   help='number of pixels by which object moves in a frame to be considered non-stationary')
    P.add_argument('-t', '--timethresh', default=5, type=int,
                   help='number of frames of inactivity before an object is marked stationary')
    return P.parse_args()

# global variables
regionDefine = False
region = []
area=[]
lastTime = time.time()
currentPos = (0,0)

from datetime import datetime
from datetime import timedelta

def get_time_from_fname(fname, seconds):
    fname = fname.split('/')[-1]
    (d, t) = fname.split('_')[0:2]
    date = datetime.strptime(d+' '+t, '%Y-%m-%d %H-%M-%S') +  timedelta(seconds=seconds)
    return date.strftime('%Y-%m-%d %H:%M:%S')

# process a single file
def process_file(video_path, out_path, model, title, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Failed to open video:', video_path)
        sys.exit(1)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = fps/3

    if args.numframes == 0: 
        args.numframes = fc

    print('Total frames: %d (%dx%d fps %d)' % (fc, h, w, fps))
    print('Output fps  : %d' % out_fps)
    print('')

    # setup video writer
    #fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1') #.avi
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w,h))

    if args.live is None:
        waitDur = 0
    else:
        waitDur = args.wait

    ids_seen = set()
    highest_id = 0

    id_in_region = {}
    highest_id_in_region = 1
    prevPos = {}
    stationary = collections.Counter()
    allstat = {}

    # read frames
    startTime = time.time()
    prevTime = time.time()
    prevHighest = 1
    fprocessed = 0
    fPeriod = 20
    probHist = collections.Counter()

    ret = True
    while ret:
        ret, frame = cap.read()
        fprocessed += 1
        if fprocessed > args.skipframes: 
            break

    while ret:
        if not args.batch:
            cache = frame.copy()
            v_string = '"v" to start video, ' if waitDur == 0 else ''
            draw_text(frame, 'Press "r" to redefine region of interest, ' + v_string + '"q" to quit', pos=(100, 50))

            if args.live: 
                draw_text(frame, '"' + args.live.split('/')[-1] + '" Time: %s' % get_time_from_fname(video_path, fprocessed/fps), pos=(100, 80))            
            draw_text(frame, 'Unique count: %d' % len(ids_seen), pos=(100, 110))
            cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,255),2)        

            cv2.imshow(title, frame)

            # allow region to be (re)defined before we detect objects
            keyp = cv2.waitKey(waitDur) & 0xFF
            if keyp == ord('q'):
                break
            elif keyp == ord('v'):
                waitDur = args.wait
                startTime = time.time()
                prevTime = time.time()
            elif keyp == ord('r'):
                frame = cache.copy()
                # blocking call to define the RoI
                frame = define_region(title, frame)

        # output stats
        if fprocessed % fPeriod == 0: 
            intTime = time.time() -  prevTime
            print('[%5.1f]%5d/%d frames processed: %3d objects (%d ms/frame)' % 
                    (intTime, fprocessed, fc, highest_id_in_region-prevHighest, intTime*1000/fPeriod))
            prevHighest = highest_id_in_region
            prevTime = time.time()
            if fprocessed >= args.numframes: break

        # read next frame
        ret, frame = cap.read()
        if not ret: break

        # detect and track objects
        results = model.track(frame, persist=True, verbose=args.verbose)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        for index,row in px.iterrows(): 
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            id = int(row[4])
            prob = round(row[5],2)
            # check if object has entered RoI
#            result = (cv2.pointPolygonTest(np.array(area, np.int32), ((x1,y2)), False) > 0) or \
#                        (cv2.pointPolygonTest(np.array(area, np.int32), ((x2,y1)), False) > 0)
            result = (cv2.pointPolygonTest(np.array(area, np.int32), ((x1+x2)/2,(y1+y2)/2), False) > 0) 
            if result:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                
                probHist[math.ceil(prob*10)/10] += 1
                if id not in ids_seen:
                    # object seen for first time
                    ids_seen.add(id)
                    if id > highest_id: highest_id = id

                    id_in_region[id] = highest_id_in_region
                    highest_id_in_region += 1
                
                draw_text(frame, str(id_in_region[id]) + ' : ' + str(prob), 
                          pos=(x1,y1), font_scale=0.8, font_thickness=1, 
                          text_color=(255,255,255), text_color_bg=(0,0,255), padding=1)
            else:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 1)

            # check if object is stationary
            pp = prevPos.get(id, -1)
            if  pp != -1:
                if (abs(x1 - pp[0]) < args.distthresh) and (abs(y2-pp[1]) < args.distthresh):
                    stationary[id] += 1
                    if stationary[id] > args.timethresh:
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 4)
                        if args.verbose: print('Stationary object: id %d(%d) frame %d/%d' %
                               (id, id_in_region.get(id,-1), stationary[id], fprocessed))
                        allstat[(id, fprocessed-stationary[id])] = stationary[id]
                else:
                    stationary[id] = 0
            else:
                stationary[id] = 0
            prevPos[id] = (x1, y2)

        if args.live: draw_text(frame, 'Time: %s' % get_time_from_fname(video_path, fprocessed/fps), pos=(100, 80))            
        draw_text(frame, 'Unique count: %d' % len(ids_seen), pos=(100, 110))
        cv2.polylines(frame,[np.array(area,np.int32)],True,(255,0,255),2)        
        if not args.batch: cv2.imshow(title, frame)

        video_writer.write(frame)
        fprocessed += 1

    # release resources
    cap.release()
    video_writer.release()
    if not args.batch: cv2.destroyAllWindows()

    print('')
    print('Total frames counted = %d' % fprocessed)
    print('Total unique objects seen = %d, (highest id %d)' % (len(ids_seen), highest_id))

    totTime = time.time() - startTime
    print('Total time: %3.1fs (%d ms/frame)' % (totTime, totTime*1000/fprocessed))
    print('')

    print('**** List of all stationary objects detected ****')
    l = []
    for i in allstat.items():
        l.append((i[0][0], i[0][1], i[1]))
    l = sorted(l, key=operator.itemgetter(1))
    print("id\tframe\tduration")
    for i in l:
        print('%d\t%d\t%d' % i)
    print('')

    probList = sorted(probHist.items(), key=operator.itemgetter(0))    
    tot = sum([x[1] for x in probList])
    print('**** Histogram of confidence probability ****')
    print('Prob : Freq  Cumulative')
    cum = 0
    for i in probList:
        p = i[0]
        f = i[1]
        cum += f
        print('%4.1f : %5d %5d %3d%%' % (p, f, cum, cum*100/tot))
    print('')

# main function
def main():
    global area

    signal.signal(signal.SIGINT, signal_handler)
    args = arg_parser()

    title = 'Viewer'
    
    if not args.batch:
        cv2.namedWindow(title)
        cv2.setMouseCallback(title, mouse_call_back)

    print('')
    if args.batch:
        print('Running in batch mode (numframes %d)...' % args.numframes)
    else:
        print('Running in interactive mode ...')

    print('Region file: %s' % args.region)
    with open(args.region) as f:
        area_str = f.readline()  
    area = eval(area_str)
    print('RoI: ', area)

    # load yolov8 model
    model = YOLO(args.model)
    print('Model used  : %s' % args.model)

    # check if live feed is provided
    if args.live is not None:
        print('Directory for live feed:', args.live)
        video_path = get_latest_file(args.live)
    else:
        # load video
        video_path = args.file
    out_path = get_output_path(video_path)

    print('\nInput file  :', video_path)
    print('Output file :', out_path)

    process_file(video_path, out_path, model, title, args)

    if args.live is not None:
        wait_count = 0
        while True:
            prev_fname = video_path
            video_path = get_latest_file(args.live)
            
            if video_path == prev_fname: 
                wait_count += 1
                if wait_count < 10:
                    print('**** No new file found, waiting %d ...' % wait_count)
                    time.sleep(2)
                    continue
                else:
                    print('**** No new file found, stopping ****')
                    break
            
            wait_count = 0
            out_path = get_output_path(video_path)
            print('Input file  :', video_path)
            print('Output file :', out_path)

            process_file(video_path, out_path, model, title, args)
            

if __name__ == "__main__":
    main()
