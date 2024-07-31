from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import time
import sys
import argparse
import collections
import operator
import os
import json

colors = []
def random_colors():
    for i in range(20):
        color1 = (list(np.random.choice(range(256), size=3)))  
        color =[255, int(color1[1]), int(color1[2])]  
        colors.append(color)

# draw text alongwith a rectangle in the background
def draw_text(img, text, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), font_scale=1.5,
        font_thickness=2, text_color=(0, 255, 0), text_color_bg=(0, 0, 0), padding=5):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w+2*padding, y + text_h+2*padding), text_color_bg, -1)
    cv2.putText(img, text, (x+padding, y + text_h + padding), font, font_scale, text_color, font_thickness)

    return text_size

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
    P.add_argument('-r', '--region', type=str,
                   help='name of file which contains region of interest coordinates')
    P.add_argument('-c', '--camera', type=str,
                   help='camera id')
    P.add_argument('-d', '--distthresh', default=2, type=int,
                   help='number of pixels by which object moves in a frame to be considered non-stationary')
    P.add_argument('-t', '--timethresh', default=5, type=int,
                   help='number of frames of inactivity before an object is marked stationary')
    return P.parse_args()

from datetime import datetime
from datetime import timedelta

def get_time_from_fname(fname, seconds):
    fname = fname.split('/')[-1]
    (d, t) = fname.split('_')[0:2]
    date = datetime.strptime(d+' '+t, '%Y-%m-%d %H-%M-%S') +  timedelta(seconds=seconds)
    return date.strftime('%Y-%m-%d %H:%M:%S')

regional_id = []
next_regional_id = []

def get_objects_in_regions(obj_list, region_list):
    global regional_id
    global next_regional_id

    objs_per_region = []

    region_index = 0
    for r in region_list:
        obj_in = []
        for index,row in obj_list.iterrows(): 
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            global_id = int(row[4])
            prob = round(row[5],2)

            # check if object has entered RoI
            result = cv2.pointPolygonTest(np.array(r['region'], np.int32), 
                                      ((x1+x2)/2,(y1+y2)/2), False) > 0
            if result: 
                if regional_id[region_index].get(global_id, -1) == -1:
                    regional_id[region_index][global_id] = next_regional_id[region_index]
                    next_regional_id[region_index] += 1
                obj_in.append(row)
        objs_per_region.append(obj_in)
        region_index += 1

    return objs_per_region

prevPos = {}
stationary = collections.Counter()
allstat = {}

def get_stationary_objects(obj_list, fprocessed):
    global prevPos
    global stationary
    global allstat

    for index,row in obj_list.iterrows(): 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        global_id = int(row[4])
        # check if object is stationary
        pp = prevPos.get(global_id, -1)
        if  pp != -1:
            if (abs(x1 - pp[0]) < args.distthresh) and (abs(y2-pp[1]) < args.distthresh):
                stationary[global_id] += 1
                if stationary[global_id] > args.timethresh:
                    #cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 4)
                    if args.verbose: print('Stationary object: id %d frame %d/%d' %
                            (global_id, stationary[global_id], fprocessed))
                    allstat[(global_id, fprocessed-stationary[global_id])] = stationary[global_id]
            else:
                stationary[global_id] = 0
        else:
            stationary[global_id] = 0
        prevPos[global_id] = (x1, y2)

# process a single file
def process_file(video_path, out_path, model, title, region_list):
    global stationary

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

    # read frames
    startTime = time.time()
    prevTime = time.time()
    fprocessed = 0
    fPeriod = 20
    probHist = collections.Counter()
    highest_id = 0

    ret = True
    if args.skipframes:
        while ret:
            ret, frame = cap.read()
            fprocessed += 1
            if fprocessed >= args.skipframes: 
                break

    while ret:
        # read next frame
        ret, frame = cap.read()
        if not ret: break

        if not args.batch:
            v_string = '"v" to start video, ' if waitDur == 0 else ''
            draw_text(frame, 'Press ' + v_string + '"q" to quit', pos=(100, 50))

            if args.live: 
                draw_text(frame, '"' + args.live.split('/')[-1] + 
                          '" Time: %s' % get_time_from_fname(video_path, fprocessed/fps), 
                          pos=(100, 80))            

        # output stats
        if fprocessed % fPeriod == 0: 
            intTime = time.time() -  prevTime
            print('[%5.1f]%5d/%d frames processed: (%d ms/frame)' % 
                    (intTime, fprocessed, fc, intTime*1000/fPeriod))
            prevTime = time.time()
            if fprocessed >= args.numframes: break

        # detect and track objects
        results = model.track(frame, persist=True, verbose=args.verbose)
        a = results[0].boxes.data
        obj_list = pd.DataFrame(a).astype("float")

        for indx, row in obj_list.iterrows(): 
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            global_id = int(row[4])
            if global_id > highest_id: highest_id = global_id
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 1)

        objs_per_region = get_objects_in_regions(obj_list, region_list)
        for i in range(len(region_list)):
            for row in objs_per_region[i]: 
                x1=int(row[0])
                y1=int(row[1])
                x2=int(row[2])
                y2=int(row[3])
                global_id = int(row[4])
                prob = round(row[5],2)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)                                
                draw_text(frame, str(regional_id[i][global_id]) + ' : ' + str(prob), 
                          pos=(x1,y1), font_scale=0.8, font_thickness=1, 
                          text_color=(255,255,255), text_color_bg=(0,0,255), padding=1)

        get_stationary_objects(obj_list, fprocessed)
        for indx, row in obj_list.iterrows(): 
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            global_id = int(row[4])
            if stationary[global_id] > args.timethresh:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 4)

        if not args.batch:
            i = 0
            for r in region_list:
                draw_text(frame, '%-10s: %d' % (r['description'][:10],next_regional_id[i]-1), 
                        pos=(100, 110+i*20))
                cv2.polylines(frame,[np.array(r['region'],np.int32)],True,(255,0,255),2)        
                i += 1

            cv2.imshow(title, frame)
    
        video_writer.write(frame)
        fprocessed += 1

        if not args.batch:
            keyp = cv2.waitKey(waitDur) & 0xFF
            if keyp == ord('q'):
                break
            elif keyp == ord('v'):
                waitDur = args.wait
                startTime = time.time()
                prevTime = time.time()


    # release resources
    cap.release()
    video_writer.release()
    if not args.batch: cv2.destroyAllWindows()

    print('')
    print('Total frames counted = %d' % fprocessed)
    print('Total objects seen = %d' % highest_id)
    i = 0
    for r in region_list:
        print('Objects in %-10s : %d' % (r['description'][:10], next_regional_id[i]))
        i += 1

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
    global args
    global next_regional_id
    global regional_id

    args = arg_parser()
    random_colors()
    title = 'Viewer'
    
    print('')
    if args.batch:
        print('Running in batch mode (numframes %d)...' % args.numframes)
    else:
        print('Running in interactive mode ...')

    region_list = get_region_list_for_camera(args.camera)
    print('Region list: %s' % region_list)
    next_regional_id = [1 for r in region_list]
    regional_id = [{} for r in region_list]

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

    process_file(video_path, out_path, model, title, region_list)

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

            process_file(video_path, out_path, model, title, region_list)
            

if __name__ == "__main__":
    main()
