import cv2
import time
import datetime as dt
import os
import signal
import argparse
import sys

# Argument parser
def arg_parser():
    P = argparse.ArgumentParser()
    P.add_argument('-v', '--verbose', action='store_true',
                   help='Verbose log output')
    P.add_argument('-n', '--num_files', default=1, type=int,
                   help='Number of frames to generate')
    P.add_argument('-b', '--basepath', default='output/', type=str,
                   help='camera id')
    return P.parse_args()

# signal handler to trap the Ctrl-C and stop the script
def signal_handler(sig, frame):
    global ctrl_c

    print('You pressed Ctrl+C!\nStopping video capture ...')
    ctrl_c = True

# initialize the video feed
def video_init(video_path, verbose):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Failed to open video:', video_path)
        return -1
    
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('Frame size: %dx%d fps %d' % (h, w, fps))
    print('')

    return cap


# capture the video feed for 'duration' seconds to one file
def video_capture_one_file(num_files, cap, out_base, verbose):
    fprocessed = 0
    fwritten = 0
    ret = True
    while ret:
        t1 = time.time()
        # read next frame
        ret, frame = cap.read()
        if not ret: break
        
        if fprocessed % 100 == 0:
            fname = out_base + 'image-' + str(fprocessed) + '.jpg'
            cv2.imwrite(fname, frame)
            print('Written', fname)
            fwritten += 1
            if fwritten >= num_files: break

        fprocessed += 1

    return fprocessed

# global variables
ctrl_c = False

# main function
def main():
    #print(cv2.getBuildInformation())
    signal.signal(signal.SIGINT, signal_handler)
    args = arg_parser()

    video_path = 'data/lemons.mp4'
    print('\nOpening video path: %s' % video_path)

    out_base = args.basepath

    cap  = video_init(video_path, args.verbose)
    if cap == -1:
        print('Video init failed !!!')
        return
    
    print('Video initialized')
    video_capture_one_file(args.num_files, cap, out_base, args.verbose)

if __name__ == "__main__":
    main()
