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
    P.add_argument('-d', '--duration', default=1, type=int,
                   help='Duration of each output file in seconds')
    P.add_argument('-n', '--num_files', default=0, type=int,
                   help='Number of files to generate (0 for infinite)')
    P.add_argument('-b', '--basepath', default='output/', type=str,
                   help='camera id')
    P.add_argument('-c', '--camera', default='camera-lab', type=str,
                   help='camera id')
    return P.parse_args()

# signal handler to trap the Ctrl-C and stop the script
def signal_handler(sig, frame):
    global ctrl_c

    print('You pressed Ctrl+C!\nStopping video capture ...')
    ctrl_c = True

# draw text alongwith a rectangle in the background
def draw_text(img, text, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), font_scale=1.5,
        font_thickness=2, text_color=(0, 255, 0), text_color_bg=(0, 0, 0), padding=5):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w+2*padding, y + text_h+2*padding), text_color_bg, -1)
    cv2.putText(img, text, (x+padding, y + text_h + padding), font, font_scale, text_color, font_thickness)

    return text_size

def get_framenum_msec(cap, fps):
    ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if True:
        fn = round(ms * fps / 1000)
    else:
        fn = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    return fn, ms

# initialize the video feed
def video_init(video_path, duration, verbose):
    if duration > 60 or (60 % duration != 0):
        print("Duration (%d) must be a factor of 60" % duration)
        return -1
    
    open_start = time.time()
    cap = cv2.VideoCapture(video_path)
    open_end = time.time()
    print('Time taken to open: %4.2fs' % (open_end-open_start))

    if not cap.isOpened():
        print('Failed to open video:', video_path)
        return -1
    
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print('Frame size: %dx%d fps %d' % (h, w, fps))
    print('')

    print('Buffersize:', cap.get(cv2.CAP_PROP_BUFFERSIZE))
    print('Set buffersize to 1:', cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))

    # Flush the buffer and align to duration
    print('\nStarting to flush ...')
    flush_duration = 2 #seconds
    num_frames = 0
    prev_time = dt.datetime.now()
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret: break
        num_frames += 1
        
        now = dt.datetime.now()
        milisec = now.microsecond/1000
        delta = (milisec - prev_time.microsecond/1000 + 1000) % 1000
        
        rem = now.second % duration
        fn, ms = get_framenum_msec(cap, fps)
        if verbose: print('[%2d] %3d %3d %2d %3d %d' % (num_frames, milisec, delta, rem, fn, ms))

        if num_frames > fps*flush_duration and milisec > 100 and milisec < 300 and rem == 0: break
        prev_time = now

    print('%d frames flushed, offset %d ms, rem %d' % (num_frames, milisec, rem))
    return cap

# capture the video feed for 'duration' seconds to one file
def video_capture_one_file(cap, out_base, duration, verbose):
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    title = 'Viewer'

    now = dt.datetime.now()
    out_path = out_base + now.strftime("%Y-%m-%d/") 
    os.makedirs(out_path, exist_ok=True)

    out_path += now.strftime("%H-%M-%S-") + now.strftime('%f')[:3] + '_' + str(duration) + '.mp4'
    print('Output video path: %s fps %d\n' % (out_path, fps))

    # setup video writer
    #fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1') #.avi
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_path, fourcc, fps, (w,h))
    tot_frames = duration * fps

    fn_start, ms_start = get_framenum_msec(cap, fps)
    ms_prev = ms_start
    t_start = t_prev = time.time()
    fprocessed = 0
    ret = True
    while ret:
        t1 = time.time()
        # read next frame
        ret, frame = cap.read()
        if not ret: break
        fprocessed += 1
        fn,ms = get_framenum_msec(cap, fps)

        t2 = time.time()
        draw_text(frame, str(fprocessed), pos=(50,50))
        cv2.imshow(title, frame)

        t3 = time.time()
    
        video_writer.write(frame)

        t4 = time.time()

        if verbose: print('[%4d] Total %3d ms: breakup [%3d %3d %3d] %3d %3d %d' % (
            fprocessed,
            int((t4-t1)*1000), 
            int((t2-t1)*1000), 
            int((t3-t2)*1000), 
            int((t4-t3)*1000),
            fn, ms, ms-ms_prev))
        ms_prev = ms

        if fprocessed % fps == 0:
            t_end = time.time()
            time_taken = t_end - t_prev
            tpf = time_taken * 1000 / fps
            print('[%d] Total time: %4.2fs TPF %dms' % (fps, time_taken, tpf))
            t_prev = t_end

        if fprocessed == tot_frames:
            break

    t_end = time.time()
    time_taken = t_end - t_start
    tpf = time_taken * 1000 / fprocessed
    tot_fns = fn - fn_start
    tot_ms = ms - ms_start
    if (tot_ms != duration*1000 or tot_fns != tot_frames):
        print('\n!!!!!!! Total MSEC %d, Total Frames %d !!!!!!\n' % (tot_ms, tot_fns))
        delta = abs(duration * 1000 - tot_ms)
        if delta > 500: sys.exit(1)
    print('[%d frames] Total time: %4.2fs TPF %dms' % (fprocessed, time_taken, tpf))

    return fprocessed

# start the video capture of the feed to a sequence of files
# if iterations is 0, the script stops only when Ctrl-C is pressed
def video_capture(num_files, cap, out_base, duration, verbose):
    global ctrl_c

    iter = 0
    while True:
        video_capture_one_file(cap, out_base, duration, verbose)
        iter += 1
        if num_files > 0 and iter >= num_files: break
        if ctrl_c: break

    print('\n%d files created\n' % iter)

# global variables
ctrl_c = False

# main function
def main():
    #print(cv2.getBuildInformation())
    signal.signal(signal.SIGINT, signal_handler)
    args = arg_parser()

    #video_path = 0 # macbook webcam
    video_path = 'rtsp://admin:Ramen%40123@172.16.20.29/profile2/media.smp' # blr lab hanhwa camera
    #video_path = 'rtsp://admin:Ramen1234@192.168.68.59/profile2/media.smp' # sandesh hanhwa camera
    #video_path = 'rtsp://admin:Ramen1234@192.168.68.55/Streaming/channels/101' # sandesh hikvision camera
    print('\nOpening video path: %s' % video_path)

    out_base = args.basepath + args.camera + '/'

    cap  = video_init(video_path, args.duration, args.verbose)
    if cap == -1:
        print('Video init failed !!!')
        return
    
    print('Video initialized')
    video_capture(args.num_files, cap, out_base, args.duration, args.verbose)

if __name__ == "__main__":
    main()
