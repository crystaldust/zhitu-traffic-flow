import cv2

def extract_frames(filepath, gap=1):
    cap = cv2.VideoCapture(filepath)
    count = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print(frame_count)
    frames = []
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        frames.append(frame)
        # Write the results back to output location.
        count = count + gap
        # If there are no more frames left
        if (count > (frame_count-1)):
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            break

    print('we got {} frames from {}'.format(len(frames), filepath))
    return frames



