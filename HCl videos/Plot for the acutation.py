
import os
import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

# ========== Global Settings ==========
state = 0
p1, p2 = None, None
p3, p4 = None, None
data = {}

# ========== Helper Functions ==========
def disp_objects(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresh, contours, -1, (0,255,0), 1)
    cv2.imshow('Frame', thresh)
    area = cv2.contourArea(contours[0]) 
    for c in contours:
        if cv2.contourArea(c) < 100: continue
        orig = thresh.copy()
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        print(box)

def get_size(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 220, 255, 0)

    if abs(p3[1]-p4[1]) < abs(p3[0]-p4[0]):
        y0 = int(((p3[1]-p2[1]) + (p4[1]-p2[1])) / 2)
        x0 = int((p4[0]-p3[0])/2)
        row = thresh[y0]
        x1, x2 = x0, x0
        while x1 > 0 and row[x1] == 0: x1 -= 1
        while x2 < len(row)-1 and row[x2] == 0: x2 += 1
        return (x2 - x1)
    else:
        x0 = abs(int(((p3[0]-p1[0]) + (p4[0]-p1[0])) / 2))
        y0 = abs(int((p4[1]-p3[1]) / 2))
        col = thresh[:, x0]
        y1, y2 = y0, y0
        while y1 > 0 and col[y1] == 0: y1 -= 1
        while y2 < len(col)-1 and col[y2] == 0: y2 += 1
        return abs(y2 - y1)

def deflectionpixels(fname, roi=None):
    global state, p1, p2, p3, p4
    cap = cv2.VideoCapture(fname)
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, userdata):
        global state, p1, p2, p3, p4
        if event == cv2.EVENT_LBUTTONUP:
            if state == 0: p1 = (x, y); state += 1
            elif state == 1: p2 = (x, y); state += 1
            elif state == 2: p3 = (x, y); state += 1
            elif state == 3: p4 = (x, y); state += 1
        if event == cv2.EVENT_RBUTTONUP:
            p1, p2, p3, p4 = None, None, None, None
            state = 0

    cv2.setMouseCallback('Frame', on_mouse)

    if roi is None:
        i = 0
        while cap.isOpened():
            i += 1
            ret, frame = cap.read()
            if i < 100: continue
            if not ret: break
            if state > 3:
                print("Selected ROI:", p1, p2, p3, p4)
                state = 0
                break
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1000) == ord('q'): break
    else:
        p1, p2, p3, p4 = roi
    cv2.destroyAllWindows()

    dat = [[], []]
    frameno = 10
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameno)
    while cap.isOpened():
        frameno += 1
        ret, frame = cap.read()
        if not ret: break
        d = get_size(frame[p2[1]:p1[1]+1, p1[0]:p2[0]+1, :])
        dat[0].append(frameno)
        dat[1].append(d)

    cap.release()
    return dat

def gen_actuation_plots(data, fps=300):
    fig, axs = plt.subplots(1, 1, layout='constrained')
    fig.set_size_inches(6, 4)
    fig.set_dpi(300)

    shiftx = [123, 22, 0, 75]
    shifty = [0, 50, 20, 80]
    i = 0
    for v in data.keys():
        # Convert frame number to milliseconds
        times_ms = [(x - shiftx[i]) * 1000 / fps for x in data[v][0]]

        ynew = savgol_filter(data[v][1], 10, 3)
        ynew = np.max(ynew) - ynew + 1
        ynew = [x + shifty[i] for x in ynew]

        axs.plot(times_ms, ynew, label=os.path.basename(v)[:25], alpha=0.5, lw=1.5)
        i += 1

    axs.set_xlabel('Time (ms)')
    axs.set_ylabel('Pixel Intensity')
    axs.grid(True)
    axs.legend()
    plt.savefig("pixel_time_ms.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def collect_data():
    from vidnames import vids
    global data
    for v in vids['10k']:
        print(f"\nSelect ROI for: {os.path.basename(v[0])}")
        data[v[0]] = deflectionpixels(v[0], roi=None)

# Run the full process
collect_data()
gen_actuation_plots(data)
