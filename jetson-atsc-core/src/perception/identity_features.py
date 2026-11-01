"""Small CPU descriptors and a static-background displacement guard. No DNN."""
import math

import cv2
import numpy as np

from src.experiment.identity_contracts import FrameIdentityEvidence, IdentityDetection


class IdentityFeatureExtractor:
    def __init__(self):
        self.previous = self.previous_mask = None
        self.displaced = False

    def reset(self):
        self.previous = self.previous_mask = None
        self.displaced = False

    def extract(self, frame, tracks, project, in_region):
        height, width = frame.shape[:2]
        scale = min(1., 160/width)
        gray = cv2.cvtColor(cv2.resize(frame, (round(width*scale), round(height*scale))), cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        for track in tracks:
            x1, y1, x2, y2 = [round(x*scale) for x in track.bbox]
            mask[max(0, y1-3):min(mask.shape[0], y2+3), max(0, x1-3):min(mask.shape[1], x2+3)] = 0
        reason = None
        if self.previous is not None and self.previous.shape == gray.shape:
            common = cv2.bitwise_and(mask, self.previous_mask)
            points = cv2.goodFeaturesToTrack(self.previous, 60, .05, 8, mask=common)
            spread = np.ptp(points.reshape(-1, 2), axis=0) if points is not None else (0., 0.)
            if points is not None and len(points) >= 8 and spread[0] >= gray.shape[1]*.3 and spread[1] >= gray.shape[0]*.3:
                stationary = common.astype(np.float32)/255
                before = (self.previous.astype(np.float32)-float(self.previous.mean()))*stationary
                after = (gray.astype(np.float32)-float(gray.mean()))*stationary
                shift, response = cv2.phaseCorrelate(before, after)
                displacement_limit=max(1.5,3.*gray.shape[1]/320)
                if response > .25 and math.hypot(*shift) > displacement_limit:
                    self.displaced = True
                moved, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None,
                    winSize=(15, 15), maxLevel=2)
                valid = status.ravel().astype(bool)
                if valid.sum() >= 8:
                    shifts = (moved-points).reshape(-1, 2)[valid]
                    displacement = np.median(shifts, axis=0)
                    support = np.mean(np.linalg.norm(shifts-displacement, axis=1) < 2.)
                    if support >= .65 and np.linalg.norm(displacement) > displacement_limit:
                        self.displaced = True
                    elif support < .65:
                        reason = 'background_unobservable'
                else:
                    reason = 'background_unobservable'
            else:
                reason = 'background_unobservable'
        self.previous, self.previous_mask = gray, mask
        detections = []
        boxes=np.asarray([v.bbox for v in tracks],dtype=float).reshape(-1,4)
        if len(boxes):
            low=np.maximum(boxes[:,None,:2],boxes[None,:,:2])
            high=np.minimum(boxes[:,None,2:],boxes[None,:,2:])
            intersection=np.prod(np.maximum(0.,high-low),axis=2)
            np.fill_diagonal(intersection,0.)
            area=np.maximum(1e-9,np.prod(boxes[:,2:]-boxes[:,:2],axis=1))
        for index, track in enumerate(tracks):
            box = tuple(float(x) for x in track.bbox)
            if not all(math.isfinite(x) for x in box) or box[2] <= box[0] or box[3] <= box[1]:
                continue
            x1, y1 = max(0, math.ceil(box[0])), max(0, math.ceil(box[1]))
            x2, y2 = min(width, math.floor(box[2])), min(height, math.floor(box[3]))
            reliable = x2-x1 >= 8 and y2-y1 >= 8 and box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height
            descriptor = None
            if x2 > x1 and y2 > y1:
                crop = cv2.resize(frame[y1:y2, x1:x2], (16, 24))
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                visible=np.full((24,16),255,np.uint8)
                # In a fixed approach view, a nearer object's lower box can
                # obscure this vehicle. Use the still-visible pixels instead
                # of learning that foreground vehicle as its appearance.
                for other in np.flatnonzero((intersection[index]>0)&(boxes[:,3]>box[3])):
                    b=boxes[other]
                    left=max(0,min(16,math.floor((b[0]-x1)*16/(x2-x1))))
                    right=max(0,min(16,math.ceil((b[2]-x1)*16/(x2-x1))))
                    top=max(0,min(24,math.floor((b[1]-y1)*24/(y2-y1))))
                    bottom=max(0,min(24,math.ceil((b[3]-y1)*24/(y2-y1))))
                    visible[top:bottom,left:right]=0
                reliable=reliable and np.mean(visible>0)>=.4
                hist = cv2.calcHist([hsv], [0, 1, 2], visible, [8, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(8,4,4)
                marginal=np.concatenate((hist.sum(axis=(1,2)),hist.sum(axis=(0,2)),hist.sum(axis=(0,1))))
                if marginal.sum()>0:descriptor = tuple(float(x) for x in np.sqrt(marginal/marginal.sum()))
            point = tuple(float(x) for x in track.bottom_center)
            ground = tuple(float(x) for x in project(*point)) if project is not None else None
            if ground is not None and not all(math.isfinite(x) for x in ground):
                ground = None
            detections.append(IdentityDetection(box, point, track.class_name,
                int(track.track_id) if getattr(track, 'stable_id', False) else None,
                int(getattr(track, 'epoch', 0)), bool(getattr(track, 'stable_id', False)),
                ground, descriptor, reliable, bool(in_region(*point))))
        return FrameIdentityEvidence(tuple(detections), not self.displaced,
            'camera_displacement' if self.displaced else reason)
