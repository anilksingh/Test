# LatentSync/eval/syncnet/syncnet_eval.py

import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from .syncnet import S
from shutil import rmtree

# -- Add Mediapipe for face detection
try:
    import mediapipe as mp
except ImportError:
    print("Installing mediapipe...")
    import sys
    !{sys.executable} -m pip install mediapipe
    import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

def calc_pdist(feat1, feat2, vshift=10):
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    dists = []
    for i in range(0, len(feat1)):
        dists.append(
            torch.nn.functional.pairwise_distance(
                feat1[[i], :].repeat(win_size, 1), 
                feat2p[i : i + win_size, :]
            )
        )
    return dists

class SyncNetEval(torch.nn.Module):
    def __init__(self, dropout=0, num_layers_in_fc_layers=1024, device="cpu"):
        super().__init__()
        self.__S__ = S(num_layers_in_fc_layers=num_layers_in_fc_layers).to(device)
        self.device = device

    def evaluate(self, video_path, temp_dir="temp", batch_size=20, vshift=15, skip_until_face=False):
        """
        Evaluate sync offset between audio & video. 
        If skip_until_face=True, skip frames (jpgs) that do not contain a face until we find one.
        """
        self.__S__.eval()

        if os.path.exists(temp_dir):
            rmtree(temp_dir)
        os.makedirs(temp_dir)

        # Extract frames as JPEG and audio
        command = f"ffmpeg -loglevel error -nostdin -y -i {video_path} -f image2 {os.path.join(temp_dir, '%06d.jpg')}"
        subprocess.call(command, shell=True, stdout=None)

        command = f"ffmpeg -loglevel error -nostdin -y -i {video_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {os.path.join(temp_dir, 'audio.wav')}"
        subprocess.call(command, shell=True, stdout=None)

        # Collect frames
        flist = glob.glob(os.path.join(temp_dir, "*.jpg"))
        flist.sort()

        images = []
        face_found = False

        # Initialize mediapipe
        face_detector = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        for fname in flist:
            img_input = cv2.imread(fname)
            # Resize to 224x224
            img_input = cv2.resize(img_input, (224, 224)) 
            
            if skip_until_face and not face_found:
                # Check if this frame has a face
                frame_rgb = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
                results = face_detector.process(frame_rgb)
                if results.detections and len(results.detections) > 0:
                    # Found a face
                    face_found = True
                    images.append(img_input)
                else:
                    # Skip
                    continue
            else:
                # Either already found a face or skip_until_face is False
                images.append(img_input)

        # Convert images to correct shape
        if len(images) == 0:
            print(f"No frames with face found in: {video_path}. Exiting early.")
            rmtree(temp_dir)
            return None, None, None

        im = numpy.stack(images, axis=3)  # => (H, W, C, num_frames)
        im = numpy.expand_dims(im, axis=0)  # => (1, H, W, C, num_frames)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))  # => (1, num_frames, C=3, H=224, W=224)

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # Load audio
        sample_rate, audio = wavfile.read(os.path.join(temp_dir, "audio.wav"))
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])

        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        min_length = min(len(images), math.floor(len(audio) / 640))
        lastframe = min_length - 5

        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0, lastframe, batch_size):
            im_batch = [
                imtv[:, :, vframe : vframe + 5, :, :]
                for vframe in range(i, min(lastframe, i + batch_size))
            ]
            im_in = torch.cat(im_batch, 0).to(self.device)
            im_out = self.__S__.forward_lip(im_in)
            im_feat.append(im_out.data.cpu())

            cc_batch = [
                cct[:, :, :, vframe * 4 : vframe * 4 + 20]
                for vframe in range(i, min(lastframe, i + batch_size))
            ]
            cc_in = torch.cat(cc_batch, 0).to(self.device)
            cc_out = self.__S__.forward_aud(cc_in)
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
        mean_dists = torch.mean(torch.stack(dists, 1), 1)

        min_dist, minidx = torch.min(mean_dists, 0)

        av_offset = vshift - minidx
        conf = torch.median(mean_dists) - min_dist

        rmtree(temp_dir)

        return av_offset.item(), min_dist.item(), conf.item()

    def extract_feature(self, opt, videofile, skip_until_face=False):
        """
        If skip_until_face=True, skip all frames until the first face is detected,
        then keep all subsequent frames.
        """

        self.__S__.eval()
        cap = cv2.VideoCapture(videofile)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {videofile}")

        # Use mediapipe for face detection
        face_detector = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )

        images = []
        face_found = False

        while True:
            ret, frame = cap.read()
            if not ret:
                # End of video
                break

            if skip_until_face and not face_found:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detector.process(frame_rgb)
                if results.detections and len(results.detections) > 0:
                    face_found = True
                    images.append(frame)
                else:
                    continue
            else:
                images.append(frame)

        cap.release()

        if len(images) == 0:
            print(f"No face found in any frame for video: {videofile}")
            return None

        im = numpy.stack(images, axis=3)  # => (H, W, C, num_frames)
        im = numpy.expand_dims(im, axis=0)  # => (1, H, W, C, num_frames)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))  # => (1, num_frames, C=3, H, W)

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        lastframe = len(images) - 4
        im_feat = []

        tS = time.time()
        for i in range(0, lastframe, opt.batch_size):
            im_batch = [
                imtv[:, :, vframe : vframe + 5, :, :]
                for vframe in range(i, min(lastframe, i + opt.batch_size))
            ]
            if len(im_batch) == 0:
                break

            im_in = torch.cat(im_batch, 0).to(self.device)
            im_out = self.__S__.forward_lipfeat(im_in)
            im_feat.append(im_out.data.cpu())

        if len(im_feat) > 0:
            im_feat = torch.cat(im_feat, 0)
        else:
            im_feat = None

        print("Compute time %.3f sec." % (time.time() - tS))
        return im_feat

    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage, weights_only=True)
        self_state = self.__S__.state_dict()

        for name, param in loaded_state.items():
            self_state[name].copy_(param)
