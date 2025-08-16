
import argparse
import os
import torch
import numpy as np
import cv2
from models.experimental import attempt_load
from utils.general import check_img_size, scale_coords
from ultralytics.utils.ops import non_max_suppression
from utils.datasets import letterbox
from utils.torch_utils import select_device
import warnings

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--weights', default=r"C:/Users/86178/Desktop/multispectral-yolov11-transformer/runs/train/exp12/weights/best.pt", type=str,
                    help='Path to model weights file.')
parser.add_argument('--video_rgb', default=r"C:/Users/86178/Desktop/dataset/rgb_to_video/output.mp4", type=str, help='Path to RGB video file.')
parser.add_argument('--video_ir', default=r"C:/Users/86178/Desktop/dataset/ir_to_video/output.mp4", type=str, help='Path to IR video file.')
parser.add_argument('--conf_thre', type=int, default=0.4, help='Confidence threshold for detections.')
parser.add_argument('--iou_thre', type=int, default=0.5, help='IoU threshold for NMS.')
parser.add_argument('--save_video', default=r"./results", type=str, help='Directory to save result videos.')
parser.add_argument('--vis', default=True, action='store_true', help='Visualize frames with detections.')
parser.add_argument('--device', type=str, default="0", help='Device: "0" for GPU, "cpu" for CPU.')
parser.add_argument('--imgsz', type=int, default=640, help='Input image size for inference.')
parser.add_argument('--merge_nms', default=False, action='store_true', help='Merge detections across classes.')
opt = parser.parse_args()


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


class Detector:
    def __init__(self, device, model_path=r'./best.pt', imgsz=640, merge_nms=False):

        self.device = device
        self.model = attempt_load(model_path, map_location=device)  # load FP32 model
        self.names = self.model.names
        self.stride = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.merge_nms = merge_nms

    def process_image(self, image, imgsz, stride, device):
        img = letterbox(image, imgsz, stride=stride)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        im = img.float()  # uint8 to fp16/32
        im /= 255.0
        im = im[None]
        return im

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray, image_ir: np.ndarray, conf, iou):
        img_vis = image_rgb.copy()
        img_vis_ir = image_ir.copy()
        img_rgb = self.process_image(image_rgb, self.imgsz, self.stride, device)
        img_ir = self.process_image(image_ir, self.imgsz, self.stride, device)

        # inference
        pred = self.model(img_rgb, img_ir)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=conf, iou_thres=iou, classes=None,
                                   agnostic=self.merge_nms)

        for i, det in enumerate(pred):  # detections per image
            det[:, :4] = scale_coords(img_rgb.shape[2:], det[:, :4], image_rgb.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xmin, ymin, xmax, ymax = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                cv2.rectangle(img_vis, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                              get_color(int(cls) + 2), 2)
                cv2.putText(img_vis, f"{self.names[int(cls)]} {conf:.1f}", (int(xmin), int(ymin - 5)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, get_color(int(cls) + 2), thickness=2)
                cv2.rectangle(img_vis_ir, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                              get_color(int(cls) + 2), 2)
                cv2.putText(img_vis_ir, f"{self.names[int(cls)]} {conf:.1f}", (int(xmin), int(ymin - 5)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, get_color(int(cls) + 2), thickness=2)
        return img_vis, img_vis_ir


if __name__ == '__main__':
    print("Start to process viedo")
    device = select_device(opt.device)
    print(device)
    detector = Detector(device=device, model_path=opt.weights, imgsz=opt.imgsz, merge_nms=opt.merge_nms)

 
    cap_rgb = cv2.VideoCapture(opt.video_rgb)
    cap_ir = cv2.VideoCapture(opt.video_ir)


    fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))


    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_rgb = cv2.VideoWriter(os.path.join(opt.save_video, 'output_rgb.mp4'), fourcc, fps, (width, height))
    out_ir = cv2.VideoWriter(os.path.join(opt.save_video, 'output_ir.mp4'), fourcc, fps, (width, height))
    out_combined = cv2.VideoWriter(os.path.join(opt.save_video, 'output.mp4'), fourcc, fps, (1300, 400))
    while True:
        ret_rgb, frame_rgb = cap_rgb.read()
        ret_ir, frame_ir = cap_ir.read()

        if not ret_rgb or not ret_ir:
            break
        
        #add
        height1, width1 = frame_rgb.shape[:2]
        height2, width2 = frame_ir.shape[:2]
        target_size = (min(width1, width2), min(height1, height2))
        frame_rgb = cv2.resize(frame_rgb, target_size)
        frame_ir = cv2.resize(frame_ir,target_size)
        #add-end

        img_vis, img_vis_ir = detector(frame_rgb, frame_ir, opt.conf_thre, opt.iou_thre)
        # 
        img_combined = cv2.hconcat([img_vis, img_vis_ir])
        img_combined = cv2.resize(img_combined, (1300, 400))

        # 
        out_rgb.write(img_vis)
        out_ir.write(img_vis_ir)
        out_combined.write(img_combined)
        if opt.vis:
            cv2.imshow('Combined Video', img_combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap_rgb.release()
    cap_ir.release()
    out_rgb.release()
    out_ir.release()
    cv2.destroyAllWindows()
