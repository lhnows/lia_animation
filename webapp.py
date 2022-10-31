import torch
import torch.nn as nn
from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from alfred import device
# device = torch.device('cpu')
import gradio as gr
import uuid
import cv2

def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def vid_preprocessing(vid_path,size):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit='sec')
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
    vid = torch.nn.functional.interpolate(vid, size=(vid.shape[2], size, size), mode='nearest')
    fps = vid_dict[2]['video_fps']
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(save_path, vid[0], fps=fps)


class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        self.args = args

        if args.model == 'vox':
            model_path = 'weights/vox.pt'
        elif args.model == 'taichi':
            model_path = 'weights/taichi.pt'
        elif args.model == 'ted':
            model_path = 'weights/ted.pt'
        else:
            raise NotImplementedError

        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).to(device)
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        print('==> loading data')
        self.save_path = os.path.join(args.save_folder + '/%s' % args.model,
                                      Path(args.source_path).stem + '_' + Path(args.driving_path).stem + '.mp4')
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.img_source = img_preprocessing(args.source_path, args.size).to(device)
        self.vid_target, self.fps = vid_preprocessing(args.driving_path,args.size)
        self.vid_target = self.vid_target.to(device)

    def run(self):

        print('==> running')
        with torch.no_grad():

            vid_target_recon = []

            if self.args.model == 'ted':
                h_start = None
            else:
                h_start = self.gen.enc.enc_motion(self.vid_target[:, 0, :, :, :])

            for i in tqdm(range(self.vid_target.size(1))):
                img_target = self.vid_target[:, i, :, :, :]
                img_recon = self.gen(self.img_source, img_target, h_start)
                vid_target_recon.append(img_recon.unsqueeze(2))

            vid_target_recon = torch.cat(vid_target_recon, dim=2)
            save_video(vid_target_recon, self.save_path, self.fps)
            return self.save_path

def video_identity(video_path,image):
    print('reieve:',video_path)
    sid = str(uuid.uuid4())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_path = 'data/'+sid+'.jpg'
    cv2.imwrite(image_path,image)
    print('image:',image_path)
    print(video_path)
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2560)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='vox')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--source_path", type=str, default=image_path)
    parser.add_argument("--driving_path", type=str, default=video_path)
    parser.add_argument("--save_folder", type=str, default='./res')
    args = parser.parse_args()

    # demo
    demo = Demo(args)
    ret_path = demo.run()
    print('ret:',ret_path)
    return ret_path

if __name__ == '__main__':
    # training params
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--size", type=int, default=256)
    # parser.add_argument("--channel_multiplier", type=int, default=1)
    # parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='')
    # parser.add_argument("--latent_dim_style", type=int, default=512)
    # parser.add_argument("--latent_dim_motion", type=int, default=20)
    # parser.add_argument("--source_path", type=str, default='')
    # parser.add_argument("--driving_path", type=str, default='')
    # parser.add_argument("--save_folder", type=str, default='./res')
    # args = parser.parse_args()

    # # demo
    # demo = Demo(args)
    # demo.run()





    # interface = gr.Interface(video_identity, 
    #                 gr.Video(), 
    #                 "playable_video", 
    #                 examples=[
    #                     os.path.join(os.path.dirname(__file__), 
    #                                  "video/video_sample.mp4")], 
    #                 cache_examples=True)
    
    interface = gr.Interface(video_identity,
                             inputs=["playable_video","image"], 
                             outputs="playable_video",
                             title="数字人demo",
                             description="上传目标视频和数字人照片!")
    interface.launch(server_name = '0.0.0.0',server_port = 7861,enable_queue = True)

    # interface = gr.Interface(fn=infer,
    #                          inputs="image", 
    #                          outputs=["image","text"],
    #                          title="侧位片分析",
    #                          description="上传图像稍等片刻即可!")
    # interface.launch(server_name = '0.0.0.0',server_port = 7861,enable_queue = True)
