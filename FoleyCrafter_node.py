# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import random
import sys
import cv2
import numpy as np
import torchaudio
from PIL import Image

import platform
import subprocess
import os.path as osp
import soundfile as sf
import torch
import torchvision
from huggingface_hub import snapshot_download, hf_hub_download
from moviepy.editor import AudioFileClip, VideoFileClip

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from .foleycrafter.models.onset import torch_utils
from .foleycrafter.models.time_detector.model import VideoOnsetNet
from .foleycrafter.pipelines.auffusion_pipeline import Generator, denormalize_spectrogram
from. foleycrafter.utils.util import build_foleycrafter, read_frames_with_moviepy
import folder_paths
from comfy.utils import common_upscale

vision_transform_list = [
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.CenterCrop((112, 112)),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
video_transform = torchvision.transforms.Compose(vision_transform_list)

MAX_SEED = np.iinfo(np.int32).max

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None and platform.system() in ['Linux', 'Darwin']:
    try:
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip()
            print(f"FFmpeg is installed at: {ffmpeg_path}")
        else:
            print("FFmpeg is not installed. Please download ffmpeg-static and export to FFMPEG_PATH.")
            print("For example: export FFMPEG_PATH=/you_path/ffmpeg-4.4-amd64-static")
    except Exception as e:
        pass

if ffmpeg_path is not None and ffmpeg_path not in os.getenv('PATH'):
    print("Adding FFMPEG_PATH to PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

device = ("cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

weigths_current_path = os.path.join(folder_paths.models_dir, "foleycrafter")
if not os.path.exists(weigths_current_path):
    os.makedirs(weigths_current_path)

weigths_semantic_current_path = os.path.join(weigths_current_path, "semantic")
if not os.path.exists(weigths_semantic_current_path):
    os.makedirs(weigths_semantic_current_path)

weigths_vocoder_current_path = os.path.join(weigths_current_path, "vocoder")
if not os.path.exists(weigths_vocoder_current_path):
    os.makedirs(weigths_vocoder_current_path)


def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image


def pil2narry(img):
    narry = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return narry


def narry_list(list_in):
    for i in range(len(list_in)):
        value = list_in[i]
        modified_value = pil2narry(value)
        list_in[i] = modified_value
    return list_in


def nomarl_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_pil(samples)
    return img_pil


def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


def download_weights(file_dir,repo_id,subfolder="",pt_name=""):
    if subfolder:
        file_path = os.path.join(file_dir,subfolder, pt_name)
        sub_dir=os.path.join(file_dir,subfolder)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        if not os.path.exists(file_path):
            pt_path = hf_hub_download(
                repo_id=repo_id,
                subfolder=subfolder,
                filename=pt_name,
                local_dir = file_dir,
            )
        else:
            pt_path=get_instance_path(file_path)
        return pt_path
    else:
        file_path = os.path.join(file_dir, pt_name)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if not os.path.exists(file_path):
            pt_path = hf_hub_download(
                repo_id=repo_id,
                filename=pt_name,
                local_dir=file_dir,
            )
        else:
            pt_path=get_instance_path(file_path)
        return pt_path


def build_models(repo,device,semantic_scale):
    # download ckpt
    if not repo:
        repo = snapshot_download("auffusion/auffusion-full-no-adapter")
    # pre models
    download_weights(weigths_current_path, "ymzhang319/FoleyCrafter", subfolder="semantic",
                         pt_name="semantic_adapter.bin")
    download_weights(weigths_current_path, "ymzhang319/FoleyCrafter", subfolder="vocoder",
                     pt_name="vocoder.pt")
    download_weights(weigths_current_path, "ymzhang319/FoleyCrafter", subfolder="vocoder",
                     pt_name="config.json")

    # ckpt path
    temporal_ckpt_path=download_weights(weigths_current_path, "ymzhang319/FoleyCrafter",
                     pt_name="temporal_adapter.ckpt")
    # load vocoder
    vocoder = Generator.from_pretrained(weigths_vocoder_current_path).to(device)

    # load time_detector
    time_detector_ckpt=download_weights(weigths_current_path, "ymzhang319/FoleyCrafter",
                     pt_name="timestamp_detector.pth.tar")

    time_detector = VideoOnsetNet(False)
    time_detector, _ = torch_utils.load_model(time_detector_ckpt, time_detector, device=device, strict=True)

    # load adapters
    pipe = build_foleycrafter(repo).to(device)
    ckpt = torch.load(temporal_ckpt_path)

    # load temporal adapter
    if "state_dict" in ckpt.keys():
        ckpt = ckpt["state_dict"]
    load_gligen_ckpt = {}
    for key, value in ckpt.items():
        if key.startswith("module."):
            load_gligen_ckpt[key[len("module.") :]] = value
        else:
            load_gligen_ckpt[key] = value
    m, u = pipe.controlnet.load_state_dict(load_gligen_ckpt, strict=False)
    print(f"### Control Net missing keys: {len(m)}; \n### unexpected keys: {len(u)};")

    # load semantic adapter
    pipe.load_ip_adapter(weigths_semantic_current_path, subfolder="", weight_name="semantic_adapter.bin", image_encoder_folder=None)
    pipe.set_ip_adapter_scale(semantic_scale)

    return pipe, vocoder, time_detector


class FoleyCrafter_LoadModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id":("STRING", {"default": "auffusion/auffusion-full-no-adapter"}),
                "ip_repo":("STRING", {"default": "h94/IP-Adapter"}),
                "semantic_scale":("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL","MODEL","MODEL","MODEL",)
    RETURN_NAMES = ("pipe","vocoder","time_detector","image_encoder",)
    FUNCTION = "main_loader"
    CATEGORY = "FoleyCrafter"

    def main_loader(self,repo_id,ip_repo,semantic_scale):
        pipe, vocoder, time_detector=build_models(repo_id, device, semantic_scale)
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(ip_repo, subfolder="models/image_encoder").to(device)
        return (pipe,vocoder, time_detector,image_encoder,)


class FoleyCrafter_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        input_path = folder_paths.get_input_directory()
        video_files = [f for f in os.listdir(input_path) if
                       os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ['webm', 'mp4', 'mkv','gif']]
        return {
            "required": {
                "pipe": ("MODEL",),
                "vocoder": ("MODEL",),
                "time_detector": ("MODEL",),
                "image_encoder": ("MODEL",),
                "video_files": (["none"] + video_files,),
                "prompt": ("STRING", {"multiline": True,"default": "1 girl"}),
                "negative_prompt": ("STRING", {"multiline": True,"default": "bad quality"}),
                "max_frame": ("INT", {"default": 150, "min": 64, "step": 1,"max": 500}),
                "seeds": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "controlnet_scale": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "sample_width": ("INT", {"default": 1024, "min": 128, "max": 1024, "step": 64, "display": "number"}),
                "sample_height": ("INT", {"default": 256, "min": 128, "max": 1024, "step": 64, "display": "number"}),
                "video_dubbing": ("BOOLEAN", {"default": False},), }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT")
    RETURN_NAMES = ("image", "audio", "frame_rate",)
    FUNCTION = "fc_main"
    CATEGORY = "FoleyCrafter"
    
    def run_inference(self,pipe, vocoder, time_detector,controlnet_scale,seeds,image_encoder,video_files,max_frame,prompt,negative_prompt,steps,width,height,video_dubbing):

        generator = torch.Generator(device=device)
        generator.manual_seed(seeds)
        image_processor = CLIPImageProcessor()

        source_video = cv2.VideoCapture(video_files)
        fps = source_video.get(cv2.CAP_PROP_FPS)
        v_width = int(source_video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的fps w h
        v_height = int(source_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_video.release()
        
        with torch.no_grad():
            print(f" >>> Begin Inference: {video_files} <<< ")
            frames, duration,frames_list = read_frames_with_moviepy(video_files, max_frame_nums=max_frame)
            
            time_frames = torch.FloatTensor(frames).permute(0, 3, 1, 2)
            time_frames = video_transform(time_frames)
            time_frames = {"frames": time_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)}
            preds = time_detector(time_frames)
            preds = torch.sigmoid(preds)
            
            # duration
            # import ipdb; ipdb.set_trace()
            time_condition = [
                -1 if preds[0][int(i / (1024 / 10 * duration) * 150)] < 0.5 else 1
                for i in range(int(1024 / 10 * duration))
            ]
            time_condition = time_condition + [-1] * (1024 - len(time_condition))
            # w -> b c h w
            time_condition = (
                torch.FloatTensor(time_condition)
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, 1, 256, 1)
                .to("cuda")
            )
            images = image_processor(images=frames, return_tensors="pt").to(device)
            image_embeddings = image_encoder(**images).image_embeds
            image_embeddings = torch.mean(image_embeddings, dim=0, keepdim=True).unsqueeze(0).unsqueeze(0)
            neg_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([neg_image_embeddings, image_embeddings], dim=1)
            
            sample = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_image_embeds=image_embeddings,
                image=time_condition,
                # audio_length_in_s=10,
                controlnet_conditioning_scale=controlnet_scale,
                num_inference_steps=steps,
                height=height,
                width=width,
                output_type="pt",
                generator=generator,
                # guidance_scale=0,
            )
            
            audio_img = sample.images[0]
            audio = denormalize_spectrogram(audio_img)
            audio = vocoder.inference(audio, lengths=160000)[0]
            
            waveform = audio[: int(duration * 16000)]
            # save
            rand_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
            save_path = osp.join(folder_paths.output_directory, f"{rand_file_prefix}.wav")
            sf.write(save_path, waveform, 16000)
            
            # output
            waveform, sample_rate = torchaudio.load(save_path)
            audio_output = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
            
            if video_dubbing:
                output_video_path = os.path.join(folder_paths.output_directory, f"{rand_file_prefix}_audio.mp4")
                video_clip = VideoFileClip(video_files)
                audio_clip = AudioFileClip(save_path)
                final_clip = video_clip.set_audio(audio_clip)
                final_clip.write_videofile(
                    output_video_path,
                    codec="libx264", audio_codec="aac")
                print(f"saving {output_video_path}")
                video_clip.reader.close()
                audio_clip.close()
                final_clip.reader.close()
                
            return frames_list, audio_output, fps,v_width,v_height
    
            
    def fc_main(self, pipe, vocoder, time_detector,image_encoder,video_files, prompt,negative_prompt, seeds,max_frame,controlnet_scale, steps,sample_width,sample_height, video_dubbing, ):
        if video_files != "none":
            video_files=osp.join(folder_paths.input_directory, video_files)
            images,audio,fps,v_width,v_height=self.run_inference( pipe, vocoder, time_detector,controlnet_scale,seeds,image_encoder,video_files,max_frame,prompt,negative_prompt,steps,sample_width,sample_height,video_dubbing)
        else:
            raise "need video file"
        frame_rate=float(fps)
        gen = narry_list(images)
        images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (v_height, v_width, 3)))))
        torch.cuda.empty_cache()
        
        return (images,audio,frame_rate,)



NODE_CLASS_MAPPINGS = {
    "FoleyCrafter_LoadModel":FoleyCrafter_LoadModel,
    "FoleyCrafter_Sampler": FoleyCrafter_Sampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FoleyCrafter_LoadModel":"FoleyCrafter_LoadModel",
    "FoleyCrafter_Sampler": "FoleyCrafter_Sampler",
}
