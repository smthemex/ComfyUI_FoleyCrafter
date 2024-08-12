# ComfyUI_FoleyCrafter
FoleyCrafter is a video-to-audio generation framework which can produce realistic sound effects semantically relevant and synchronized with videos.

FoleyCrafter  From: [FoleyCrafter](https://github.com/open-mmlab/FoleyCrafter)

Update
---
2024/08/22  
--修复clip关闭的错误；节点改成字典，避免太多线了，运行离线模型失败的，请看3.2和3.3内容；   
--Fix the error of closing clip; Change the node to a dictionary to avoid too many lines. If the offline model fails to run, please refer to sections 3.2 and 3.3；  

1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_FoleyCrafter.git
```  
  
2.requirements  
----
按理是不需要装特别的库，如果还是库缺少，请单独安装。或者打开no need requirements.txt，查看缺失的库是否在里面。  
秋叶包因为是沙箱模式，所以把缺失的库安装在系统的python库里，官方的便携包，用python -m pip install 库名。  
If the module is missing, please open "no need requirements.txt" , pip install missing module.    

可能会出现的问题，开启video_dubbing 是合成音视频，如果报错，打开控制台CMD，按以下步骤操作：  
Possible issues may arise. Enabling "video_fubbing" produces synthesized audio and video. If an error occurs, open the CMD console and follow these steps:   

```
python -m pip uninstall moviepy decorator
python -m pip install moviepy decorator
```

3 Need  models
----
3.1   
"ymzhang319/FoleyCrafter"  [link](https://huggingface.co/ymzhang319/FoleyCrafter/tree/main) ,
全部下载，并按如下结构存放在ComfyUI/models/foleycrafter 文件夹下，联外网会自动下载：    
Download all and store them in the "ComfyUI/models/foleycrafter folder" according to the following structure，online will auto download:      

```
└── ComfyUI/models/foleycrafter/
    ├── semantic
    │   ├── semantic_adapter.bin
    ├── vocoder
    │   ├── vocoder.pt
    │   ├── config.json
    ├── temporal_adapter.ckpt
    │   │
    └── timestamp_detector.pth.tar
```
3.2  
online,fill "h94/IP-Adapter" [link](https://huggingface.co/h94/IP-Adapter/tree/main/models/image_encoder),
离线使用时，部分下载，文件结构如下,联外网会自动下载，if offline, Partial download, file structure as follows，online will auto download：
离线使用时，只需要填写：any_path 。。。。 When used offline, only need to fill in： any_path；   
虽然是随意地址，但是模型存放路径必须是models/image_encoder/（模型文件）；
```
└── any_path
    ├── models/image_encoder
    │   ├── model.safetensors
    │   ├── config.json
```
3.3  
"auffusion/auffusion-full-no-adapter" [link](https://huggingface.co/auffusion/auffusion-full-no-adapter/tree/main),
离线使用时，部分下载，文件结构如下,联外网会自动下载，if offline, Partial download, file structure as follows，online will auto download：   
离线使用时，只需要填写：any_path/auffusion/auffusion-full-no-adapter 。。。When used offline, only need to fill in：any_path/auffusion/auffusion-full-no-adapter；   
```
├── any_path/auffusion/auffusion-full-no-adapter
|      ├──model_index.json
|      ├──vae
|          ├── config.json
|          ├── diffusion_pytorch_model.bin
|      ├──unet
|          ├── config.json
|          ├── diffusion_pytorch_model.bin
|      ├──tokenizer
|          ├── merges.txt
|          ├── special_tokens_map.json
|          ├── tokenizer_config.json
|          ├── vocab.json
|      ├── text_encoder
|          ├── config.json
|          ├── pytorch_model.bin  
|      ├── scheduler
|          ├── scheduler_config.json
|      ├──feature_extractor
|          ├──preprocessor_config.json
|      ├──vocoder
|          ├──config.json
|          ├──vocoder.pt
```
4 Example
----
video_dubbing using prompt an uprompt (Latest version)        
![](https://github.com/smthemex/ComfyUI_FoleyCrafter/blob/main/example/new.png)


5 Function Description of Nodes  
---
--semantic_scale :ip adatpter scale 调整音频的跟视频的相似度;   
--max_frame ：audio length  音频长度，还未测试极限，目前设置500;   
--controlnet_scale： another audio sacle  未测试;  
--sample_width/sample_width： weights type, don't change it 模型尺寸，不要动，这个跟视频长宽无关;   
--video_dubbing: save  or not（if using example） 是否用内置的音视频合成，如果用示例的VH合成，可以关闭。  

6 My ComfyUI node list：
-----

1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     
2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      
3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)   
4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       
5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  
6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)   
7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)        
8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)   
9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)           
10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    
11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)   
12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker)      
13、ComfyUI_Streamv2v_Plus node:[ComfyUI_Streamv2v_Plus](https://github.com/smthemex/ComfyUI_Streamv2v_Plus)   
14、ComfyUI_MS_Diffusion node:[ComfyUI_MS_Diffusion](https://github.com/smthemex/ComfyUI_MS_Diffusion)   
15、ComfyUI_AnyDoor node: [ComfyUI_AnyDoor](https://github.com/smthemex/ComfyUI_AnyDoor)  
16、ComfyUI_Stable_Makeup node: [ComfyUI_Stable_Makeup](https://github.com/smthemex/ComfyUI_Stable_Makeup)  
17、ComfyUI_EchoMimic node:  [ComfyUI_EchoMimic](https://github.com/smthemex/ComfyUI_EchoMimic)   
18、ComfyUI_FollowYourEmoji node: [ComfyUI_FollowYourEmoji](https://github.com/smthemex/ComfyUI_FollowYourEmoji)   
19、ComfyUI_Diffree node: [ComfyUI_Diffree](https://github.com/smthemex/ComfyUI_Diffree)    
20、ComfyUI_FoleyCrafter node: [ComfyUI_FoleyCrafter](https://github.com/smthemex/ComfyUI_FoleyCrafter)

7 Citation
------
"open-mmlab/FoleyCrafter"
```
@misc{zhang2024pia,
  title={FoleyCrafter: Bring Silent Videos to Life with Lifelike and Synchronized Sounds},
  author={Yiming Zhang, Yicheng Gu, Yanhong Zeng, Zhening Xing, Yuancheng Wang, Zhizheng Wu, Kai Chen},
  year={2024},
  eprint={2407.01494},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

