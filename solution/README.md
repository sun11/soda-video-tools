## Modelscope-Sora Challenge Soda Solution

### 安装

1. `pip install -r requirements.txt`
2. tarsier, dover, Practical-RIFE 若缺乏依赖库，到文件夹中安装相应依赖
3. 下载模型权重文件

    tarsier: https://huggingface.co/omni-research/Tarsier-7b
    
    将Tarsier-7b模型文件夹放入`solution/tarsier/weights`

    dover: https://huggingface.co/teowu/DOVER/resolve/main/DOVER_plus_plus.pth
    
    放入`dover/pretrained_weights`文件夹

### 数据处理流程

#### Stage1 Video Split

`bash stage1_video_cut.sh`

使用pyscenedetect切分视频场景

#### Stage2 Coarse Filter

`bash stage2_coarse_filter.sh`

使用分辨率和文本检测过滤视频，并计算视频运动得分motion_score，质量评分dover_score，计算综合得分coarse_score，粗略过滤视频

#### Stage3 Video Caption

`bash stage3_video_caption_tarsier.sh`

使用tarsier模型对视频做文本打标，并计算输出文本结果的置信度作为caption_confidence

#### Stage4 Convert FPS

使用RIFE模型对视频插帧，并转换帧率到24fps

#### Stage5 Refine Filter

综合motion_score，dover_score和caption_confidence加权计算总分，对总分排序并筛选视频