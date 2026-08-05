# 3分钟演示视频剪辑指南 | Video Editing Guide

## 快速版：剪映一键搞定（推荐）

### 1. 准备素材
- 主素材：`assets/demo_policy.mp4` (6.1MB, 2分47秒)
- 预览素材：`assets/preview.gif` (28KB)
- 字幕文件：`demo_subtitle.srt` (已生成)

### 2. 剪映导入
1. 打开剪映（专业版/移动版均可）
2. "导入素材" → 选择 `demo_policy.mp4` 和 `preview.gif`
3. 拖拽到时间轴

### 3. 时间轴布局（3分钟结构）

```
0:00-0:15  [preview.gif循环] + 开场文字
0:15-0:30  问题说明文字卡片
0:30-1:30  demo_policy.mp4 精华片段（取第10-70秒，共60秒）
1:30-2:00  技术说明文字卡片
2:00-2:30  结果展示文字卡片
2:30-3:00  下一步文字卡片 + GitHub信息
```

### 4. 添加文字（复制粘贴即用）

#### 0:00-0:15 开场
```
主标题（大字，粗体）：
AMD Physical AI Hackathon
Quadruped Locomotion RL

副标题（中字）：
训练四足机器人按速度指令行走
Training Go1 to Follow Velocity Commands
```
配preview.gif循环播放作为背景

---

#### 0:15-0:30 问题
```
标题：Challenge 挑战

内容：
Train Unitree Go1 to walk under velocity commands
训练宇树Go1按速度指令行走

Goal 目标:
✓ Walk forward/backward 前进/后退
✓ Turn left/right 左转/右转
✓ Sidestep left/right 左移/右移
✓ Complex maneuvers 复杂机动

All controlled by ONE neural network
同一个神经网络控制所有动作
```

---

#### 0:30-1:30 演示（60秒）
使用 demo_policy.mp4 的第10-70秒（最精彩部分）

**左上角overlay文字（小字，持续显示）：**
```
Trained PPO Policy
训练的PPO策略
Real MuJoCo Physics
真实物理仿真
```

**分段字幕（每10秒切换）：**
- 0:30-0:40: "Stand 站立"
- 0:40-0:50: "Walk Forward 前进"
- 0:50-1:00: "Turn Left/Right 左右转"
- 1:00-1:10: "Sidestep 横移"
- 1:10-1:20: "Complex Gaits 复杂步态"
- 1:20-1:30: "8 Gaits Learned 8种步态习得"

---

#### 1:30-2:00 技术（30秒）

**卡片1 (1:30-1:40)：**
```
Tech Stack 技术栈

• JAX + MuJoCo MJX + Brax PPO
• AMD Radeon gfx1100
• ROCm 7.2.1
• Python 3.12
• 1024 parallel envs
```

**卡片2 (1:40-1:50)：**
```
Implementation 实现

From-scratch single-GPU jit PPO
从零实现单卡jit PPO

Real physics simulation
真实物理仿真

Learned gaits, NOT scripted
学习步态，非脚本
```

**卡片3 (1:50-2:00)：**
```
Training 训练

CPU Training
ROCm profiler bug blocked GPU
ROCm profiler bug阻塞了GPU

275 min, 62.26M steps
20 checkpoints saved
```

---

#### 2:00-2:30 结果（30秒）

**卡片1 (2:00-2:10)：**
```
Results 结果

Reward: 0.001 → 23.9
Training: 62.26M env steps
Checkpoints: 20 saved
Log: train.log
```

**卡片2 (2:10-2:20)：**
```
8 Gaits Learned
8种步态全部习得

Stand | Walk F/B | Turn L/R
Sidestep L/R | Complex moves

Min trunk height: 0.288m
Never falls 从不跌倒
```

**卡片3 (2:20-2:30)：**
```
Key Achievement
关键成果

SAME network controls ALL motions
同一网络控制所有动作

Learned from scratch
从零开始学习

Real PPO convergence
真实PPO收敛
```

---

#### 2:30-3:00 下一步（30秒）

**卡片1 (2:30-2:40)：**
```
Next Steps 下一步

1. Fix ROCm profiler race bug
   修复ROCm profiler竞态bug

2. GPU training → 10x+ speedup
   GPU训练 → 10倍以上提速
```

**卡片2 (2:40-2:50)：**
```
Future Work 未来工作

• Domain randomization
  域随机化

• Sim-to-Real transfer
  仿真到真实迁移

• More complex terrains
  更复杂地形
```

**卡片3 (2:50-3:00)：**
```
Open Source 开源

GitHub:
AMD-DEV-CONTEST/
Radeon-hackathon-2026-07
amd-physical-ai-locomotion/

MIT License

Contact: istics1998
```

---

### 5. 字幕样式设置（剪映）

**主标题：**
- 字体：思源黑体 Bold / 微软雅黑 Bold
- 字号：60-72
- 颜色：白色 #FFFFFF
- 描边：黑色 3px
- 背景：深蓝色半透明条 #0F172A 60%透明度

**副标题/正文：**
- 字号：32-42
- 颜色：白色 #FFFFFF
- 描边：黑色 2px

**小字/overlay：**
- 字号：24-28
- 位置：左上角或右上角
- 背景：深色半透明框

**动画：**
- 入场：淡入 0.3秒
- 出场：淡出 0.3秒

---

### 6. 导出设置

**剪映导出参数：**
- 分辨率：1920x1080 (1080p)
- 帧率：30fps
- 码率：8-10 Mbps（高质量）
- 格式：MP4 (H.264)
- 音频：可选静音轨或背景音乐（轻音乐，音量20%）

**预计大小：** 8-15 MB

---

## 进阶版：纯ffmpeg命令行

### 方案A：使用已生成的视频 + 字幕

```bash
cd /home/ist/桌面/myamd/amd-physical-ai-locomotion

# 给已生成的3分钟视频添加字幕
ffmpeg -i assets/demo_hackathon_3min.mp4 \
  -vf "subtitles=demo_subtitle.srt:force_style='FontName=DejaVu Sans,FontSize=28,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,BackColour=&H80000000,MarginV=50'" \
  assets/demo_hackathon_3min_subtitled.mp4
```

### 方案B：手动拼接（完全自定义）

#### 1. 提取demo_policy.mp4精华60秒
```bash
ffmpeg -i assets/demo_policy.mp4 -ss 10 -t 60 -c copy assets/demo_clip_60s.mp4
```

#### 2. 制作文字卡片（深色背景）
```bash
# 开场卡片
ffmpeg -f lavfi -i color=c=0x0F172A:s=1920x1080:d=5 \
  -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:\
text='AMD Physical AI Hackathon':fontcolor=white:fontsize=72:x=(w-text_w)/2:y=400,\
drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
text='Quadruped Locomotion RL on Radeon':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=500" \
  card_intro.mp4

# 问题卡片
ffmpeg -f lavfi -i color=c=0x0F172A:s=1920x1080:d=5 \
  -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:\
text='Challenge':fontcolor=0x38BDF8:fontsize=64:x=(w-text_w)/2:y=350,\
drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
text='Train Go1 to follow velocity commands':fontcolor=white:fontsize=40:x=(w-text_w)/2:y=500" \
  card_challenge.mp4

# 技术卡片
ffmpeg -f lavfi -i color=c=0x0F172A:s=1920x1080:d=10 \
  -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:\
text='Tech Stack':fontcolor=0x38BDF8:fontsize=64:x=(w-text_w)/2:y=300,\
drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
text='JAX + MuJoCo MJX + Brax PPO':fontcolor=white:fontsize=40:x=(w-text_w)/2:y=450,\
drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
text='AMD Radeon gfx1100 + ROCm 7.2.1':fontcolor=white:fontsize=40:x=(w-text_w)/2:y=520" \
  card_tech.mp4

# 结果卡片
ffmpeg -f lavfi -i color=c=0x0F172A:s=1920x1080:d=10 \
  -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:\
text='Results':fontcolor=0xF97316:fontsize=64:x=(w-text_w)/2:y=300,\
drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
text='Reward\: 0.001 → 23.9':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=450,\
drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
text='62.26M steps, 20 checkpoints':fontcolor=white:fontsize=40:x=(w-text_w)/2:y=530" \
  card_results.mp4

# GitHub卡片
ffmpeg -f lavfi -i color=c=0x0F172A:s=1920x1080:d=10 \
  -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:\
text='Open Source':fontcolor=0x22D3EE:fontsize=64:x=(w-text_w)/2:y=350,\
drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
text='GitHub\: AMD-DEV-CONTEST/Radeon-hackathon-2026-07':fontcolor=white:fontsize=36:x=(w-text_w)/2:y=480,\
drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
text='MIT License':fontcolor=white:fontsize=36:x=(w-text_w)/2:y=550" \
  card_github.mp4
```

#### 3. preview.gif循环
```bash
ffmpeg -stream_loop 15 -i assets/preview.gif -t 15 \
  -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30" \
  preview_loop.mp4
```

#### 4. 拼接所有片段
```bash
# 创建拼接列表
cat > concat_manual.txt << EOF
file 'card_intro.mp4'
file 'preview_loop.mp4'
file 'card_challenge.mp4'
file 'assets/demo_clip_60s.mp4'
file 'card_tech.mp4'
file 'card_results.mp4'
file 'card_github.mp4'
EOF

# 拼接
ffmpeg -f concat -safe 0 -i concat_manual.txt \
  -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
  assets/demo_manual_3min.mp4
```

#### 5. 添加字幕
```bash
ffmpeg -i assets/demo_manual_3min.mp4 \
  -vf "subtitles=demo_subtitle.srt:force_style='FontName=DejaVu Sans,FontSize=28,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2'" \
  assets/demo_final.mp4
```

---

## 方案C：最简单 - 直接用已生成的视频

`assets/demo_hackathon_3min.mp4` 已经生成好了（8.3MB, 3分06秒）

只需：
1. 用剪映打开
2. "文本" → "识别字幕" → 自动生成字幕
3. 手动调整字幕位置和样式
4. 导出

或者直接用ffmpeg加字幕：
```bash
cd /home/ist/桌面/myamd/amd-physical-ai-locomotion
ffmpeg -i assets/demo_hackathon_3min.mp4 \
  -vf "subtitles=demo_subtitle.srt:force_style='FontSize=28,PrimaryColour=&H00FFFFFF,Outline=2'" \
  assets/demo_with_subs.mp4
```

---

## 上传B站检查清单

- [ ] 时长 ≤ 3分钟 ✅
- [ ] 分辨率 1080p ✅
- [ ] 中英双语字幕 
- [ ] 前5秒有吸引力
- [ ] 包含GitHub链接
- [ ] 标题：AMD Physical AI Hackathon - Quadruped Locomotion RL
- [ ] 分区：科技 → 计算机技术
- [ ] 标签：AMD, ROCm, 强化学习, 机器人, Physical AI
- [ ] 简介包含GitHub链接

---

## 常见问题

**Q: 字体找不到？**
```bash
# 查看系统字体
fc-list | grep -i "sans\|dejavu\|noto"
# 如果没有，用系统默认
-vf "drawtext=text='...'..." # 不指定fontfile
```

**Q: 预览视频？**
```bash
mpv assets/demo_hackathon_3min.mp4
# 或
vlc assets/demo_hackathon_3min.mp4
# 或
firefox assets/demo_hackathon_3min.mp4
```

**Q: 视频太大？**
```bash
# 降低码率
ffmpeg -i input.mp4 -b:v 6M output.mp4
```

**Q: 需要背景音乐？**
在剪映里添加：音频 → 音乐 → 搜索"科技感""未来"，音量20-30%
