#!/bin/bash
# 手动剪辑命令集合 - 按需使用

# 1. 提取demo_policy.mp4的精华片段（比如第10-70秒）
ffmpeg -i assets/demo_policy.mp4 -ss 10 -t 60 -c copy assets/demo_clip_60s.mp4

# 2. 制作单个文字卡片（可自定义文字）
ffmpeg -f lavfi -i color=c=0x0F172A:s=1920x1080:d=5 \
  -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:\
text='Your Text Here':fontcolor=white:fontsize=60:x=(w-text_w)/2:y=(h-text_h)/2" \
  card_custom.mp4

# 3. 给视频添加文字overlay（角落显示文字）
ffmpeg -i assets/demo_policy.mp4 \
  -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:\
text='Reward: 0.001 → 23.9':fontcolor=white:fontsize=36:x=50:y=50:\
box=1:boxcolor=black@0.5:boxborderw=5" \
  assets/demo_with_text.mp4

# 4. 拼接多个视频
cat > /tmp/my_concat.txt << EOF
file 'clip1.mp4'
file 'clip2.mp4'
file 'clip3.mp4'
EOF
ffmpeg -f concat -safe 0 -i /tmp/my_concat.txt -c copy output.mp4

# 5. 给视频加边框+文字（画中画效果）
ffmpeg -i assets/demo_policy.mp4 \
  -vf "pad=iw+40:ih+120:20:100:color=0x0F172A,\
drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:\
text='AMD Physical AI - Quadruped Locomotion':fontcolor=white:fontsize=42:x=(w-text_w)/2:y=30" \
  assets/demo_framed.mp4

# 6. 调整视频速度（1.5倍速，节省时间）
ffmpeg -i assets/demo_policy.mp4 -filter:v "setpts=0.67*PTS" assets/demo_fast.mp4

# 7. 从视频截取静帧做封面
ffmpeg -i assets/demo_policy.mp4 -ss 30 -frames:v 1 assets/thumbnail.jpg

# 8. 添加字幕文件（需要先准备.srt文件）
# 字幕文件格式参考：
cat > /tmp/example.srt << 'EOF'
1
00:00:00,000 --> 00:00:05,000
Quadruped Locomotion RL on AMD Radeon
训练四足机器人按摇杆指令行走

2
00:00:05,000 --> 00:00:10,000
Training with PPO on ROCm
使用PPO在ROCm上训练
EOF

# 烧录字幕到视频
ffmpeg -i assets/demo_policy.mp4 -vf "subtitles=/tmp/example.srt:force_style='FontSize=24,PrimaryColour=&H00FFFFFF'" assets/demo_subtitled.mp4

# 9. 检查视频信息
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 assets/demo_policy.mp4
