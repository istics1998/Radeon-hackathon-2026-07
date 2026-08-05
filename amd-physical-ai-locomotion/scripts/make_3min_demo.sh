#!/bin/bash
# 快速生成3分钟hackathon演示视频
# 结构：问题(30s) → 方案演示(60s) → 技术(30s) → 结果(30s) → 下一步(30s)

set -e

ASSETS_DIR="assets"
OUT_VIDEO="assets/demo_hackathon_3min.mp4"

echo "=== 制作3分钟演示视频 ==="

# 0. 创建文字卡片函数
make_title_card() {
    local text="$1"
    local duration=$2
    local output=$3

    ffmpeg -f lavfi -i color=c=0x0F172A:s=1920x1080:d=$duration \
        -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:\
text='$text':fontcolor=white:fontsize=72:x=(w-text_w)/2:y=(h-text_h)/2" \
        -y "$output"
}

# 1. 问题卡片 (0:00-0:30, 30秒)
echo "[1/7] 生成问题卡片..."
make_title_card "Quadruped Locomotion RL on AMD Radeon\n训练四足机器人按摇杆指令行走" 5 /tmp/card1_problem.mp4

make_title_card "Challenge: Train Go1 to follow velocity commands\n挑战: 训练Go1跟随速度指令" 5 /tmp/card2_challenge.mp4

make_title_card "Goal: Walk | Turn | Sidestep under joystick control\n目标: 摇杆控制下的行走|转向|横移" 5 /tmp/card3_goal.mp4

# 前15秒: preview.gif循环
ffmpeg -stream_loop 15 -i assets/preview.gif -t 15 -vf "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30" -y /tmp/preview_loop.mp4

# 合并问题部分 (15s卡片 + 15s预览)
echo "file '/tmp/card1_problem.mp4'" > /tmp/concat_problem.txt
echo "file '/tmp/card2_challenge.mp4'" >> /tmp/concat_problem.txt
echo "file '/tmp/card3_goal.mp4'" >> /tmp/concat_problem.txt
echo "file '/tmp/preview_loop.mp4'" >> /tmp/concat_problem.txt
ffmpeg -f concat -safe 0 -i /tmp/concat_problem.txt -c copy -y /tmp/part1_problem.mp4

# 2. 方案演示 (0:30-1:30, 60秒) - 取demo_policy.mp4精华片段
echo "[2/7] 提取方案演示片段..."
# demo_policy.mp4是167秒,取前60秒最精彩的部分
ffmpeg -i assets/demo_policy.mp4 -ss 10 -t 60 -c copy -y /tmp/part2_demo.mp4

# 3. 技术卡片 (1:30-2:00, 30秒)
echo "[3/7] 生成技术说明卡片..."
make_title_card "Tech Stack 技术栈\nJAX + MuJoCo MJX + PPO\nAMD Radeon gfx1100 + ROCm 7.2.1" 10 /tmp/card4_tech.mp4

make_title_card "From-scratch single-GPU jit PPO\n从零实现单卡jit PPO\nReal physics, learned gaits" 10 /tmp/card5_impl.mp4

make_title_card "CPU Training (ROCm profiler bug blocked GPU)\nCPU训练(ROCm profiler bug阻塞GPU)\n275 min, 62.26M steps" 10 /tmp/card6_training.mp4

echo "file '/tmp/card4_tech.mp4'" > /tmp/concat_tech.txt
echo "file '/tmp/card5_impl.mp4'" >> /tmp/concat_tech.txt
echo "file '/tmp/card6_training.mp4'" >> /tmp/concat_tech.txt
ffmpeg -f concat -safe 0 -i /tmp/concat_tech.txt -c copy -y /tmp/part3_tech.mp4

# 4. 结果卡片 (2:00-2:30, 30秒)
echo "[4/7] 生成结果展示卡片..."
make_title_card "Results 结果\nReward: 0.001 → 23.9\n62.26M steps, 20 checkpoints saved" 10 /tmp/card7_results.mp4

make_title_card "All 8 gaits learned 8种步态全部习得\nWalk | Turn | Sidestep | Complex moves\nMin trunk height: 0.288m (never falls)" 10 /tmp/card8_gaits.mp4

make_title_card "Same network controls all motions\n同一网络控制所有动作\nNot scripted - learned from scratch" 10 /tmp/card9_learned.mp4

echo "file '/tmp/card7_results.mp4'" > /tmp/concat_results.txt
echo "file '/tmp/card8_gaits.mp4'" >> /tmp/concat_results.txt
echo "file '/tmp/card9_learned.mp4'" >> /tmp/concat_results.txt
ffmpeg -f concat -safe 0 -i /tmp/concat_results.txt -c copy -y /tmp/part4_results.mp4

# 5. 下一步卡片 (2:30-3:00, 30秒)
echo "[5/7] 生成下一步计划卡片..."
make_title_card "Next Steps 下一步\nFix ROCm profiler race bug\n修复ROCm profiler竞态bug" 10 /tmp/card10_next.mp4

make_title_card "GPU training → Large speedup\nGPU训练 → 大幅提速\nDomain randomization → Sim-to-Real" 10 /tmp/card11_future.mp4

make_title_card "GitHub: istics1998/Radeon-hackathon-2026-07\namd-physical-ai-locomotion/\nMIT License | Open Source" 10 /tmp/card12_github.mp4

echo "file '/tmp/card10_next.mp4'" > /tmp/concat_next.txt
echo "file '/tmp/card11_future.mp4'" >> /tmp/concat_next.txt
echo "file '/tmp/card12_github.mp4'" >> /tmp/concat_next.txt
ffmpeg -f concat -safe 0 -i /tmp/concat_next.txt -c copy -y /tmp/part5_next.mp4

# 6. 合并所有部分
echo "[6/7] 合并最终视频..."
echo "file '/tmp/part1_problem.mp4'" > /tmp/concat_final.txt
echo "file '/tmp/part2_demo.mp4'" >> /tmp/concat_final.txt
echo "file '/tmp/part3_tech.mp4'" >> /tmp/concat_final.txt
echo "file '/tmp/part4_results.mp4'" >> /tmp/concat_final.txt
echo "file '/tmp/part5_next.mp4'" >> /tmp/concat_final.txt

ffmpeg -f concat -safe 0 -i /tmp/concat_final.txt \
    -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
    -y "$OUT_VIDEO"

# 7. 添加音频静音轨（如果需要）
# ffmpeg -i "$OUT_VIDEO" -f lavfi -i anullsrc=r=44100:cl=stereo -c:v copy -c:a aac -shortest -y "${OUT_VIDEO%.mp4}_with_audio.mp4"

echo "[7/7] ✅ 完成！"
du -h "$OUT_VIDEO"
echo ""
echo "生成的视频: $OUT_VIDEO"
echo "时长: 约3分钟 (30+60+30+30+30=180秒)"
echo ""
echo "建议添加字幕："
echo "1. 用剪映自动识别字幕（推荐，快速）"
echo "2. 或用 ffmpeg 手动烧录 .srt 字幕文件"
echo ""
echo "下一步："
echo "1. 检查视频: mpv $OUT_VIDEO"
echo "2. 添加字幕（剪映或ffmpeg）"
echo "3. 上传B站替换现有链接"
