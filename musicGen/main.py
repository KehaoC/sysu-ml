# pip install pydub simpleaudio

from pydub import AudioSegment
import simpleaudio as sa
import time
from tqdm import tqdm

# 加载音频文件
audio = AudioSegment.from_file("music1.flac")

# 转换音频数据为适当的格式
audio = audio.set_frame_rate(44100).set_channels(2).set_sample_width(2)

# 播放音频
playback = sa.play_buffer(audio.raw_data,
                          num_channels=audio.channels,
                          bytes_per_sample=audio.sample_width,
                          sample_rate=audio.frame_rate)

# 可视化播放进度条
duration = len(audio) / 1000  # 音频持续时间（秒）
for i in tqdm(range(int(duration)), desc=f"Playing ({duration:.2f}s)", unit="s"):
    time.sleep(1)  # 每秒更新一次进度条
    if not playback.is_playing():
        break

# 等待音频播放完成
playback.wait_done()
