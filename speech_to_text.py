import whisper
import glob

mp3_files = glob.glob("*.m4a")
print(mp3_files)

# 加载模型
model = whisper.load_model("large")  # 可选 tiny, base, small, medium, large

for mp3_file in mp3_files:
    print("converting", mp3_file)
    result = model.transcribe(mp3_file, language='zh')

    mp3_file_name = mp3_file[:-4]
    print("saving result to file", f"{mp3_file_name}.txt")
    with open(f"{mp3_file_name}.txt", "w", encoding="utf-8") as file:
        file.write(result["text"])
