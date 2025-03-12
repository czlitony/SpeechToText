# https://huggingface.co/1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase
from typing import List
import glob
import opencc
from punctuators.models import PunctCapSegModelONNX

m: PunctCapSegModelONNX = PunctCapSegModelONNX.from_pretrained(
    "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
)

converter = opencc.OpenCC('t2s')  # 't2s' 表示 "繁体转简体"

input_texts : List[str] = []

txt_files = glob.glob("*.txt")
for txt_file in txt_files:
    with open(txt_file, "r", encoding="utf-8") as file:
        text = file.read()
        text_simplified = converter.convert(text)
        input_texts.append(text_simplified)

results: List[List[str]] = m.infer(
    texts=input_texts, apply_sbd=True,
)
# for input_text, output_texts in zip(input_texts, results):
#     print(f"Input: {input_text}")
#     print(f"Outputs:")
#     for text in output_texts:
#         print(f"\t{text}")
#     print()

output_texts : List[str] = []
for result in results:
    output_texts.append("\n".join(result))
print(output_texts)
with open("新桥镇.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(output_texts))
