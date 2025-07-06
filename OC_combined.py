from OCR_1 import *
from OCR_2 import *

image_path = "test_pic_C_list.jpeg"

lines_with_ci = read_with_easyocr(image_path)
lines = [(img, word) for (img, word, ci) in lines_with_ci]

conf_easy = [ci for (img, word, ci) in lines_with_ci]

words_with_ci = detect_words_transformers(lines)
words = [post_process_detections(word) for (word, ci) in words_with_ci]

conf_dl = [ci for (word, ci) in words_with_ci]

joint_confidence = conf_easy #* conf_dl

print(list(zip(words, joint_confidence)))
