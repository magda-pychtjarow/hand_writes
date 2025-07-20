from OCR_1 import *
from OCR_2 import *
#easy ocr had to be installed via pip

def full_process_test_pic(image_path):
    lines_with_ci = read_with_easyocr(image_path)
    lines = [(img, word) for (img, word, ci) in lines_with_ci]

    conf_easy = [ci for (img, word, ci) in lines_with_ci]

    words_with_ci = detect_words_transformers(lines)
    words = [post_process_detections(word) for (word, ci) in words_with_ci]

    conf_dl = [ci for (word, ci) in words_with_ci]

    joint_confidence = [x * y for (x, y) in zip(conf_easy, conf_dl)]

    print(list(zip(words, joint_confidence)))
    return list(zip(words, joint_confidence))

