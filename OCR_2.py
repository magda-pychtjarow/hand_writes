import cv2
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def detect_words_transformers(lines):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    special_token_id = processor.tokenizer.eos_token_id
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    detected_ocr_words = []
    for (img, index) in lines:

        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        pixel_values = processor(img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)

        sequences = generated_ids.sequences
        scores = generated_ids.scores

        important_token_ids = [s for s in sequences.numpy()[0] if s != special_token_id]
        token_scores = scores[:len(important_token_ids)]  # exclude EOS score

        import torch.nn.functional as F
        token_probs = []
        for logits, token_id in zip(token_scores, important_token_ids):
            probs = F.softmax(logits, dim=-1)
            prob = probs[0, token_id].item()
            token_probs.append(prob)

        # Combine into a single word confidence
        # Option 1: Multiply probabilities (joint likelihood)
        word_confidence = 1.0
        for p in token_probs:
            word_confidence *= p

        generated_text = processor.batch_decode(sequences, skip_special_tokens=True)[0]
        detected_ocr_words.append((generated_text, word_confidence))

    return detected_ocr_words


def post_process_detections(text):
    import string
    # Remove punctuation and spaces, convert to lowercase
    translator = str.maketrans('', '', string.punctuation + ' ')
    return text.translate(translator).lower()
