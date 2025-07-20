from OC_combined import *


def read_labels(file_name):
    words = []
    with open(file_name, 'r') as f:
        contents = f.read()


    return contents.split()


test_name = "test_cases/test_case_1"
image_path = test_name + ".jpeg"
image_labels = test_name + ".txt"

labels = read_labels(image_labels)
predictions = full_process_test_pic(image_path)


prediction_results = list(zip(labels, predictions))
print(prediction_results)

CS_correct = []
CS_incorrect = []

for (label, (prediction, confidence_score)) in prediction_results:
    if label == prediction:
        CS_correct.append(confidence_score)
    else:
        CS_incorrect.append(confidence_score)

print("Avg correct pred", np.mean(CS_correct))
print("Avg incorrect pred", np.mean(CS_incorrect))
