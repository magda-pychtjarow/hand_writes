from OC_combined import *
import matplotlib.pyplot as plt

READY_TEST_CASES = 3
def read_labels(file_name):
    words = []
    with open(file_name, 'r') as f:
        contents = f.read()
    return contents.split()

def run_test_case(test_case_id):
    print("Running Test Case ",  test_case_id , "...")
    test_name = "test_cases/test_case_" + str(test_case_id)
    image_path = test_name + ".jpeg"
    image_labels = test_name + ".txt"

    labels = read_labels(image_labels)
    print(labels)
    print("Generating predictions...")
    predictions = full_process_test_pic(image_path)

    print("Processing predictions...")
    prediction_results = list(zip(labels, predictions))

    print("Prediction results ")
    print(prediction_results)

    CS_correct = []
    CS_incorrect = []

    for (label, (prediction, confidence_score)) in prediction_results:
        if label == prediction:
            CS_correct.append(confidence_score)
        else:
            CS_incorrect.append(confidence_score)
    return np.mean(CS_correct), np.mean(CS_incorrect)


avgs_correct, avgs_incorrect = [], []

for i in range(1,READY_TEST_CASES+1):
    avg_correct, avg_incorrect = run_test_case(i)
    avgs_correct.append(avg_correct)
    avgs_incorrect.append(avg_incorrect)

plt.boxplot([avgs_correct, avgs_incorrect], labels=['Avgs correct', 'Avgs incorrect'])
plt.title('Average confidence score distribution')
plt.ylabel('Average CS')
plt.show()