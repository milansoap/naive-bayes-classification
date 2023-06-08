from csv import reader
from math import sqrt
from math import exp
from math import pi
import matplotlib.pyplot as plt


###################### MATHEMATICAL METHODS ###################################

# izracunaj standardno devijacijo za stevila
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1) # variance from mean (how spread out are the numbers from the mean)
    return sqrt(variance) # korijen vsih stevila


# izracunamo probability uprabljajuci guassian probability function
def calculate_probability(x, mean, stdev):
    epsilon = 1e-6  # small constant to prevent division by zero
    exponent = exp(-((x - mean) ** 2 / (2 * (stdev ** 2 + epsilon))))
    return (1 / (sqrt(2 * pi) * (stdev + epsilon))) * exponent


# average value from all numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


################ CSV AND CONVERSIONS TO INT AND FLOAT ########################

# Load a CSV file and conversions to int and float
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row[0:])
        return dataset


def writeFile(filename, data):
    with open(filename, 'a') as file:
        file.write(data)
        file.write('\n')


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


############ DATASET OPERATIONS ###################

# splituje dataset na 2 dela, en del kje imamo za 0 in en del za 1
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


# Calculate the mean, stdev and count for each column in a dataset
# uporablja matematicke operacije za vsaki column (mean, stdev
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)] # zip converta iz glede matrike v gled array-a
    del (summaries[-1])
    return summaries


# prvo splitamo za class potem dobimo stevilo tuple-ova katerih imamo v datasetu
# za class 0
# (2.7420144012, 0.9265683289298018, 5),
# (3.0054686692, 1.1073295894898725, 5),
# (mean,variance,count) for each


#statistic training dataseta po klasama
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


############ PROBABILITY HELPERS ######################

#First the total number of training records is calculated from the counts stored in the summary statistics.
# This is used in the calculation of the probability of a given class or P(class) as the ratio of rows with a given class of all rows in the training data.

#Next, probabilities are calculated for each input value in the row using the Gaussian probability
# density function and the statistics for that column and of that class. Probabilities are multiplied together as they accumulated.

# P(class|data) = P(X|class) * P(class)
def calculate_class_probabilities(summaries, row):
    # total number of rows in a dataset
    total_rows = sum([summaries[label][0][2] for label in summaries])

    # inicializacija probabilitiesa
    probabilities = dict()

    # Calculate the prior probability for each class
    # P(class) = count(class) / total_rows
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)  # Prior probability P(class)

        # kalkulisi probability za vsaki atribut notraj enega reda te klase
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]

            # Multiply the prior probability by the conditional probability P(X|class)
            # multiply P(class) z (P(X|class)
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)

    # Return the calculated probabilities for each class (0, 1, etc.)
    return probabilities


# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1 #none da recemo da prvega nima a drugi je -1 da bi bili sure da prvi ki je naslednji je vecji od -1 kjer imamo samo vrednosti 0 ali 1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob: # preverjaamo ali je best label prvo none kjer je to prva vrednost pa potem ali je vecni od best-proba
            best_prob = probability
            best_label = class_value
    return best_label


# confusion matrix calculation
def calculation_of_matrix(conf_matrix):
    calc_matrix = dict()
    calc_matrix['accuracy'] = (conf_matrix['TP'] + conf_matrix['TN']) / (
                conf_matrix['TP'] + conf_matrix['TN'] + conf_matrix['FP'] + conf_matrix['FN'])
    calc_matrix['sensitivity'] = conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FN'])
    calc_matrix['specificity'] = conf_matrix['TN'] / (conf_matrix['TN'] + conf_matrix['FP'])
    calc_matrix['precision'] = conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FP'])
    calc_matrix['recall'] = calc_matrix['sensitivity']  # Recall is the same as sensitivity


    # calc_matrix['recall_Negative'] = calc_matrix['sensitivity']
    # calc_matrix['recall_Positive'] = calc_matrix['specificity']
    # calc_matrix['precision_Negative'] = conf_matrix['TP'] / (conf_matrix['TP'] + conf_matrix['FP'])
    # calc_matrix['precision_Positive'] = conf_matrix['TN'] / (conf_matrix['TN'] + conf_matrix['FN'])
    # calc_matrix['Fmeasure_Negative'] = (2 * calc_matrix['recall_Negative'] * calc_matrix['precision_Negative']) / (
    #             calc_matrix['recall_Negative'] + calc_matrix['precision_Negative'])
    # calc_matrix['Fmeasure_Positive'] = (2 * calc_matrix['recall_Positive'] * calc_matrix['precision_Positive']) / (
    #             calc_matrix['recall_Positive'] + calc_matrix['precision_Positive'])
    # Printing the results with spacing
    for key, value in calc_matrix.items():
        print(key + ':', value)
    return calc_matrix



if __name__ == "__main__":
    dataset_learn = load_csv('dataset_learn.csv')

    print(calculate_probability(1.0, 1.0, 1.0))

    for i in range(len(dataset_learn[0]) - 1):
        str_column_to_float(dataset_learn, i)

    str_column_to_int(dataset_learn, len(dataset_learn[0]) - 1)

    model = summarize_by_class(dataset_learn)

    test_dataset = load_csv('dataset_test.csv')
    output = 'predictions.csv'
    for i in range(len(test_dataset[0])):
        str_column_to_float(test_dataset, i)
    prediction_dataset = []
    for i in range(len(test_dataset)):
        predicted_value = predict(model, test_dataset[i])
        predicted_data = test_dataset[i].copy()

        predicted_data.append(predicted_value)
        prediction_dataset.append(predicted_data)
        predicted_data = ','.join(map(str, predicted_data))
        writeFile(output, predicted_data)

    str_column_to_int(prediction_dataset, len(dataset_learn[0]) - 1)
    answers = load_csv('dataset_test_answers.csv')

    for i in range(len(answers[0]) - 1):
        str_column_to_float(answers, i)
    str_column_to_int(answers, len(answers[0]) - 1)

    conf_matrix = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    for i in range(len(answers)):
        if answers[i][len(answers[0]) - 1] == 0 and prediction_dataset[i][len(prediction_dataset[0]) - 1] == 0:
            conf_matrix['TP'] += 1
        elif answers[i][len(answers[0]) - 1] == 1 and prediction_dataset[i][len(prediction_dataset[0]) - 1] == 1:
            conf_matrix['TN'] += 1
        elif answers[i][len(answers[0]) - 1] == 1 and prediction_dataset[i][len(prediction_dataset[0]) - 1] == 0:
            conf_matrix['FP'] += 1
        elif answers[i][len(answers[0]) - 1] == 0 and prediction_dataset[i][len(prediction_dataset[0]) - 1] == 1:
            conf_matrix['FN'] += 1
    print(conf_matrix)
    print()
    metrics = calculation_of_matrix(conf_matrix)
    print(metrics)


    # Data
    models = ['SVM', 'Random Forest', 'Neural Network', 'My Algorithm']
    ca = [0.9075, 0.8775, 0.8975,0.8]
    precision = [0.9082745648810355, 0.8777647070702992, 0.8981219834578336,0.651]
    recall = [0.9110308348153573, 0.8781330437580438, 0.899689182768451,0.9645]

    # Plotting
    x = range(len(models))
    width = 0.2

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, recall, width, label='Recall')
    rects2 = ax.bar([i + width for i in x], ca, width, label='Precision')
    rects3 = ax.bar([i + width * 2 for i in x], precision, width, label='CA')

    # Add x-axis labels, y-axis label, and title
    ax.set_ylabel('Scores')
    ax.set_xlabel('Models')
    ax.set_title('Model Performance')
    ax.set_xticks([i + width * 1.5 for i in x])
    ax.set_xticklabels(models)
    ax.legend()


    # Add data labels above each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.show()
