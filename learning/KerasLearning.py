from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas, os, gzip, pickle
from sklearn.metrics import precision_score, accuracy_score, f1_score, roc_auc_score

model = Sequential()
model.add(Dense(2000, input_dim=3494, activation='relu'))   
model.add(Dense(2000, input_dim=3494, activation='relu'))
model.add(Dense(2957, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

test_set_file = pandas.read_csv("/home/aigul/Retro/descriptors_8_3_4/type_8_descriptors/test_set/TestSet.csv")
rule_nums = 2957
x_test = test_set_file.iloc[:, 2:3496].values
test_knowledge_matrix = {}
for i3 in range(rule_nums):
    test_knowledge_matrix[i3] = []
for el2 in pandas.Series.tolist(test_set_file.NewNumRule):
    for i4 in range(rule_nums):
        if i4 != el2:
            test_knowledge_matrix[i4].append(0)
        elif i4 == el2:
            test_knowledge_matrix[i4].append(1)

test_knowledge_matrix = pandas.DataFrame(test_knowledge_matrix).values
# y_test = test_set_file.NewNumRule

f1_list = []
precision_list = []
accuracy_list = []
roc_auc_list = []
rule_nums = 2957
trainset_file_names = os.listdir("/home/aigul/Retro/descriptors_8_3_4/type_8_descriptors/transforms_with_rules_pickle/")
count_files = 0
epochs_in_keras = 80
for i in range(epochs_in_keras):
    print(i, "EPOCH")
    for file_name in trainset_file_names[:2]:

        count_files +=1
        if count_files % 10 == 0:
            print(count_files, "NUMBER OF FILES")
        with gzip.open("/home/aigul/Retro/descriptors_8_3_4/type_8_descriptors/transforms_with_rules_pickle/" + file_name, "rb") as ds:
            all_dataframe = pickle.load(ds)

        x_train = all_dataframe[0]

        y_train = all_dataframe[1]

        model.fit(x_train, y_train, epochs=1, batch_size=200)   #batch_size 50

    predictions = model.predict(x_test)

    pandas.DataFrame(predictions).to_csv("/home/aigul/Retro/descriptors_8_3_4/type_8_descriptors/stat_from_keras/80_200_834_4*500_sig/predictions" + str(i) + ".csv")


    predictions_int = predictions >= 0.5

    precision_m = precision_score(y_true=test_knowledge_matrix, y_pred=predictions_int, average=None)
    precision_m1 = precision_m.tolist()
    precision_list.append(precision_m)
    print(precision_m.mean(), "PRECISION")

    accuracy_m = accuracy_score(y_true=test_knowledge_matrix, y_pred=predictions_int)
    accuracy_list.append(accuracy_m)
    print(accuracy_m, "ACCURACY")

    f1 = f1_score(y_pred=predictions_int, y_true=test_knowledge_matrix, average=None)
    f1_list.append(f1)
    f11 = f1.tolist()
    print(np.mean(f1), "F1")

    mask = (test_knowledge_matrix.sum(axis=0)) > 1
    auc_w = roc_auc_score(y_true=test_knowledge_matrix[:, mask], y_score=predictions[:, mask], average= "weighted")

    auc = roc_auc_score(y_true=test_knowledge_matrix[:, mask], y_score=predictions[:, mask], average=None)
    roc_auc_list.append(auc)

    #auc_1 = pandas.DataFrame(auc).to_csv("/home/aigul/Retro/stat_from_keras/80_200_325_5*500/auc_weighted" + str(i) + ".csv")
    pandas.DataFrame(auc).to_csv("/home/aigul/Retro/descriptors_8_3_4/type_8_descriptors/stat_from_keras/80_200_834_4*500_sig/auc_" + str(i) + ".csv")

    with open("/home/aigul/Retro/descriptors_8_3_4/type_8_descriptors/stat_from_keras/80_200_834_4*500_sig/auc_weighted.txt", "a") as f7:
        f7.write("%s\n" % auc_w)
    print(auc_w, "AUC")

    with open("/home/aigul/Retro/descriptors_8_3_4/type_8_descriptors/stat_from_keras/80_200_834_4*500_sig/f1.txt", "a") as f1:
        f1.write("%s\n" % f11)
    with open("/home/aigul/Retro/descriptors_8_3_4/type_8_descriptors/stat_from_keras/80_200_834_4*500_sig/precision.txt", "a") as f2:
        f2.write("%s\n" % precision_m1)
    with open("/home/aigul/Retro/descriptors_8_3_4/type_8_descriptors/stat_from_keras/80_200_834_4*500_sig/accuracy.txt", "a") as f3:
        f3.write("%s\n" % accuracy_m)

    # serialize model to JSON
    model_json = model.to_json()
    with open("/home/aigul/Retro/descriptors_8_3_4/type_8_descriptors/keras_models/80_200_834_4*500_sig/model_" + str(i) + "_epochs.json", "w") as json_file:
        json_file.write(model_json)
    serialize weights to HDF5
    model.save_weights("/home/aigul/Retro/descriptors_8_3_4/type_8_descriptors/keras_models/80_200_834_4*500_sig/" + str(i) + "_epochs.h5")
    print("Saved model to disk")
