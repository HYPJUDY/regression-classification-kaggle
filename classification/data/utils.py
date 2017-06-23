# data source: https://inclass.kaggle.com/c/large-scale-classification-sysu-2017/data

# remove fetures whose indices>132 in train data,
# since the indices of all features in test data <= 132
f=open("train_data.txt","r")
f1=open("train_data_new.txt","w")
for line in f:
	data_line = str(line).split()
	t = 0
	for i in range(1,len(data_line)):
		key_value = data_line[i].split(":")
		key = int(key_value[0])
		value = float(key_value[1])
		if (key<=132):
			if (t==0):
				f1.write(str(int(data_line[0])))
				t = 1
			f1.write(" ")
			f1.write(str(key))
			f1.write(":")
			f1.write(str(value))
	if (t!=0):
		f1.write("\n")

# split train dataset into train and validation data
fr=open("/home/classification/data/train_data.txt","r")
ft=open("/home/classification/data/train10_9.txt","w")
fv=open("/home/classification/data/val10_1.txt","w")
i = 1
fold = 10 # val_data:train_data_exclude_val = 1:(fold-1)
for line in fr:
	if (i != fold):
		i = i + 1
		ft.write(line)
	else:
		i = 1
		fv.write(line)

# transform the output of xgboost prediction
# into the result format we want
f=open("pred.txt","r")
f1=open("result_etad01_ga0_mcw7_dep12_iter800.txt","w")
f1.write("id,label\n")
i=0
for line in f:
	data_line = str(line)
	f1.write(str(i))
	f1.write(",")
	f1.write(data_line)
	i = i+1