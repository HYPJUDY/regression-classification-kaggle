# preprocess: perform standard scale to data
# before performing logistic regression
# value'=(value-min_value)/(max_value-min_value)

## step 1: find min_value and max_value
# fr=open("/home/classification/data/train_data_new.txt","r")
# min_value = 10
# max_value = -10
# for line in fr:
# 	data_line = str(line).split()
# 	for i in range(1,len(data_line)):
# 		key_value = data_line[i].split(":")
# 		key = int(key_value[0])
# 		value = float(key_value[1])
# 		if (value < min_value):
# 			min_value = value
# 		if (value > max_value):
# 			max_value = value
# print min_value, max_value

## step 2: modify train value'=(value-min_value)/(max_value-min_value)
# fr=open("/home/classification/data/train_data_new.txt","r")
# fw=open("/home/classification/data/train_data_standard.txt","w")
# min_value = 3.9459613743e-06
# max_value = 4.24878248131e+12
# min_value = 0
# diff = max_value - min_value
# for line in fr:
# 	flag = 0
# 	data_line = str(line).split()
# 	for i in range(1,len(data_line)):
# 		key_value = data_line[i].split(":")
# 		key = int(key_value[0])
# 		value = float(key_value[1])
# 		value = value / diff
# 		if (flag == 0):
# 			fw.write(str(int(data_line[0])))
# 			flag = 1
# 		fw.write(" ")
# 		fw.write(str(key))
# 		fw.write(":")
# 		fw.write(str(value))
# 	if (flag!=0):
# 		fw.write("\n")

# step 3: modify test value'=(value-min_value)/(max_value-min_value)
fr=open("/home/classification/data/test_data.txt","r")
fw=open("/home/classification/data/test_data_standard.txt","w")
min_value = 3.9459613743e-06
max_value = 4.24878248131e+12
min_value = 0
diff = max_value - min_value
for line in fr:
	flag = 0
	data_line = str(line).split()
	for i in range(1,len(data_line)):
		key_value = data_line[i].split(":")
		key = int(key_value[0])
		value = float(key_value[1])
		value = value / diff
		if (flag == 0):
			fw.write(str(int(data_line[0])))
			flag = 1
		fw.write(" ")
		fw.write(str(key))
		fw.write(":")
		fw.write(str(value))
	if (flag!=0):
		fw.write("\n")

## (optional) step 4: check several head lines
# fr=open("/home/classification/data/train_data_standard.txt","r")
# t=0
# for line in fr:
# 	print line
# 	t = t + 1
# 	if (t == 5):
# 		break
