#with open('/Users/Jimmy/train.conllx') as f:
import os
import glob
from collections import defaultdict
from tabulate import tabulate


# folder_path = "/Users/Jimmy/Downloads/Universal_Dependencies_2.3/ud-treebanks-v2.3/"
# for file_path in sorted(glob.glob(folder_path + '/*/*train.conllu')):
# 	file = file_path.split('/')[-2]
# 	print('***********************************************')
# 	print(file)
# 	print('***********************************************')
# 	with open(file_path) as f_in:
# 		count = 0
# 		for line in f_in:
# 			line = line.strip()
# 			if line.startswith('# text ='):
# 				count += 1
# 	print(count)

corpora = {}
language = defaultdict(int)
with open('omg') as f:
	name, val = None, None
	for line in f:
		line = line.strip()
		if name and val:
			corpora[name] = val
			language[name.split('-')[0][3:]] += val
			name, val = None, None

		if line.startswith('*'):
			continue
		if line.startswith('UD'):
			name = line
		else:
			val = int(line)
print(tabulate(sorted(language.items(), key=lambda x: x[1]), tablefmt='github'))


# path = "debugging/"
# d ={'UD_Catalan-AnCora': 25,
# 	'UD_Czech-CLTT': 17,
# 	'UD_Czech-PDT': 21,
# 	'UD_French-GSD': 27,
# 	'UD_French-ParTUT': 19,
# 	'UD_Italian-ISDT': 19,
# 	'UD_Italian-ParTUT': 19,
# 	'UD_Urdu-UDTB': 32}

# batch_size = 1
# for file, batch in d.items():
# 	folder_path = os.path.join(path, file)
# 	# print(folder_path)
# 	try:
# 		file_path = glob.glob(folder_path + '.conllx')[0]
# 	except:
# 		# print(file)
# 		assert 1 == 2
# 	print('***********************************************')
# 	print(file)
# 	print('***********************************************')
# 	with open(file_path) as f_in, open('debugging_one_sentence/' + file + '.conllx','w') as f_out:
# 		count = 0
# 		for line in f_in:
# 			line = line.strip()
# 			if line.startswith('# sent_id ='):
# 				count += 1
# 			if count > (batch + 1) * batch_size:
# 				break
# 			if count >= (batch - 1) * batch_size:
# 				if line.startswith('# text ='):
# 					print(line)
# 				f_out.write(line + '\n')
			
# 	print(' ')




# path = "/Users/Jimmy/Downloads/Universal_Dependencies_2.3/ud-treebanks-v2.3/"
# d ={'UD_Catalan-AnCora': 219,
# 	'UD_Czech-CLTT': 19,
# 	'UD_Czech-PDT': 4066,
# 	'UD_French-GSD': 402,
# 	'UD_French-ParTUT': 36,
# 	'UD_Italian-ISDT': 348,
# 	'UD_Italian-ParTUT': 36,
# 	'UD_Urdu-UDTB': 63}	

# batch_size = 16
# for file, batch in d.items():
# 	folder_path = os.path.join(path, file)
# 	try:
# 		file_path = glob.glob(folder_path + '/*train.conllu')[0]
# 	except:
# 		print(file)
# 		assert 1 == 2
# 	print('***********************************************')
# 	print(file)
# 	print('***********************************************')
# 	with open(file_path) as f_in, open('debugging/' + file + '.conllx','w') as f_out:
# 		count = 0
# 		for line in f_in:
# 			line = line.strip()
# 			if line.startswith('# text ='):
# 				count += 1
# 			# if count >= (batch - 1) * batch_size:
# 			# 	if line.startswith('# text ='):
# 			# 		print(line)
# 			# 	f_out.write(line + '\n')
# 			# if count > (batch + 2) * batch_size:
# 			# 	break
# 	print(count)




# with open('data/train.conllx') as f:
# 	lines = []
# 	count = 0
# 	for line in f:
# 		line = line.strip()
# 		#if line.startswith('# text ='):
# 		if len(line) == 0:
# 			count += 1
# 	# 	if count >= 11551:
# 	# 		if line.startswith('# text ='):
# 	# 			lines.append(line)
# 	# 	if count >= 11551 + 20:
# 	# 		break

# 	# for line in lines:
# 	# 	print(line)
# print(count)