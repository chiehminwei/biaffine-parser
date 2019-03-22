win = 0
	
for line in open('data/test.conllx'):
	line = line.strip()
	try: 
		win = max(win, int(line.split('\t')[0]))
	except:
		pass
print(win) 