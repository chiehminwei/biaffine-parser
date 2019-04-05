with open('/Users/Jimmy/train.conllx') as f:
	lines = []
	count = 0
	for line in f:
		line = line.strip()
		if line.startswith('# text ='):
			count += 1
		if count >= 11551:
			if line.startswith('# text ='):
				lines.append(line)
		if count >= 11551 + 20:
			break

	for line in lines:
		print(line)