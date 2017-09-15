import sys
f = open(sys.argv[1], 'r')
#f = open('Q1.txt', 'r')
sete = f.read()
f.close()

word = sete.split()
data = []
count = []


for i in word:
	if i in data:
		count[data.index(i)] = count[data.index(i)] + 1
	else:
		data.append(i)
		count.append(0)
		count[data.index(i)] = count[data.index(i)] + 1


with open('Q1.txt', 'w') as f2:
    for i in range(0, len(data)-1):
        f2.write(data[i])
        f2.write(' ')
        f2.write(str(i))
        f2.write(' ')
        f2.write(str(count[i]))
        f2.write('\n')
    #f2.write('\n')
    n = len(data) - 1
    f2.write(data[n])
    f2.write(' ')
    f2.write(str(n))
    f2.write(' ')
    f2.write(str(count[n]))



