import pandas as pd
import numpy as np

f = open('s2s.txt', 'w')

a = pd.read_csv('C:/MLProjects/venv7/BTC-USD.csv', sep=';', usecols=[1, 4])
print(a.columns)
# print(a.head())

list_chars = []
list_var = []

a['mid'] = (a['Open'] + a['Close']) / 2
var = ((a['Open'] - a['Close']) / a['mid'])
max_var = (max(var))
print(max_var)
# list_var.append(var)
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z']
char_indices = {i: c for i, c in enumerate(sorted(np.linspace(-max_var, max_var, len(alphabet))))}
print(len(alphabet))
print(char_indices)


def int_to_char(int_prev, int_next, int_mid):
    sentence = []
    var1 = (int_next - int_prev) / int_mid
    # print(var1)
    for i in range(0, len(alphabet) - 1):
        if char_indices[i] < var1 <= char_indices[i + 1]:
            sentence.append(alphabet[i])

    sentence2 = (''.join(map(str, sentence)))
    return sentence2


f = open('vanga-1/btc/BTC-s2s.txt', 'w')
for i in range(5, 25):
    for all in range(0, len(a['Open']) - i, 5):
        sentence3 = []
        for week in range(all, all + i - 1):
            chars = a.at[week, 'Open']
            # print(type(float(chars)))
            chars2 = (a.at[week, 'Close'])
            chars3 = (a.at[week, 'mid'])
            # print(chars2)
            sentence3.append(int_to_char(chars, chars2, chars3))
            # print(int_to_char(chars, chars2))
        list_chars.append(''.join(map(str, sentence3)))

output_chars = ('\t'.join(map(str, list_chars)))

# print(list_chars)

# print(sentence3)
# print(output_chars)
list_output = ['qwertyuiopasdfghjklzxcvbnm\t','qwertyuiopasdfghjklzxcvbnm\n']

for i in range(0, len(list_chars)):
    if i % 2 == 0:
        c = list_chars[i] + '\t'
        list_output.append(c)
    elif i % 2 != 0:
        c = list_chars[i] + '\n'
        list_output.append(c)

# print(list_output)
# print(list_var)

output_chars2 = (''.join(map(str, list_output)))
# print(output_chars2)

#print(output_chars2)
f.write(output_chars2)
f.close()

