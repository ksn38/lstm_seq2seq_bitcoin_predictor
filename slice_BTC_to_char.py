import pandas as pd
import numpy as np

# exel == slice + 2
btc = 2695

f = open('s2s.txt', 'w')

a = pd.read_csv('C:/MLProjects/venv7/BTC-USD.csv', sep=';', usecols=[1, 4])
print(a.columns)
# print(a.head())

list_chars = []
list_var = []

a['mid'] = (a['Open'] + a['Close']) / 2
var = ((a['Open'] - a['Close']) / a['mid'])
max_var = (max(var))*0.1
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
    #print(sentence)
    return sentence2

razmer_stupeni = 24
#for razmer_stupeni in range(5, 25):
sentence3 = []
for stupenb in range(btc, razmer_stupeni + btc - 1):
    chars = a.at[stupenb, 'Open']
    # print(type(float(chars)))
    chars2 = (a.at[stupenb, 'Close'])
    # print(chars2)
    chars3 = (a.at[stupenb, 'mid'])
    # print(chars2)
    sentence3.append(int_to_char(chars, chars2, chars3))
    # print(int_to_char(chars, chars2))
list_chars.append(''.join(map(str, sentence3)))

print('list_chars', list_chars)
print('sentence3', sentence3)

listik = []
for i in range(5, len(sentence3) + 1):
    print(sentence3[-i:])
    listik.append(''.join(map(str, sentence3[-i:])))

print('listik', listik)
output_chars = ('\t'.join(map(str, listik)))

print('output_chars', output_chars)

list_output = ['qwertyuiopasdfghjklzxcvbnm \t','qwertyuiopasdfghjklzxcvbnm \n']
print(list_output)

for i in range(0, 18):
    c = listik[i] + ' \tqwertyuiopasdfghjklzxcvbnm \n'
    list_output.append(c)

print(list_output)
# print(list_var)

output_chars2 = (''.join(map(str, list_output)))
# print(output_chars2)

print(output_chars2)
f.write(output_chars2)
f.close()