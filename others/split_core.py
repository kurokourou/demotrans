from os import path, mkdir
from tqdm import tqdm
import pandas as pd

filepath = 'EVBNews_2_0_transformer_200k.csv'
df = pd.read_csv(filepath)
core_sentences = df['core_en'].tolist()
num_of_split = 10 

lines = core_sentences
num_line = len(lines)
count = 0
for i in tqdm(range(num_of_split-1)):
    temp_file_name = 'core_%02d_%d.txt'%(i+1, num_of_split)
    temp_file = open(path.join('temp_input',temp_file_name),'w')
    num_line_per_file = num_line//num_of_split
    for j in range(num_line_per_file): 
        temp_file.write(lines[count]+'\n')
        count = count + 1
# # Last file 
temp_file_name = 'core_%d_%d.txt'%(num_of_split, num_of_split)
temp_file = open(path.join('temp_input',temp_file_name),'w')
for i in range(count,num_line):
    temp_file.write(lines[i]+'\n')