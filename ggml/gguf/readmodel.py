#!/usr/bin/env python3
#author:rangapv@yahoo.com
#11-11-2026

import sys
from pprint import pprint

from gguf import GGUFReader

#reader = GGUFReader('/Volumes/Lexar/gemma-4-12b-it-Q4_K_M.gguf')
reader = GGUFReader('./Meta-Llama-3.1-8B-Instruct-Q6_K.gguf')
print(f'the type of Reader is {type(reader)} and its size is {sys.getsizeof(reader)}')

vars1 = vars(reader)
j = 0 
for j in range(5):
   pprint(f'the top level')
   pprint(vars(reader), depth=2)
print(f'the keys are...{dir(vars1)}')
#print(f'Reading fileds for the Model : {reader[6]}')
iter = 0

for key in reader.fields:
    iter += 1
    print(f'{iter}::{key}')
iter = 0
for key in reader.fields:
    iter += 1
    if ( iter < 13 ):
       print(f'{iter}::{key}')
       field2 = reader.fields[key]
       #version = int(field2.parts[field2.data[0]][0])
       print(f'decode-bytes is {bytes(field2.parts[field2.data[0]]).decode()}')
       #print(reader.fields[key])
    # break

totalfields = len(reader.fields)
print (f'Total fileds are {totalfields}')


field2 = reader.fields['GGUF.version']
version = int(field2.parts[field2.data[0]][0])
print(f"GGUF version: {version}")
print(f'decode-bytes is {bytes(field2.parts[field2.data[0]]).decode()}')
#field3 = reader.fields['tensor_count']
#print(f"tensor count: {field3}")

field31 = reader.fields['general.languages']
print(f"tensor general.languages: {field31}")
print(f'decode-bytes is {bytes(field31.parts[field31.data[0]]).decode()}')

field32 = reader.fields['general.architecture']
print(f"tensor general.architecture: {field32}")
print(f'decocded-bytes {bytes(field32.parts[field32.data[0]]).decode()}')

# Print all key-value metadata
#for key, field in reader.fields.items():
#    print(f"{key}: {field.parts[field.data[0]] if len(field.data) == 1 else field.data}")
