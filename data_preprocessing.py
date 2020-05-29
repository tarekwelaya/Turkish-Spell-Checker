import csv
import re
import math
import random
import numpy as np




class DataProcessing():
    
    def __init__(self):
        self.keyboard_cartesian = {'q': {'x': 1, 'y': 1}, 'w': {'x': 2, 'y': 1}, 'e': {'x': 3, 'y': 1},
                                   'r': {'x': 4, 'y': 1}, 't': {'x': 5, 'y': 1}, 'y': {'x': 6, 'y': 1},
                                   'u': {'x': 7, 'y': 1}, 'ı': {'x': 8, 'y': 1}, 'o': {'x': 9, 'y': 1},
                                   'p': {'x':10, 'y': 1}, 'ğ': {'x':11, 'y': 1}, 'ü': {'x':12, 'y': 1},
                                   'a': {'x': 1, 'y': 2}, 's': {'x': 2, 'y': 2}, 'd': {'x': 3, 'y': 2},
                                   'f': {'x': 4, 'y': 2}, 'g': {'x': 5, 'y': 2}, 'h': {'x': 6, 'y': 2},
                                   'j': {'x': 7, 'y': 2}, 'k': {'x': 8, 'y': 2}, 'l': {'x': 9, 'y': 2},
                                   'ş': {'x':10, 'y': 2}, 'i': {'x':11, 'y': 2},
                                   'z': {'x': 1, 'y': 3}, 'x': {'x': 2, 'y': 3}, 'c': {'x': 3, 'y': 3},
                                   'v': {'x': 4, 'y': 3}, 'b': {'x': 5, 'y': 3}, 'n': {'x': 6, 'y': 3},
                                   'm': {'x': 7, 'y': 3}, 'ö': {'x': 8, 'y': 3}, 'ç': {'x': 9, 'y': 3}}
        
        self.nearest_to_i = self.get_nearest_to_i()
        self.lines = []
        self.corrupted_lines = []

    def get_nearest_to_i(self):
        nearest_to_i = {}
        for i in self.keyboard_cartesian.keys():
            nearest_to_i[i] = []
            for j in self.keyboard_cartesian.keys():
                distance = self.euclidean_distance(i, j) 
                if distance < 2.0:
                    nearest_to_i[i].append(j)
        return nearest_to_i


    def euclidean_distance(self, a, b):
        X = (self.keyboard_cartesian[a]['x'] - self.keyboard_cartesian[b]['x']) ** 2      
        Y = (self.keyboard_cartesian[a]['y'] - self.keyboard_cartesian[b]['y']) ** 2
        return math.sqrt(X + Y)
        
    def convert_data_to_lines(self, filename, max_chars=100, min_words =8, min_char =15):
        pattern = re.compile("^[a-zA-Z0-9ğüşöçıİĞÜŞÖÇ \n]+$")
        with open(filename, "r") as file:
            lines_num = 0
            lines =[]
            title = False
            lines_num = 0
            for line in file:
                if re.match('<[^<]+>',line):
                    title = True
                    continue
                if title:
                    title = False
                    continue
                line = re.sub('[\(\[].*?[\)\]]', '', line)
                line = line.rstrip("\n")
                line = re.sub(r'\d+','',line)
                line = re.sub(r"\s+", " ", line)
                if not pattern.match(line):
                    continue
                lines_num += 1
                line = re.sub('I','ı',line) # convert I to lower case ı
                line = re.sub('İ','i',line)
                line = line.lower()
                line = line.lstrip() # remove white-sapace if exsist in the begining of the string
                line = line.strip() #remove white-space at the end
                if not line.strip():
                    continue
                if len(line.split()) < min_words-1 and len(line) < min_char:
                    continue
                if len(line) > max_chars:
                    sub_lines = []
                    while len(line) >= max_chars:
                        ar = self.find_space(line, ' ')
                        index = 0
                        last_space = ar[index];
                        while 1:
                            index = index + 1
                            if len(ar) < index+1:
                                break
                            if ar[index] >= max_chars:
                                break
                            last_space = ar[index]
                        sub_lines.append(line[:last_space])
                        line = line[last_space+1:]
                    for sub in sub_lines:
                        lines.append(sub)
                    if len(line.split()) > min_words-1 and len(line) > min_char:
                        lines.append(line)
                else:
                    lines.append(line)
                    
                lines = list(set(lines))
                lines = sorted(lines, key=len)
                
                self.lines = lines

    def find_space(self, s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    def corrupt_lines(self, threshold = 7):
        corrupted_lines = []
        for line in self.lines:
            for x in range(random.randint(1, 10)):
                corrupt_arr = np.random.randint(100, size=(len(line),))
                li = list(line)
                cor = corrupt_arr < threshold
                for i, corrupt in enumerate(cor):
                    if corrupt and li[i] != ' ':
                        corrupt_method = np.random.randint(1,4)
                        if corrupt_method == 1:
                            li[i] = random.choice(self.nearest_to_i[li[i]])
                        elif corrupt_method == 2:
                            li[i] = ''
                        elif corrupt_method == 3:
                            li.insert(i,random.choice(self.nearest_to_i[li[i]]))
                corrupted_lines.append((''.join(li), line))
        self.corrupted_lines = corrupted_lines

    def write_data(self, outfname):
        with open(outfname, 'w') as f:
            for c, l in self.corrupted_lines:
                f.write(c + "|" +l + "\n")

if __name__ == "__main__":
    process = DataProcessing()
    process.convert_data_to_lines("wiki_train")
    process.corrupt_lines() #max_type_num
    process.write_data("train.txt")
    
    process.convert_data_to_lines("wiki_test")
    process.corrupt_lines() #max_type_num
    process.write_data("test.txt")
    
    process.convert_data_to_lines("wiki_val")
    process.corrupt_lines() #max_type_num
    process.write_data("val.txt")
