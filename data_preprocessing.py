import csv
import re
            
def process_data():
    with open('wiki_00') as file, open('wiki_tr.csv', 'w') as csvfile:
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
            pattern = re.compile("^[a-zA-Z0-9ğüşöçıİĞÜŞÖÇ \n]+$")
            if not pattern.match(line):
                continue
            if len(line) < 25:
                continue
            if not line.strip():
                continue
            if line in  lines:
                continue
            if lines_num == 1000:
                break
            lines_num += 1
            lines.append(line)
            lines = sorted(lines, key=len)
        writer = csv.writer(csvfile, delimiter='|')
        writer.writerow(('ID', 'Line'))
        for idx, line in enumerate(lines, 1):
            writer.writerow((idx, line))

if __name__ == "__main__":
    process_data()

