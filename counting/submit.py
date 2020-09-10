import glob

data = sorted(glob.glob("counting_result/*.txt"))

with open("submit.txt", 'w') as w:
    for path in data:
        with open(path, 'r') as fr:
            part = fr.read()
            # print(part)
        w.write(part)
        w.write('\n')
