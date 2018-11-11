import csv
import random

row_samples = []
with open('./csvs/correct_sidewalk_data.csv', 'r', newline='', encoding='latin-1') as f:
    sidewalks_csv = csv.reader(f, delimiter=',')
    lisit = list(sidewalks_csv)
    
    for i in range(1000):
        row_samples.append(random.choice(list(lisit)))


with open('./csvs/rand-sample.csv', mode='w') as rand_sample:
    writer = csv.writer(rand_sample, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for row in row_samples:
        writer.writerow(row)
