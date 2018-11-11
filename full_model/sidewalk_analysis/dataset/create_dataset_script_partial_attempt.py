import csv
from .google_street_view import get_image


def save_image(image):
    return ''


labels_list = []
with open('./csvs/sidewalks.csv', 'r', newline='', encoding='latin-1') as f:
    sidewalks_csv = csv.reader(f, delimiter=',')

    for i, row in enumerate(sidewalks_csv):
        if i is not 0:
            sidewalk = {
                'swk_id': row[1],
                'lat': row[2],
                'long': row[3],
                'damage': row[4]
            }

            image = get_image(sidewalk['lat'], sidewalk['long'], i)
            save_image(image)
            labels_list.append(sidewalk['damage'])


with open('./csvs/labels.csv', mode='w') as labels:
    writer = csv.writer(labels, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for label in labels_list:
        writer.writerow(label)
