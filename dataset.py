import csv
import pandas as pd
from cytomine.models import ImageInstance, Annotation

x_size = 524

def build_dataset(filename):
    reader = csv.reader(open(filename, 'r'), delimiter=';')
    df = pd.read_csv(filename, sep=';')

    #print(df)

    i = 0
    for row in reader:
        if(i > 0):
            annotation = Annotation()
            annotation.id = int(row[0])
            annotation.fetch()

            image = ImageInstance()
            image.id = int(row[5])
            image.fetch()

            slice_image = image.reference_slice()
            x = round(float(row[3])-(x_size/2))
            y = image.height - round(float(row[4])+(x_size/2))

            slice_image.window(x, y, x_size, x_size, 
                               dest_pattern="dataset/" + str(i) + "_x.jpg")
            slice_image.window(x, y, x_size, x_size, 
                               dest_pattern="dataset/" + str(i) + "_y.jpg",
                               mask=True,
                               terms=annotation.term)
        i += 1
        if(i == 5):
            break