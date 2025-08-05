import csv
import numpy as np
from geopy.distance import distance
'''
lat = {}
lon = {}
with open('/home/parthpk/Mapilliary_data/mapilliary_train_seq.csv', 'r') as f:
    csvreader = csv.reader(f)
    next(csvreader, None)
    for line in csvreader:
        country = line[2].split('/')[-4]
        if country in lat:
            lat[country].append(float(line[-2]))
        else:
            lat[country] = [float(line[-2])]
        if country in lon:
            lon[country].append(float(line[-1]))
        else:
            lon[country] = [float(line[-1])]

padding = 0.005
res = 0.1
'''
padding = 0
res = 0.5

mm = {'cali': [36.5, 39, -122.78, -121],
      'ny': [39, 43, -75, -70],
      'israel-n': [31.06, 33.03, 34.56, 35.52],
      'israel-s': [29.54, 29.59, 34.93, 34.98]
}
'''
mm = {}
for i in lat:
    mm[i] = [min(lat[i])-padding, max(lat[i])+padding, min(lon[i])-padding, max(lon[i])+padding]
'''
c = 0
for i in mm:
    d = distance([mm[i][0], mm[i][2]], [mm[i][0], mm[i][3]]).km
    N = d//res
    lon_dis = (mm[i][3] - mm[i][2])/N 

    d = distance([mm[i][0], mm[i][2]], [mm[i][1], mm[i][2]]).km
    M = d//res
    lat_dis = (mm[i][1] - mm[i][0])/M
    
    c += N*M
    
print(c) 
exit()
l = []
for i in mm:
    lat = mm[i][0]
    while lat < mm[i][1]:
        lon = mm[i][2]
        while lon < mm[i][3]:
            l.append([lat,lon])
            lon += lon_dis
        lat += lat_dis
            
fin = np.array(l)
np.save(f'vid_add_{res}_{padding}.npy', fin)

