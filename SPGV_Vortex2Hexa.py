data_2d = []
with open('Unidirectional_Raster_Scan_5000.txt', 'r') as file:
    for line in file:
        data = line.split()
        row_data = [float(num) for num in line.split()]
        data_2d.append(row_data)

max_abs_value_x = 0
max_abs_value_y = 0
for row in data_2d:
    abs_value_x = abs(row[0])
    abs_value_y = abs(row[1])
    if abs_value_x > max_abs_value_x :
        max_abs_value_x = abs_value_x
    if abs_value_y > max_abs_value_y:
        max_abs_value_y = abs_value_y
print(max_abs_value_x)
print(max_abs_value_y)

SCALE_FACTOR = 15 #position 32768 = degree SCALE_FACTOR
user_x_degree = 10 #half of the full range x degree
user_y_degree = 5 # half of the full range y degree

x_MF = user_x_degree / SCALE_FACTOR * 32768 / max_abs_value_x
y_MF = user_y_degree / SCALE_FACTOR * 32768 / max_abs_value_y
for i in range(len(data_2d)):
    data_2d[i][0] = int(data_2d[i][0] * x_MF) + 32768
    data_2d[i][1] = int(data_2d[i][1] * y_MF) + 32768

data_2d

hexa_pos_data = []

for row in data_2d:
    hexa_x = format(row[0], '04x')
    hexa_y = format(row[1], '04x')
    hexa_pos_data.append(hexa_x + hexa_y)

print(hexa_pos_data)
len(hexa_pos_data)

file_path = 'Unidirectional_Hexa_Raster_Scan_5000.txt'

with open(file_path, 'w') as file:
    for item in hexa_pos_data:
        file.write(item + '\n')
