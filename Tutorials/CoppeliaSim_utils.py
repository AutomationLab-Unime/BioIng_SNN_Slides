import csv 
import numpy as np
import cv2 
import os

#log_filename deve essere essere il nome del file csv dove salvare i dati
#save_path deve essere la cartella dove salvare i dati
log_filename = os.path.join(save_path, "log.csv")

def init_csv(log_filename, block_positions):
    #Dentro header scrivo i nomi delle colonne del file csv
    header = ['run', 'time', 'pioneer_x', 'pioneer_y', 'goal_x', 'goal_y', 'left_velocity', 'right_velocity'] + [f'sensor_{i}' for i in range(16)] + [f"'obstacle_{i}_x" for i in range(len(block_positions))] + [f"obstacle_{i}_y" for i in range(len(block_positions))]
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)    

def update_csv( log_filename, run, frame_id, pioneer_position, goal_position, v_l, v_r, sensors_vals, block_positions_array):
    #dentro row deve esserci una lista di valori da salvare su csv (una sola riga)
    row = [run + 1, frame_id, pioneer_position[0], pioneer_position[1], goal_position[0], goal_position[1], v_l.item(), v_r.item()] + sensors_vals + list(block_positions_array[:,0]) + list(block_positions_array[:,1])
    with open(log_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def init_sensors():
    #creo una lista di handle dei sensori di prossimità del Pioneer
    sensors = []
    for i_s in range(16):
        sensors.append(sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i_s}]'))
    return sensors

def save_frame(frame_dir, frame_id):
    #camera è l'handle della camere ottenuto con sim.getObject()
    image, resolution = sim.getVisionSensorImg(camera)
    img = np.array(sim.unpackUInt8Table(image), dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
    img = np.flipud(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #frame_dir è la cartella dove salvare le immagini
    filename = os.path.join(frame_dir, f"frame_{frame_id:04d}.png")
    cv2.imwrite(filename, img)

def read_proximity_sensors( sensors):
    #Creo una lista di valori letti dai sensori di prossimità (-1.0 se non leggo nulla, altrimenti la distanza letta)
    sensors_vals = []
    for sensor in sensors:
        detected, distance, *_ = sim.readProximitySensor(sensor)
        sensors_vals.append(distance if detected else -1.0)
    return sensors_vals