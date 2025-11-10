import argparse
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import csv
import os
import numpy as np
import cv2
from tqdm import tqdm

# === Argomenti da riga di comando ===
parser = argparse.ArgumentParser(description='Esegui simulazioni CoppeliaSim e salva log e immagini.')
parser.add_argument('--N', type=int, default=1, help='Numero di esecuzioni da effettuare')
parser.add_argument('--log', type=str, default='runtime_log', help='Nome del file CSV per il log')
args = parser.parse_args()

N = args.N
log_filename = args.log + ".csv"

# === Connessione a CoppeliaSim ===
client = RemoteAPIClient()
sim = client.getObject('sim')

# === Prepara header CSV ===
sensor_headers = [f'sensor_{i}' for i in range(16)]
block_headers = [f'block_{i}_{axis}' for i in range(15) for axis in ['x', 'y', 'z']]  # massimo 20 blocchi
header = ['run', 'time', 'pioneer_x', 'pioneer_y', 'pioneer_z', 'goal_x', 'goal_y', 'goal_z'] + sensor_headers + block_headers
with open(log_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

time.sleep(5)  # Tempo per avvio sicuro

# === Ciclo di esecuzione ===
for run in tqdm(range(N), desc="Esecuzioni", unit="run"):
    sim.setBoolParam(sim.boolparam_display_enabled, False)

    sim.loadScene('C:/Users/anton/Documents/PhD/PotentialField_Sim/Simulazione_Threshold_variabile/potential_fields_sim_n.ttt')
    sim.startSimulation()

    pioneer = sim.getObject('/PioneerP3DX')
    goal = sim.getObject('/goal')
    camera = sim.getObject('/PioneerP3DX/Vision_sensor')

    goal_pos = sim.getObjectPosition(goal, -1)

    block_handles = []
    i = 0
    while True:
        try:
            h = sim.getObject(f'/ConcretBlock_{i}')
            block_handles.append(h)
            i += 1
        except Exception:
            break

    # Crea cartella per le immagini
    frame_dir = os.path.join('camera_frames', f'run_{run + 1}')
    os.makedirs(frame_dir, exist_ok=True)

    frame_id = 0
    t0 = time.time()

    while sim.getSimulationState() != sim.simulation_stopped:
        t = time.time() - t0
        pioneer_pos = sim.getObjectPosition(pioneer, -1)

        # Lettura sensori
        sensor_vals = []
        for i in range(16):
            sensor = sim.getObject(f'/PioneerP3DX/ultrasonicSensor[{i}]')
            detected, distance, *_ = sim.readProximitySensor(sensor)
            sensor_vals.append(distance if detected else -1.0)

        # Posizioni blocchi
        block_positions = []
        for h in block_handles:
            pos = sim.getObjectPosition(h, -1)
            block_positions.extend(pos)

        # Scrittura su CSV
        row = [run + 1, round(t, 2)] + pioneer_pos + goal_pos + sensor_vals + block_positions
        with open(log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Acquisizione immagine
        image, resolution = sim.getVisionSensorImg(camera)
        img = np.array(sim.unpackUInt8Table(image), dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        filename = os.path.join(frame_dir, f"frame_{frame_id:04d}.png")
        cv2.imwrite(filename, img)
        frame_id += 1



        sim.setFloatSignal('repulsion_threshold', 0.8) ### Modifica soglia di repulsione 

        
        time.sleep(0.1)

    print(f"-- Fine esecuzione {run + 1} --")