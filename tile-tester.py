import subprocess
import time

tile_sizes = [
    (8, 0),
    (8, 4),
    (8, 8),
    (8, 16),
    (8, 24),
    (8, 32),
    (16, 0),
    (16, 4),
    (16, 8),
    (16, 16),
    (16, 32),
    (16, 48),
    (32, 0),
    (32, 4),
    (32, 8),
    (32, 16),
    (32, 32),
    (64, 0),
    (64, 4),
    (64, 8),
    (64, 16),
    (64, 32),
    (64, 64),
    (128, 0),
    (128, 4),
    (128, 8),
    (128, 16),
    (128, 32),
    (128, 64),
    (128, 128)
]

def queue_avail():
    cmd = subprocess.run(['squeue', '-u', 'shirsch', '-r', '-h'], capture_output=True, encoding="utf-8")
    lines = cmd.stdout.split('\n')
    if len(lines) >= 6:
        return False
    else:
        return True

for tile in tile_sizes:
    print(f"Queueing process {tile[0]} {tile[1]}")
    subprocess.run(['sbatch', 'run.sh', str(tile[0]), str(tile[1])])
    while not queue_avail():
        time.sleep(5)
