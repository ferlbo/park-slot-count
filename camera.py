import cv2
import numpy
import json
import subprocess
import time
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(
        description='Checks how many parking slots are in use from the input video stream')

parser.add_argument(
        '-config',
        default='config.json',
        help='Path to json config file (default: %(default)s)')
args = parser.parse_args()

# Carrega arquivo de configurações
with open(args.config, 'r') as f:
    print('Loanding config.json ...')
    config = json.load(f)
    config_stream = config.get('stream', 'http://localhost:1984/api/stream.mp4?src=file')
    print('config_stream: ' + config_stream)
    config_output_stream = config.get('output_stream', '')
    print('config_output_stream: ' + config_output_stream)
    config_fps = int(config.get('fps', 1))
    print('config_fps: ' + str(config_fps))
    config_spots = config.get('spots')

# Inicializa o stream
print('Inicializing input stream ...')
cap = cv2.VideoCapture()
cap.open(config_stream)

# Pega resolução do video de entrada para passar para o ffmpeg
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
resolution = f'{width}x{height}'

# Mesmo que o vídeo tenha FPS alto, vamos limitar a 1 FPS no processamento
target_fps = 1
frame_interval = 1.0 / target_fps  # intervalo entre frames (em segundos)

ffmpeg = None
if config_output_stream:
    print('Inicializing output stream ...')
    # Comando ffmpeg para enviar o vídeo para o servidor RTSP
    ffmpeg_cmd = [
        'ffmpeg',
        '-hide_banner',
        '-v', 'error',
        '-re',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', resolution,
        '-r', str(target_fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-tune', 'zerolatency',
        '-f', 'rtsp',
        '-rtsp_transport', 'tcp',
        'rtsp://localhost:8554/output_stream'
    ]

    # Inicializa o processo do ffmpeg
    ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

last_frame_time = time.time()
processed = 0
skipped = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print('\nStream ended...')
            break
        
        now = time.time()
        if now - last_frame_time >= frame_interval:
            last_frame_time = now
            time_tuple = time.localtime(now)
            time_string = time.strftime("%Y-%m-%d %H:%M:%S", time_tuple)
            print(skipped, 'skipped', end='')
            print('\r', time_string, sep='', end=' ', flush=True)

            # Processamento do frame
            frameCinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameTh = cv2.adaptiveThreshold(
                frameCinza, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
            )
            frameBlur = cv2.medianBlur(frameTh, 5)
            kernel = numpy.ones((3, 3), numpy.int8)
            frameDil = cv2.dilate(frameBlur, kernel)
            contador = 0

            print('[', end='', flush=True)
            # Analise das spots
            for spot in config_spots:
                mask = [tuple(p) for p in spot['mask']]  # Transforma listas [x, y] em tuplas (x, y)
                threshold = spot['threshold']

                mask_np = numpy.array([mask], dtype=numpy.int32)
                x, y, w, h = cv2.boundingRect(mask_np)
                mask_rel = mask_np - [x, y]

                # Criação da máscara no formato da vaga
                mask = numpy.zeros((h, w), dtype=numpy.uint8)
                cv2.fillPoly(mask, [mask_rel], 255)

                # Aplica a máscara ao frame dilatado
                recorte_ret = frameDil[y:y+h, x:x+w]
                recorte = cv2.bitwise_and(recorte_ret, recorte_ret, mask=mask)

                # Conta os pixels brancos
                qtPxBranco = cv2.countNonZero(recorte)

                # Exibe a contagem na imagem
                cv2.putText(
                    frame,
                    str(qtPxBranco),
                    (x + 5, y + h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 255),
                    3,
                )

                # Verifica se está ocupada
                if qtPxBranco < threshold:
                    cor = (0, 255, 0)  # Vaga livre
                    print(' ', end ='', flush=True)
                else:
                    cor = (0, 0, 255)  # Vaga ocupada
                    contador += 1
                    print('#', end='', flush=True)

                # Desenha o contorno da vaga
                cv2.polylines(frame, [mask_np], isClosed=True, color=cor, thickness=2)
            print(']', end='', flush=True)

            # Envia frame
            if ffmpeg is not None:
                ffmpeg.stdin.write(frame.tobytes())
            cv2.imwrite(f'/tmp/cam_output_{processed:04d}.jpg', frame)
            processed += 1

            print(f' {contador:02d}/{len(config_spots):02d} {time.time() - now:8.6f} {processed:8d} processed', end=' ', flush=True)
        else:
            skipped += 1
            #print('.', end='', flush=True)
except KeyboardInterrupt:
    print('\nUser requested exit')
except Exception as e:
    print('\nError code raise:')
    print(e)
finally:
    print('Releasing input stream...', end='')
    cap.release()
    print('done')
    if ffmpeg is not None:
        print('Releasing output stream...', end='')
        ffmpeg.stdin.close()
        ffmpeg.wait()
        print('done')
