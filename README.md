# Tech Challenge IADT - Fase 4: Sistema de Visao Computacional

Este projeto entrega uma aplicacao completa de analise de video com:
- reconhecimento facial
- analise de expressoes emocionais
- deteccao de atividades
- geracao de resumo automatico

## Requirements
- Python 3.10+
- SO: Windows, Linux ou macOS
- OpenCV com codecs de video padrao
- Modelos YOLO em `models/`
- Video de entrada em `src/Entrada/Activities.mp4`
- Fotos conhecidas em `src/Entrada/conhecidos/`

Dependencias estao em `requirements.txt` (com comentarios por script).

## Instalacao

Windows (PowerShell):
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/download_yolo_model.py
```

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/download_yolo_model.py
```

## Como executar

### Execucao completa (recomendado)
```bash
python src/v9_final.py
```

### Execucoes por versao
```bash
python src/v1_detectar_face_imagens.py
python src/v2_detectar_face_video_completo.py
python src/v3_detectar_face_video_completo.py
python src/v4_detectar_face_distinta.py
python src/v5_salvar_rostos.py
python src/v6_reconhecendo.py
python src/v7_detectar_movimentos.py
python src/v8_anomalias.py
python src/v9_final.py
```

## Estrutura do projeto
```
TC4_Visao/
  doc/
  models/
    yolov8n-face.pt
    yolo11n-pose.pt
    yolov11n.pt
  src/
    Entrada/
      Activities.mp4
      calibrar/
      conhecidos/
    Saida/
      v9/
    v1_detectar_face_imagens.py
    v2_detectar_face_video_completo.py
    v3_detectar_face_video_completo.py
    v4_detectar_face_distinta.py
    v5_salvar_rostos.py
    v6_reconhecendo.py
    v7_detectar_movimentos.py
    v8_anomalias.py
    v9_final.py
    download_yolo_model.py
  requirements.txt
  README.md
  ENTREGA.md
```

## Evolucao por versao

### v1 - Deteccao em imagens
- Detecta faces em imagens estaticas.
- Suporte a YOLO, MediaPipe e Haar Cascade (modo auto).

### v2 - Deteccao em video
- Processa video frame a frame.
- Estatisticas basicas do video.

### v3 - Deteccao + emocoes
- Integra DeepFace para analise de emocoes.
- Estatisticas de distribuicao de emocoes.

### v4 - Rastreamento com IDs
- Rastreia faces ao longo do tempo.
- Atribui IDs consistentes por pessoa.

### v5 - Extracao de rostos
- Salva imagens das faces detectadas.
- Cria base inicial de rostos.

### v6 - Reconhecimento facial
- Carrega rostos conhecidos de `src/Entrada/conhecidos`.
- Mostra nomes no video quando reconhecidos.

### v7 - Atividades e movimentos
- Adiciona deteccao de atividades via YOLO pose.

### v8 - Anomalias
- Detecta movimentos anormais por pessoa.
- Gera estatisticas de anomalias.

### v9 - Resumo final
- Consolida deteccao, reconhecimento, atividades e anomalias.
- Gera relatorio TXT com resumo automatico.

## v9_final: o que faz
- Carrega o video de entrada.
- Carrega fotos conhecidas e cria embeddings.
- Detecta faces com YOLO.
- Rastreia pessoas e reconhece nomes.
- Analisa emocoes por face com DeepFace.
- Detecta atividades/pose e anomalias.
- Desenha caixas, labels e dados no video.
- Gera estatisticas JSON.
- Gera relatorio TXT resumido.

## v9_final: principais blocos de codigo

### Importacao e validacao de dependencias
```python
try:
    from ultralytics import YOLO
except ImportError:
    print("Erro: pacote 'ultralytics' nao esta instalado.")
    print("Instale com: pip install ultralytics")
    raise

try:
    from deepface import DeepFace
except ImportError:
    print("Erro: pacote 'deepface' nao esta instalado.")
    print("Instale com: pip install deepface")
    raise
```

### Classe de rastreamento (trecho)
```python
class FaceTracker:
    def _calculate_center(self, box):
        x, y, w, h = box
        cx = x + w / 2.0
        cy = y + h / 2.0
        return (cx, cy)

    def _calculate_distance(self, box1, box2):
        c1 = self._calculate_center(box1)
        c2 = self._calculate_center(box2)
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])
```

### Loop principal de processamento (trecho)
```python
iterator = tqdm(range(total_frames), desc="Processando frames", unit="frame") if mostrar_progresso else range(total_frames)
for _ in iterator:
    ret, frame = cap.read()
    if not ret:
        break

    boxes = _detect_faces_yolo(
        frame,
        model_path=yolo_model_path,
        min_conf=yolo_min_conf,
    )
    tracked_faces = face_tracker.update(boxes, frame=frame)
    pose_detections = _detect_pose_yolo(
        frame,
        model_path=yolo_pose_model_path,
        min_conf=yolo_pose_min_conf,
    )
    pose_by_face = _match_faces_to_poses(tracked_faces, pose_detections)
```

### Geracao de relatorio (trecho)
```python
linhas = [
    "RELATORIO AUTOMATICO - TECH CHALLENGE FASE 4",
    "",
    f"Video de entrada: {Path(resultado.get('video_entrada', 'N/A')).name}",
    f"Video de saida: {Path(resultado.get('video_saida', 'N/A')).name if resultado.get('video_saida') else 'N/A'}",
    "",
    f"Total de frames analisados: {total_frames}",
    f"Numero de anomalias detectadas: {total_anomalias}",
    f"Frames com anomalias: {frames_com_anomalias}",
    "",
    f"Atividade predominante: {resultado.get('atividade_predominante', 'N/A')}",
    "Distribuicao de atividades:",
]
arquivo_txt.write_text("\n".join(linhas), encoding="utf-8")
```

## Saidas geradas
- Video processado: `src/Saida/v9/Activities_detectado.mp4`
- Estatisticas: `src/Saida/v9/video_detection_anomalias_stats.json`
- Relatorio: `src/Saida/v9/relatorio_resumo.txt`

## Observacoes e limitacoes
- Performance depende do hardware (processamento pode ser lento).
- DeepFace pode baixar modelos na primeira execucao.
- Modelos YOLO precisam estar em `models/`.
- Mais detalhes de entrega estao em `ENTREGA.md`.
