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

### Parte tecnica / bibliotecas e ferramentas
OpenCV - base para ler videos, cortar frames, desenhar caixas e salvar resultados. 
NumPy - calculos e manipulacao de arrays de imagem e embeddings. 
tqdm - acompanhar progresso em video longo. 
pathlib - caminhos organizados. 
json e time - estatisticas e medir tempo. 
math e usado - distancias e geometria. 
A biblioteca ultralytics - carrega os modelos YOLO.

Modelos:
YOLOv8n-face, - rapido bem robusto para rosto em video real, com angulos diferentes e iluminacao variada. (Eu comparei com MediaPipe e Haar, mas o YOLO foi o mais consistente.)
YOLOv11n, para detectar corpo e movimento geral 
YOLOv11 pose, -fornece keypoints do corpo e permite classificar atividades. 
DeepFacePara reconhecimento e emocao, ja vem pronto com embeddings e analise de emocao. 
MediaPipe e Haar ficaram apenas como comparacao inicial.




Notas sobre os scripts por versao (contexto e resultados):

Script auxiliar: `src/face_detection.py`. Centraliza o fallback de deteccao com MediaPipe e Haar. Usei no inicio para comparar detectores e validar a escolha do YOLO. O MediaPipe teve desempenho razoavel, mas ainda perde para o YOLO em qualidade quando o rosto esta pequeno ou com oclusao. O Haar e bem rapido, mas erra muito e gera falsos positivos, especialmente com ruido. Por isso ficou como ultimo fallback e quase nao foi usado nas versoes finais.

Script v1: `src/v1_detectar_face_imagens.py`. Calibro com imagens em `src/Entrada/calibrar`. O script tenta YOLO primeiro, depois MediaPipe e depois Haar. As caixas usam cores: verde para YOLO, azul para MediaPipe e vermelho para Haar. O objetivo era decidir, com base em evidencias, qual detector usar no resto do projeto. O resultado foi claro: o verde do YOLO foi o melhor, com caixas estaveis e menos erro. O vermelho do Haar foi ruim, com deteccoes falsas. Por isso continuei com o que salva em verde e deixei o vermelho apenas como ultimo recurso. Essa foi a base da mudanca depois.

Script v2: `src/v2_detectar_face_video_completo.py`. Removi o fallback e usei so o YOLOv8n-face no video inteiro. O objetivo foi medir desempenho real e criar estatisticas base. Gero um video com caixas e um JSON com total de faces e media por frame. O resultado foi estavel, com milhares de faces detectadas e media consistente por frame. Isso confirmou que o modelo principal estava funcionando bem e justificou continuar com o YOLO como detector unico.

Script v3: `src/v3_detectar_face_video_completo.py`. Adicionei o DeepFace para emocao. O fluxo e: YOLO detecta a face, recorto o rosto e chamo o `DeepFace.analyze` para classificar emocao. Escolhi o DeepFace porque ja vem treinado e entrega varias emocoes com probabilidades. O custo e que o processamento fica mais lento. O resultado foi uma distribuicao coerente, com predominio de Triste e outras emocoes como Neutro e Feliz aparecendo. Mantive esse passo porque a emocao agrega valor no resumo final, mesmo aumentando o tempo.

Script v4: `src/v4_detectar_face_distinta.py`. Mudei o foco para rastreamento e identidade persistente. Criei um rastreador de faces baseado em embeddings do `DeepFace.represent`, comparando por similaridade de cosseno, com limiar de associacao e tolerancia de frames perdidos. Fiz essa alteracao porque so detectar em cada frame cria duplicidade e instabilidade. O resultado foram IDs mais consistentes e base melhor para relatorios temporais. Essa foi uma mudanca importante porque melhora a narrativa por pessoa.

Script v5: `src/v5_salvar_rostos.py`. Foquei em montar uma base de rostos. Salvo a primeira imagem de cada pessoa rastreada e depois salvo de tempos em tempos. Adicionei margem ao redor do rosto para pegar mais contexto. O resultado foi uma pasta com dezenas de imagens e pessoas diferentes. Isso alimenta o reconhecimento, porque sem essa base o reconhecimento ficaria limitado. Mantive essa estrategia porque o ganho foi grande.

Script v6: `src/v6_reconhecendo.py`. Aqui entra reconhecimento facial de fato. Leio imagens rotuladas em `src/Entrada/conhecidos`, gero embeddings com DeepFace e comparo com as faces detectadas no video. Coloquei um limiar mais permissivo no inicio, em 0.6 de similaridade, para reconhecer mais pessoas, mas isso aumenta risco de falso positivo. O resultado foi que consegui rotular pessoas conhecidas no video, mas notei que precisava ajustar limiares para melhorar confiabilidade. Essa observacao leva para as mudancas de v8 e v9.

Script v7: `src/v7_detectar_movimentos.py`. Adicionei atividades. Usei o modelo YOLOv11 pose para keypoints e classifiquei atividades simples como parado, andando, correndo, sentado ou gesticulando. Tambem usei o YOLOv11n para reforcar a deteccao de movimento do corpo. Essa alteracao foi feita para atender o requisito de analisar atividades humanas. O resultado e que consigo adicionar uma camada de contexto por pessoa, mesmo com classificacao simples.

Script v8: `src/v8_anomalias.py`. Criei a ideia de anomalias. Observei historico de movimento e sinalizei quando alguem foge do padrao. Nessa versao aumentei o limiar de reconhecimento para 0.91, para evitar chamar pessoas erradas. Isso e uma alteracao importante: ao aumentar o limiar perco alguns reconhecimentos, mas ganho em confiabilidade. O resultado ficou mais coerente para falar de eventos fora do padrao.

Script v9: `src/v9_final.py`. Versao final que junta tudo. Tenho deteccao facial com YOLOv8n-face, rastreamento, reconhecimento com DeepFace, emocao, atividades com YOLO pose e anomalias. O script gera tres saidas: video final, JSON de estatisticas e um relatorio TXT. Ajustei parametros para estabilidade, como `embedding_threshold` em 0.75 e `recognition_threshold` em 0.70, alem de usar historico de embeddings para suavizar flutuacoes. Essa mudanca foi feita porque nomes trocavam rapido entre frames. O resultado final e um resumo automatico por pessoa, com periodo de aparicao, emocao predominante, atividade e anomalias. Essa e a entrega completa do projeto.

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
