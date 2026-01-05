# Tech Challenge IADT - Fase 4: Sistema de Visão Computacional

Este projeto entrega uma aplicação completa de análise de vídeo com:
- reconhecimento facial
- análise de expressões emocionais
- detecção de atividades
- geração de resumo automático

## Requirements
- Python 3.10+
- SO: Windows, Linux ou macOS
- OpenCV com codecs de vídeo padrão
- Modelos YOLO em `models/`
- Vídeo de entrada em `src/Entrada/Activities.mp4`
- Fotos conhecidas em `src/Entrada/conhecidos/`

Dependências estão em `requirements.txt` (com comentários por script).

## Instalação

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

### Execução completa (recomendado)
```bash
python src/v9_final.py
```

### Parte técnica / bibliotecas e ferramentas
OpenCV - base para ler vídeos, cortar frames, desenhar caixas e salvar resultados. 
NumPy - cálculos e manipulação de arrays de imagem e embeddings. 
tqdm - acompanhar progresso em vídeo longo. 
pathlib - caminhos organizados. 
json e time - estatísticas e medir tempo. 
math é usado - distâncias e geometria. 
A biblioteca ultralytics - carrega os modelos YOLO.

Modelos:
YOLOv8n-face - rápido e robusto para rosto em vídeo real, com ângulos diferentes e iluminação variada. (Eu comparei com MediaPipe e Haar, mas o YOLO foi o mais consistente.)
YOLOv11n - para detectar corpo e movimento geral.
YOLOv11 pose - fornece keypoints do corpo e permite classificar atividades.
DeepFace - para reconhecimento e emoção, já vem pronto com embeddings e análise de emoção.
MediaPipe e Haar ficaram apenas como comparação inicial.




Notas sobre os scripts por versão (contexto e resultados):

Script auxiliar: `src/face_detection.py`. Centraliza o fallback de detecção com MediaPipe e Haar. Usei no início para comparar detectores e validar a escolha do YOLO. O MediaPipe teve desempenho razoável, mas ainda perde para o YOLO em qualidade quando o rosto está pequeno ou com oclusão. O Haar é bem rápido, mas erra muito e gera falsos positivos, especialmente com ruído. Por isso ficou como último fallback e quase não foi usado nas versões finais.

Script v1: `src/v1_detectar_face_imagens.py`. Calibro com imagens em `src/Entrada/calibrar`. O script tenta YOLO primeiro, depois MediaPipe e depois Haar. As caixas usam cores: verde para YOLO, azul para MediaPipe e vermelho para Haar. O objetivo era decidir, com base em evidências, qual detector usar no resto do projeto. O resultado foi claro: o verde do YOLO foi o melhor, com caixas estáveis e menos erro. O vermelho do Haar foi ruim, com detecções falsas. Por isso continuei com o que salva em verde e deixei o vermelho apenas como último recurso. Essa foi a base da mudança depois.

Script v2: `src/v2_detectar_face_video_completo.py`. Removi o fallback e usei só o YOLOv8n-face no vídeo inteiro. O objetivo foi medir desempenho real e criar estatísticas base. Gero um vídeo com caixas e um JSON com total de faces e média por frame. O resultado foi estável, com milhares de faces detectadas e média consistente por frame. Isso confirmou que o modelo principal estava funcionando bem e justificou continuar com o YOLO como detector único.

Script v3: `src/v3_detectar_face_video_completo.py`. Adicionei o DeepFace para emoção. O fluxo é: YOLO detecta a face, recorto o rosto e chamo o `DeepFace.analyze` para classificar emoção. Escolhi o DeepFace porque já vem treinado e entrega várias emoções com probabilidades. O custo é que o processamento fica mais lento. O resultado foi uma distribuição coerente, com predomínio de Triste e outras emoções como Neutro e Feliz aparecendo. Mantive esse passo porque a emoção agrega valor no resumo final, mesmo aumentando o tempo.

Script v4: `src/v4_detectar_face_distinta.py`. Mudei o foco para rastreamento e identidade persistente. Criei um rastreador de faces baseado em embeddings do `DeepFace.represent`, comparando por similaridade de cosseno, com limiar de associação e tolerância de frames perdidos. Fiz essa alteração porque só detectar em cada frame cria duplicidade e instabilidade. O resultado foram IDs mais consistentes e base melhor para relatórios temporais. Essa foi uma mudança importante porque melhora a narrativa por pessoa.

Script v5: `src/v5_salvar_rostos.py`. Foquei em montar uma base de rostos. Salvo a primeira imagem de cada pessoa rastreada e depois salvo de tempos em tempos. Adicionei margem ao redor do rosto para pegar mais contexto. O resultado foi uma pasta com dezenas de imagens e pessoas diferentes. Isso alimenta o reconhecimento, porque sem essa base o reconhecimento ficaria limitado. Mantive essa estratégia porque o ganho foi grande.

Script v6: `src/v6_reconhecendo.py`. Aqui entra reconhecimento facial de fato. Leio imagens rotuladas em `src/Entrada/conhecidos`, gero embeddings com DeepFace e comparo com as faces detectadas no vídeo. Coloquei um limiar mais permissivo no início, em 0.6 de similaridade, para reconhecer mais pessoas, mas isso aumenta risco de falso positivo. O resultado foi que consegui rotular pessoas conhecidas no vídeo, mas notei que precisava ajustar limiares para melhorar confiabilidade. Essa observação leva para as mudanças de v8 e v9.

Script v7: `src/v7_detectar_movimentos.py`. Adicionei atividades. Usei o modelo YOLOv11 pose para keypoints e classifiquei atividades simples como parado, andando, correndo, sentado ou gesticulando. Também usei o YOLOv11n para reforçar a detecção de movimento do corpo. Essa alteração foi feita para atender o requisito de analisar atividades humanas. O resultado é que consigo adicionar uma camada de contexto por pessoa, mesmo com classificação simples.

Script v8: `src/v8_anomalias.py`. Criei a ideia de anomalias. Observei histórico de movimento e sinalizei quando alguém foge do padrão. Nessa versão aumentei o limiar de reconhecimento para 0.91, para evitar chamar pessoas erradas. Isso é uma alteração importante: ao aumentar o limiar perco alguns reconhecimentos, mas ganho em confiabilidade. O resultado ficou mais coerente para falar de eventos fora do padrão.

Script v9: `src/v9_final.py`. Versão final que junta tudo. Tenho detecção facial com YOLOv8n-face, rastreamento, reconhecimento com DeepFace, emoção, atividades com YOLO pose e anomalias. O script gera três saídas: vídeo final, JSON de estatísticas e um relatório TXT. Ajustei parâmetros para estabilidade, como `embedding_threshold` em 0.75 e `recognition_threshold` em 0.70, além de usar histórico de embeddings para suavizar flutuações. Essa mudança foi feita porque nomes trocavam rápido entre frames. O resultado final é um resumo automático por pessoa, com período de aparição, emoção predominante, atividade e anomalias. Essa é a entrega completa do projeto.

### Execuções por versão
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

## Evolução por versão

### v1 - Detecção em imagens
- Detecta faces em imagens estáticas.
- Suporte a YOLO, MediaPipe e Haar Cascade (modo auto).

### v2 - Detecção em vídeo
- Processa vídeo frame a frame.
- Estatísticas básicas do vídeo.

### v3 - Detecção + emoções
- Integra DeepFace para análise de emoções.
- Estatísticas de distribuição de emoções.

### v4 - Rastreamento com IDs
- Rastreia faces ao longo do tempo.
- Atribui IDs consistentes por pessoa.

### v5 - Extração de rostos
- Salva imagens das faces detectadas.
- Cria base inicial de rostos.

### v6 - Reconhecimento facial
- Carrega rostos conhecidos de `src/Entrada/conhecidos`.
- Mostra nomes no vídeo quando reconhecidos.

### v7 - Atividades e movimentos
- Adiciona detecção de atividades via YOLO pose.

### v8 - Anomalias
- Detecta movimentos anormais por pessoa.
- Gera estatísticas de anomalias.

### v9 - Resumo final
- Consolida detecção, reconhecimento, atividades e anomalias.
- Gera relatório TXT com resumo automático.

## v9_final: o que faz
- Carrega o vídeo de entrada.
- Carrega fotos conhecidas e cria embeddings.
- Detecta faces com YOLO.
- Rastreia pessoas e reconhece nomes.
- Analisa emoções por face com DeepFace.
- Detecta atividades/pose e anomalias.
- Desenha caixas, labels e dados no vídeo.
- Gera estatísticas JSON.
- Gera relatório TXT resumido.

## v9_final: principais blocos de código

### Importação e validação de dependências
```python
try:
    from ultralytics import YOLO
except ImportError:
    print("Erro: pacote 'ultralytics' não está instalado.")
    print("Instale com: pip install ultralytics")
    raise

try:
    from deepface import DeepFace
except ImportError:
    print("Erro: pacote 'deepface' não está instalado.")
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

### Geração de relatório (trecho)
```python
linhas = [
    "RELATÓRIO AUTOMÁTICO - TECH CHALLENGE FASE 4",
    "",
    f"Vídeo de entrada: {Path(resultado.get('video_entrada', 'N/A')).name}",
    f"Vídeo de saída: {Path(resultado.get('video_saida', 'N/A')).name if resultado.get('video_saida') else 'N/A'}",
    "",
    f"Total de frames analisados: {total_frames}",
    f"Número de anomalias detectadas: {total_anomalias}",
    f"Frames com anomalias: {frames_com_anomalias}",
    "",
    f"Atividade predominante: {resultado.get('atividade_predominante', 'N/A')}",
    "Distribuição de atividades:",
]
arquivo_txt.write_text("\n".join(linhas), encoding="utf-8")
```

## Saídas geradas
- Vídeo processado: `src/Saida/v9/Activities_detectado.mp4`
- Estatísticas: `src/Saida/v9/video_detection_anomalias_stats.json`
- Relatório: `src/Saida/v9/relatorio_resumo.txt`

## Observações e limitações
- Performance depende do hardware (processamento pode ser lento).
- DeepFace pode baixar modelos na primeira execução.
- Modelos YOLO precisam estar em `models/`.
- Mais detalhes de entrega estão em `ENTREGA.md`.
