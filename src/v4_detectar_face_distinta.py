"""
Script para detectar faces e emoções em vídeo completo usando YOLOv8-face e DeepFace.
Usa yolov8n-face.pt para detecção de faces e DeepFace para análise de emoções.
"""

# Importa Path para manipulação de caminhos de arquivos de forma multiplataforma
from pathlib import Path
# Importa tipos de anotação para melhor documentação e verificação de tipos
from typing import List, Tuple, Optional, Dict, Any
# Importa json para serialização de dados em formato JSON
import json
# Importa time para medir tempo de processamento
import time

# Importa OpenCV para processamento de vídeo e detecção de faces
import cv2
# Importa NumPy para manipulação de arrays e operações matemáticas
import numpy as np
# Importa tqdm para exibir barras de progresso durante o processamento
from tqdm import tqdm
# Importa math para cálculos matemáticos (distância euclidiana)
import math

# Import do YOLO (ultralytics) - biblioteca para detecção de objetos usando YOLOv8
try:
    # Tenta importar a classe YOLO do pacote ultralytics
    from ultralytics import YOLO  # type: ignore
except ImportError:
    # Se o pacote não estiver instalado, imprime mensagem de erro
    print("Erro: pacote 'ultralytics' não está instalado.")
    # Informa como instalar o pacote
    print("Instale com: pip install ultralytics")
    # Lança exceção para interromper a execução
    raise

# Import do DeepFace para análise de emoções
try:
    # Tenta importar DeepFace do pacote deepface
    from deepface import DeepFace  # type: ignore
except ImportError:
    # Se o pacote não estiver instalado, imprime mensagem de erro
    print("Erro: pacote 'deepface' não está instalado.")
    # Informa como instalar o pacote
    print("Instale com: pip install deepface")
    # Lança exceção para interromper a execução
    raise

# Determina o diretório base do projeto (pasta que contém src/)
# Obtém o diretório absoluto onde este script está localizado
SCRIPT_DIR = Path(__file__).parent.absolute()
# Se o diretório do script se chama "src", o projeto root é o pai, senão é o próprio diretório
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

# Cache global para o modelo YOLO (evita recarregar)
# Variável global que armazena o modelo YOLO carregado para reutilização entre frames
_YOLO_MODEL = None  # type: ignore[var-annotated]

# Emoções suportadas pelo DeepFace
# Lista com os nomes das emoções em inglês que o DeepFace pode detectar
EMOCOES_SUPORTADAS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
# Dicionário para traduzir emoções de inglês para português
EMOCOES_PT = {
    "angry": "Raiva",  # Traduz "angry" para "Raiva"
    "disgust": "Nojo",  # Traduz "disgust" para "Nojo"
    "fear": "Medo",  # Traduz "fear" para "Medo"
    "happy": "Feliz",  # Traduz "happy" para "Feliz"
    "sad": "Triste",  # Traduz "sad" para "Triste"
    "surprise": "Surpresa",  # Traduz "surprise" para "Surpresa"
    "neutral": "Neutro",  # Traduz "neutral" para "Neutro"
}


class FaceTracker:
    """
    Classe para rastrear faces entre frames, atribuindo IDs únicos e sequenciais.
    Usa combinação de posição e reconhecimento facial por embeddings para identificar
    a mesma pessoa mesmo quando ela reaparece em posições diferentes.
    Mantém um contador global que nunca diminui, garantindo que cada nova face
    receba um ID único mesmo quando faces anteriores desaparecem.
    """
    
    def __init__(self, max_distance: float = 100.0, max_frames_lost: int = 30, 
                 embedding_threshold: float = 0.6):
        """
        Inicializa o rastreador de faces.
        
        Parâmetros
        ----------
        max_distance: distância máxima em pixels para associar faces entre frames
        max_frames_lost: número máximo de frames que uma face pode estar perdida
                         antes de ser considerada desaparecida (aumentado para permitir reaparições)
        embedding_threshold: threshold de similaridade (0-1) para reconhecimento facial.
                             Valores mais altos são mais restritivos (ex: 0.7 = 70% de similaridade)
        """
        # Contador global de IDs (sempre incrementa, nunca diminui)
        self.next_id = 1
        # Dicionário de faces ativas: {face_id: {'box': (x, y, w, h), 'frames_lost': 0, 'embedding': array}}
        self.active_faces: Dict[int, Dict[str, Any]] = {}
        # Dicionário de faces perdidas recentemente (para tentar reencontrá-las): {face_id: {'embedding': array, 'frames_lost': 0}}
        self.lost_faces: Dict[int, Dict[str, Any]] = {}
        # Distância máxima para associar faces entre frames (em pixels)
        self.max_distance = max_distance
        # Número máximo de frames que uma face pode estar perdida
        self.max_frames_lost = max_frames_lost
        # Threshold de similaridade de embedding para reconhecimento facial
        self.embedding_threshold = embedding_threshold
    
    def _calculate_center(self, box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Calcula o centro de uma caixa delimitadora.
        
        Parâmetros
        ----------
        box: tupla (x, y, w, h) da caixa delimitadora
        
        Retorno
        -------
        Tupla (cx, cy) com as coordenadas do centro
        """
        x, y, w, h = box
        cx = x + w / 2.0
        cy = y + h / 2.0
        return (cx, cy)
    
    def _calculate_distance(self, box1: Tuple[int, int, int, int], 
                           box2: Tuple[int, int, int, int]) -> float:
        """
        Calcula a distância euclidiana entre os centros de duas caixas.
        
        Parâmetros
        ----------
        box1: primeira caixa (x, y, w, h)
        box2: segunda caixa (x, y, w, h)
        
        Retorno
        -------
        Distância euclidiana entre os centros das caixas
        """
        cx1, cy1 = self._calculate_center(box1)
        cx2, cy2 = self._calculate_center(box2)
        dx = cx2 - cx1
        dy = cy2 - cy1
        return math.sqrt(dx * dx + dy * dy)
    
    def update(self, boxes: List[Tuple[int, int, int, int]], 
               frame: Optional[np.ndarray] = None) -> List[Tuple[int, int, int, int, int]]:
        """
        Atualiza o rastreador com novas detecções e retorna as faces com seus IDs.
        Usa posição e reconhecimento facial por embeddings para associar faces.
        
        Parâmetros
        ----------
        boxes: lista de caixas detectadas no formato (x, y, w, h)
        frame: frame completo do vídeo (opcional, necessário para extração de embeddings)
        
        Retorno
        -------
        Lista de tuplas (x, y, w, h, face_id) com as caixas e seus IDs únicos
        """
        # Atualiza faces perdidas (incrementa contador)
        for face_id in list(self.lost_faces.keys()):
            self.lost_faces[face_id]['frames_lost'] += 1
            # Se foi perdida por muito tempo, remove da lista de perdidas
            if self.lost_faces[face_id]['frames_lost'] > self.max_frames_lost:
                del self.lost_faces[face_id]
        
        # Se não há faces detectadas, marca todas as faces ativas como perdidas
        if not boxes:
            # Move faces ativas para a lista de perdidas
            for face_id, face_data in list(self.active_faces.items()):
                if 'embedding' in face_data and face_data['embedding'] is not None:
                    self.lost_faces[face_id] = {
                        'embedding': face_data['embedding'],
                        'frames_lost': face_data.get('frames_lost', 0)
                    }
                del self.active_faces[face_id]
            return []
        
        # Extrai embeddings das novas faces detectadas (se frame foi fornecido)
        face_embeddings: List[Optional[np.ndarray]] = [None] * len(boxes)
        if frame is not None:
            for idx, (x, y, w, h) in enumerate(boxes):
                # Garante coordenadas válidas
                h_frame, w_frame = frame.shape[:2]
                x = max(0, min(x, w_frame - 1))
                y = max(0, min(y, h_frame - 1))
                w = min(w, w_frame - x)
                h = min(h, h_frame - y)
                
                if w > 0 and h > 0:
                    face_roi = frame[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        face_embeddings[idx] = _extract_face_embedding(face_roi)
        
        # Lista para marcar quais faces detectadas já foram associadas
        matched_detections = [False] * len(boxes)
        # Dicionário para armazenar os resultados: {face_id: box}
        tracked_boxes: Dict[int, Tuple[int, int, int, int]] = {}
        
        # ETAPA 1: Associação por posição (para faces que estão próximas)
        for face_id, face_data in list(self.active_faces.items()):
            best_match_idx = None
            best_distance = float('inf')
            old_box = face_data['box']
            
            # Procura a detecção mais próxima desta face ativa
            for idx, new_box in enumerate(boxes):
                if matched_detections[idx]:
                    continue
                
                distance = self._calculate_distance(old_box, new_box)
                
                # Se está dentro do limite de distância
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match_idx = idx
            
            # Se encontrou correspondência por posição
            if best_match_idx is not None:
                matched_detections[best_match_idx] = True
                new_box = boxes[best_match_idx]
                new_embedding = face_embeddings[best_match_idx]
                
                # Atualiza a face ativa com nova posição e embedding
                self.active_faces[face_id] = {
                    'box': new_box,
                    'frames_lost': 0,
                    'embedding': new_embedding if new_embedding is not None else face_data.get('embedding')
                }
                tracked_boxes[face_id] = new_box
            # Se não encontrou por posição, deixamos para a ETAPA 2 tentar por embeddings
        
        # ETAPA 2: Associação por embeddings (para faces que reapareceram ou mudaram de posição)
        # Tenta associar faces não associadas usando embeddings
        for idx, (box, embedding) in enumerate(zip(boxes, face_embeddings)):
            if matched_detections[idx] or embedding is None:
                continue
            
            best_match_id = None
            best_similarity = 0.0
            is_lost_face = False
            
            # Primeiro, procura nas faces ativas que não foram associadas por posição
            for active_id, active_data in self.active_faces.items():
                # Pula se esta face já foi associada (está em tracked_boxes)
                if active_id in tracked_boxes:
                    continue
                    
                active_embedding = active_data.get('embedding')
                if active_embedding is None:
                    continue
                
                similarity = _calculate_embedding_similarity(embedding, active_embedding)
                
                if similarity >= self.embedding_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = active_id
                    is_lost_face = False
            
            # Depois, procura nas faces perdidas recentemente
            for lost_id, lost_data in self.lost_faces.items():
                lost_embedding = lost_data.get('embedding')
                if lost_embedding is None:
                    continue
                
                similarity = _calculate_embedding_similarity(embedding, lost_embedding)
                
                if similarity >= self.embedding_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = lost_id
                    is_lost_face = True
            
            # Se encontrou correspondência por embeddings
            if best_match_id is not None:
                matched_detections[idx] = True
                
                if is_lost_face:
                    # Remove das perdidas e adiciona às ativas
                    self.active_faces[best_match_id] = {
                        'box': box,
                        'frames_lost': 0,
                        'embedding': embedding
                    }
                    del self.lost_faces[best_match_id]
                else:
                    # Atualiza a face ativa (não encontrada por posição, mas encontrada por embeddings)
                    self.active_faces[best_match_id] = {
                        'box': box,
                        'frames_lost': 0,
                        'embedding': embedding
                    }
                
                tracked_boxes[best_match_id] = box
        
        # Atualiza contadores de frames perdidos para faces ativas não associadas
        for face_id, face_data in list(self.active_faces.items()):
            # Se esta face não foi associada (não está em tracked_boxes)
            if face_id not in tracked_boxes:
                frames_lost = face_data.get('frames_lost', 0) + 1
                
                # Se excedeu o limite, move para lista de perdidas (se tem embedding)
                if frames_lost > self.max_frames_lost:
                    if 'embedding' in face_data and face_data['embedding'] is not None:
                        self.lost_faces[face_id] = {
                            'embedding': face_data['embedding'],
                            'frames_lost': 0  # Reset ao mover para lost_faces
                        }
                    del self.active_faces[face_id]
                else:
                    # Ainda dentro do limite, incrementa contador
                    self.active_faces[face_id]['frames_lost'] = frames_lost
        
        # ETAPA 3: Para detecções não associadas, cria novas faces com novos IDs
        for idx, box in enumerate(boxes):
            if not matched_detections[idx]:
                new_id = self.next_id
                self.next_id += 1
                embedding = face_embeddings[idx]
                
                # Adiciona à lista de faces ativas
                self.active_faces[new_id] = {
                    'box': box,
                    'frames_lost': 0,
                    'embedding': embedding
                }
                tracked_boxes[new_id] = box
        
        # Converte o dicionário em lista de tuplas (x, y, w, h, face_id)
        result = [(x, y, w, h, face_id) for face_id, (x, y, w, h) in tracked_boxes.items()]
        
        # Retorna ordenado por ID para consistência
        return sorted(result, key=lambda x: x[4])


def _load_yolo_model(model_path: Path) -> "YOLO":  # type: ignore[name-defined]
    """
    Carrega o modelo YOLO apenas uma vez e reaproveita nas próximas chamadas.
    """
    # Declara que vamos usar a variável global _YOLO_MODEL
    global _YOLO_MODEL

    # Se o modelo já foi carregado anteriormente, retorna o modelo em cache
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    # Verifica se o arquivo do modelo existe no caminho especificado
    if not model_path.exists():
        # Se não existir, lança exceção com mensagem informativa
        raise FileNotFoundError(
            f"Modelo YOLO não encontrado em '{model_path}'. "
            f"Baixe o yolov8n-face.pt e salve nesse caminho."
        )

    # Tenta carregar o modelo YOLO do arquivo
    try:
        # Carrega o modelo YOLO a partir do caminho do arquivo (converte Path para string)
        _YOLO_MODEL = YOLO(str(model_path))  # type: ignore[call-arg]
        # Informa que o modelo foi carregado com sucesso
        print(f"Modelo YOLO carregado de: {model_path}")
    except Exception as e:
        # Se houver erro, lança exceção com mensagem de erro
        raise RuntimeError(f"Erro ao carregar modelo YOLO: {e}")

    # Retorna o modelo carregado
    return _YOLO_MODEL


def _detect_faces_yolo(
    image_bgr: np.ndarray,  # Imagem no formato BGR (Blue, Green, Red) do OpenCV
    model_path: Path,  # Caminho para o arquivo do modelo YOLO
    min_conf: float = 0.30,  # Confiança mínima para aceitar uma detecção (30%)
) -> List[Tuple[int, int, int, int]]:  # Retorna lista de tuplas (x, y, largura, altura)
    """
    Detecta faces utilizando YOLOv8-face.

    Retorna lista de caixas no formato (x, y, w, h) em pixels.
    """
    # Carrega o modelo YOLO (usa cache se já estiver carregado)
    model = _load_yolo_model(model_path)

    # YOLO aceita numpy em BGR; definimos verbose=False para não poluir a saída
    # Executa a detecção na imagem usando o modelo YOLO
    results = model(image_bgr, verbose=False)

    # Inicializa lista vazia para armazenar as caixas delimitadoras das faces detectadas
    boxes: List[Tuple[int, int, int, int]] = []
    # Itera sobre os resultados retornados pelo YOLO
    for r in results:
        # Verifica se o resultado tem o atributo 'boxes' (caixas detectadas)
        if getattr(r, "boxes", None) is None:
            # Se não tiver, pula para o próximo resultado
            continue

        # Extrai as coordenadas das caixas no formato (x1, y1, x2, y2) e converte para numpy
        # Move os dados da GPU (cpu()) para a CPU e converte para array numpy
        xyxy = r.boxes.xyxy.cpu().numpy()
        # Extrai os valores de confiança das detecções e converte para numpy
        confs = r.boxes.conf.cpu().numpy()

        # Itera sobre cada caixa detectada junto com sua confiança
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            # Verifica se a confiança é menor que o mínimo aceito
            if float(conf) < min_conf:
                # Se for menor, ignora esta detecção e continua para a próxima
                continue
            # Converte as coordenadas de float para inteiro (coordenadas de pixel)
            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
            # Calcula a largura da caixa (diferença entre x2 e x1)
            w = x2_i - x1_i
            # Calcula a altura da caixa (diferença entre y2 e y1)
            h = y2_i - y1_i
            # Verifica se a largura e altura são válidas (maiores que zero)
            if w > 0 and h > 0:
                # Adiciona a caixa no formato (x, y, largura, altura) à lista
                boxes.append((x1_i, y1_i, w, h))

    # Retorna a lista de caixas detectadas
    return boxes


def _extract_face_embedding(face_roi: np.ndarray) -> Optional[np.ndarray]:
    """
    Extrai o embedding (vetor de características) de uma face usando DeepFace.
    
    Parâmetros
    ----------
    face_roi: região da imagem contendo o rosto (BGR)
    
    Retorno
    -------
    Array numpy com o embedding da face, ou None em caso de erro
    """
    try:
        # DeepFace espera RGB, então convertemos
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Extrai o embedding da face (representação numérica das características)
        result = DeepFace.represent(
            face_rgb,
            model_name="VGG-Face",  # Modelo para extração de características
            enforce_detection=False,  # Não força detecção (já detectamos com YOLO)
            silent=True,  # Suprime mensagens de log
        )
        
        # DeepFace pode retornar lista ou dict
        if isinstance(result, list):
            result = result[0]
        
        # Extrai o embedding (vetor de características)
        embedding = result.get("embedding")
        if embedding is not None:
            return np.array(embedding)
        return None
        
    except Exception:
        # Em caso de erro, retorna None
        return None


def _calculate_embedding_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calcula a similaridade entre dois embeddings usando similaridade de cosseno.
    
    Parâmetros
    ----------
    embedding1: primeiro embedding (vetor de características)
    embedding2: segundo embedding (vetor de características)
    
    Retorno
    -------
    Valor entre 0 e 1, onde 1 significa idêntico e 0 significa completamente diferente
    """
    # Normaliza os embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Evita divisão por zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Calcula similaridade de cosseno
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    # Garante que o valor está entre 0 e 1 (similaridade de cosseno varia entre -1 e 1)
    # Mas embeddings de faces geralmente produzem valores positivos
    return max(0.0, min(1.0, similarity))


def _detect_emotion(face_roi: np.ndarray) -> Tuple[str, float]:
    """
    Detecta a emoção em uma região de rosto usando DeepFace.

    Parâmetros
    ----------
    face_roi: região da imagem contendo o rosto (BGR)

    Retorno
    -------
    Tupla (emoção, confiança) onde emoção está em português
    """
    # Tenta detectar a emoção
    try:
        # DeepFace espera RGB, então convertemos
        # Converte a imagem de BGR (formato OpenCV) para RGB (formato DeepFace)
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

        # Analisa a emoção (enforce_detection=False para não falhar se já detectamos a face)
        # Chama o DeepFace para analisar a emoção na região do rosto
        result = DeepFace.analyze(
            face_rgb,  # Passa a imagem em RGB
            actions=["emotion"],  # Especifica que queremos apenas análise de emoção
            enforce_detection=False,  # Não força detecção de face (já detectamos com YOLO)
            silent=True,  # Suprime mensagens de log do DeepFace
        )

        # DeepFace pode retornar lista ou dict
        # Verifica se o resultado é uma lista
        if isinstance(result, list):
            # Se for lista, pega o primeiro elemento
            result = result[0]

        # Extrai a emoção predominante
        # Obtém o dicionário de emoções do resultado
        emotion_dict = result.get("emotion", {})
        # Se não houver dicionário de emoções, retorna neutro
        if not emotion_dict:
            return ("Neutro", 0.0)

        # Encontra a emoção com maior confiança
        # Usa max() para encontrar a emoção com maior valor (confiança) no dicionário
        emotion_en = max(emotion_dict.items(), key=lambda x: x[1])[0]
        # Obtém o valor de confiança da emoção predominante
        confidence = emotion_dict[emotion_en]

        # Converte para português
        # Busca a tradução da emoção no dicionário, ou capitaliza o nome em inglês se não encontrar
        emotion_pt = EMOCOES_PT.get(emotion_en, emotion_en.capitalize())

        # Retorna a emoção em português e a confiança como float
        return (emotion_pt, float(confidence))

    # Captura qualquer exceção que ocorra durante a detecção de emoção
    except Exception as e:
        # Em caso de erro, retorna neutro com confiança zero
        return ("Neutro", 0.0)


def detectar_faces_emocoes_video(
    caminho_video: Path,  # Caminho para o arquivo de vídeo de entrada
    caminho_saida: Optional[Path] = None,  # Caminho opcional para o vídeo de saída
    yolo_model_path: Optional[Path] = None,  # Caminho opcional para o modelo YOLO
    yolo_min_conf: float = 0.30,  # Confiança mínima para YOLO (30%)
    salvar_resultado: bool = True,  # Se True, salva o vídeo processado
    mostrar_progresso: bool = True,  # Se True, exibe barra de progresso
    processar_emocoes: bool = True,  # Se True, analisa emoções das faces detectadas
) -> Dict[str, Any]:  # Retorna dicionário com informações sobre o processamento
    """
    Detecta faces e emoções em um vídeo completo usando YOLOv8-face e DeepFace.

    Parâmetros
    ----------
    caminho_video: caminho para o vídeo de entrada
    caminho_saida: caminho para o vídeo de saída (padrão: src/Saida/v4/video_detectado.mp4)
    yolo_model_path: caminho para o modelo YOLOv8-face (.pt)
                     (padrão: PROJECT_ROOT/models/yolov8n-face.pt)
    yolo_min_conf: confiança mínima para aceitar detecções do YOLO
    salvar_resultado: se True, salva o vídeo processado
    mostrar_progresso: se True, exibe barra de progresso
    processar_emocoes: se True, analisa emoções das faces detectadas

    Retorno
    -------
    dict com informações sobre o processamento do vídeo
    """
    # Verifica se o arquivo de vídeo existe
    if not caminho_video.exists():
        # Se não existir, lança exceção
        raise FileNotFoundError(f"Vídeo não encontrado: {caminho_video}")

    # Define caminho do modelo YOLO
    # Se o caminho não foi especificado, usa o padrão
    if yolo_model_path is None:
        # Define o caminho padrão do modelo YOLO
        yolo_model_path = PROJECT_ROOT / "models" / "yolov8n-face.pt"

    # Define caminho de saída
    # Se o caminho de saída não foi especificado, cria um padrão
    if caminho_saida is None:
        # Define a pasta de saída padrão
        pasta_saida = PROJECT_ROOT / "src" / "Saida" / "v4"
        # Cria a pasta de saída (e pastas pai se necessário) se não existir
        pasta_saida.mkdir(parents=True, exist_ok=True)
        # Cria o nome do arquivo de saída adicionando "_detectado" antes da extensão
        nome_arquivo = caminho_video.stem + "_detectado" + caminho_video.suffix
        # Define o caminho completo do arquivo de saída
        caminho_saida = pasta_saida / nome_arquivo

    # Abre o vídeo
    # Cria um objeto VideoCapture para ler o vídeo (converte Path para string)
    cap = cv2.VideoCapture(str(caminho_video))
    # Verifica se o vídeo foi aberto com sucesso
    if not cap.isOpened():
        # Se não foi possível abrir, lança exceção
        raise ValueError(f"Não foi possível abrir o vídeo: {caminho_video}")

    # Obtém propriedades do vídeo
    # Obtém o FPS (frames por segundo) do vídeo e converte para inteiro
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Obtém a largura do vídeo em pixels e converte para inteiro
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # Obtém a altura do vídeo em pixels e converte para inteiro
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Obtém o número total de frames do vídeo e converte para inteiro
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Exibe informações sobre o vídeo
    print(f"📹 Vídeo: {caminho_video.name}")
    print(f"   Resolução: {largura}x{altura} | FPS: {fps} | Total de frames: {total_frames}")
    # Se a análise de emoções está ativada, informa
    if processar_emocoes:
        print(f"   Análise de emoções: ATIVADA")

    # Prepara o writer de vídeo
    # Define o codec de vídeo (mp4v é um codec comum para MP4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Inicializa variável para o objeto VideoWriter
    out = None
    # Se deve salvar o resultado
    if salvar_resultado:
        # Cria um objeto VideoWriter para escrever o vídeo processado
        # Parâmetros: caminho, codec, FPS, dimensões (largura, altura)
        out = cv2.VideoWriter(str(caminho_saida), fourcc, fps, (largura, altura))
        # Verifica se o VideoWriter foi criado com sucesso
        if not out.isOpened():
            # Se não foi possível criar, lança exceção
            raise RuntimeError(f"Não foi possível criar o vídeo de saída: {caminho_saida}")

    # Estatísticas
    # Inicializa contador para total de faces detectadas em todo o vídeo
    total_faces_detectadas = 0
    # Inicializa contador para frames que contêm pelo menos uma face
    frames_com_faces = 0
    # Inicializa contador para frames que não contêm faces
    frames_sem_faces = 0
    # Inicializa lista para armazenar o número de faces em cada frame
    faces_por_frame: List[int] = []
    # Inicializa dicionário para contar quantas vezes cada emoção foi detectada
    # Cria um dicionário com todas as emoções em português inicializadas com zero
    emocoes_detectadas: Dict[str, int] = {emo: 0 for emo in EMOCOES_PT.values()}
    # Registra o tempo de início do processamento
    tempo_inicio = time.time()

    # Inicializa o rastreador de faces para manter IDs únicos entre frames
    # Usa reconhecimento facial por embeddings para identificar a mesma pessoa
    # mesmo quando ela reaparece após passar atrás de objetos ou virar de costas
    face_tracker = FaceTracker(
        max_distance=100.0,  # Distância máxima em pixels para associação por posição
        max_frames_lost=30,  # Frames que uma face pode estar perdida antes de ser removida (aumentado)
        embedding_threshold=0.6  # Threshold de similaridade para reconhecimento facial (60%)
    )
    
    # Processa cada frame
    # Inicializa contador de frames processados
    frame_num = 0
    # Cria iterador com barra de progresso se solicitado, senão usa range simples
    iterator = tqdm(range(total_frames), desc="Processando frames", unit="frame") if mostrar_progresso else range(total_frames)

    # Itera sobre cada frame do vídeo
    for _ in iterator:
        # Lê o próximo frame do vídeo
        # ret indica se o frame foi lido com sucesso, frame contém a imagem do frame
        ret, frame = cap.read()
        # Se não conseguiu ler o frame (fim do vídeo ou erro)
        if not ret:
            # Interrompe o loop
            break

        # Incrementa o contador de frames processados
        frame_num += 1

        # Detecta faces no frame
        # Chama a função para detectar faces usando YOLO no frame atual
        boxes = _detect_faces_yolo(
            frame,  # Passa o frame atual
            model_path=yolo_model_path,  # Passa o caminho do modelo
            min_conf=yolo_min_conf,  # Passa a confiança mínima
        )

        # Atualiza o rastreador de faces e obtém as faces com seus IDs únicos
        # tracked_faces é uma lista de tuplas (x, y, w, h, face_id)
        # Passa o frame completo para permitir extração de embeddings faciais
        tracked_faces = face_tracker.update(boxes, frame=frame)

        # Atualiza estatísticas
        # Conta quantas faces foram detectadas neste frame
        num_faces = len(tracked_faces)
        # Adiciona o número de faces ao total acumulado
        total_faces_detectadas += num_faces
        # Adiciona o número de faces deste frame à lista
        faces_por_frame.append(num_faces)
        # Se encontrou pelo menos uma face neste frame
        if num_faces > 0:
            # Incrementa o contador de frames com faces
            frames_com_faces += 1
        else:
            # Se não encontrou faces, incrementa o contador de frames sem faces
            frames_sem_faces += 1

        # Desenha retângulos nas faces detectadas e analisa emoções
        # Define a cor verde para os retângulos (formato BGR do OpenCV)
        cor = (0, 255, 0)  # verde para YOLO
        # Itera sobre cada face rastreada para desenhar retângulos, labels e emoções
        for x, y, w, h, face_id in tracked_faces:
            # Garante que as coordenadas estão dentro dos limites do frame
            # Garante que x não seja negativo (limite esquerdo)
            x = max(0, x)
            # Garante que y não seja negativo (limite superior)
            y = max(0, y)
            # Garante que a largura não ultrapasse o limite direito do frame
            w = min(w, largura - x)
            # Garante que a altura não ultrapasse o limite inferior do frame
            h = min(h, altura - y)

            # Extrai a região do rosto
            # Recorta a região da imagem que contém o rosto detectado
            face_roi = frame[y:y+h, x:x+w]

            # Detecta emoção se solicitado
            # Inicializa string vazia para o texto da emoção
            emotion_text = ""
            # Se deve processar emoções e a região do rosto não está vazia
            if processar_emocoes and face_roi.size > 0:
                # Chama a função para detectar a emoção na região do rosto
                emotion, confidence = _detect_emotion(face_roi)
                # Formata o texto da emoção com nome e confiança em porcentagem
                emotion_text = f"{emotion} ({confidence:.0f}%)"
                # Incrementa o contador da emoção detectada no dicionário de estatísticas
                emocoes_detectadas[emotion] = emocoes_detectadas.get(emotion, 0) + 1

            # Desenha retângulo
            # Desenha um retângulo ao redor da face detectada
            # Parâmetros: frame, ponto inicial (x,y), ponto final (x+w, y+h), cor, espessura
            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)

            # Desenha label com ID único da pessoa
            # Calcula a posição y do label (acima do retângulo, mínimo y=20)
            label_y = max(y - 10, 20)
            # Adiciona texto com o ID único da pessoa acima do retângulo
            cv2.putText(
                frame,  # Frame onde desenhar
                f"Pessoa {face_id}",  # Texto a ser desenhado (ID único da pessoa)
                (x, label_y),  # Posição do texto
                cv2.FONT_HERSHEY_SIMPLEX,  # Fonte a ser usada
                0.6,  # Escala da fonte
                cor,  # Cor do texto (mesma do retângulo)
                2,  # Espessura do texto
            )

            # Desenha emoção abaixo do retângulo
            # Se uma emoção foi detectada
            if emotion_text:
                # Adiciona texto com a emoção detectada abaixo do retângulo
                cv2.putText(
                    frame,  # Frame onde desenhar
                    emotion_text,  # Texto com a emoção e confiança
                    (x, y + h + 25),  # Posição do texto (25 pixels abaixo do retângulo)
                    cv2.FONT_HERSHEY_SIMPLEX,  # Fonte a ser usada
                    0.5,  # Escala da fonte (menor que o label da face)
                    (255, 255, 0),  # Cor amarela para a emoção (formato BGR)
                    2,  # Espessura do texto
                )

        # Salva o frame processado
        # Se deve salvar o resultado e o VideoWriter está disponível
        if salvar_resultado and out is not None:
            # Escreve o frame processado no vídeo de saída
            out.write(frame)

    # Finaliza
    # Libera o objeto VideoCapture (fecha o arquivo de vídeo)
    cap.release()
    # Se o VideoWriter foi criado
    if out is not None:
        # Libera o objeto VideoWriter (finaliza a escrita do vídeo)
        out.release()

    # Calcula o tempo total de processamento
    # Subtrai o tempo inicial do tempo atual para obter a duração
    tempo_total = time.time() - tempo_inicio

    # Calcula estatísticas finais
    # Calcula a média de faces por frame (total de faces dividido pelo número de frames)
    media_faces_por_frame = total_faces_detectadas / frame_num if frame_num > 0 else 0
    # Encontra o número máximo de faces em um único frame
    max_faces_em_frame = max(faces_por_frame) if faces_por_frame else 0
    # Encontra o número mínimo de faces em um único frame
    min_faces_em_frame = min(faces_por_frame) if faces_por_frame else 0

    # Calcula estatísticas de emoções
    # Soma todas as emoções detectadas para obter o total
    total_emocoes = sum(emocoes_detectadas.values())
    # Calcula o percentual de cada emoção em relação ao total
    # Cria um dicionário com o percentual de cada emoção (arredondado para 2 casas decimais)
    emocoes_percentuais = {
        emo: round((count / total_emocoes * 100), 2) if total_emocoes > 0 else 0.0
        for emo, count in emocoes_detectadas.items()
    }
    # Encontra a emoção que foi detectada mais vezes (emoção predominante)
    # Usa max() para encontrar a emoção com maior contagem
    emocao_predominante = max(emocoes_detectadas.items(), key=lambda x: x[1])[0] if total_emocoes > 0 else "N/A"

    # Cria dicionário com todos os resultados do processamento
    resultado = {
        "video_entrada": str(caminho_video),  # Caminho do vídeo de entrada
        "video_saida": str(caminho_saida) if salvar_resultado else None,  # Caminho do vídeo de saída (ou None)
        "resolucao": f"{largura}x{altura}",  # Resolução do vídeo (largura x altura)
        "fps": fps,  # Frames por segundo do vídeo
        "total_frames": frame_num,  # Número total de frames processados
        "total_faces_detectadas": total_faces_detectadas,  # Total de faces detectadas em todo o vídeo
        "frames_com_faces": frames_com_faces,  # Número de frames que contêm faces
        "frames_sem_faces": frames_sem_faces,  # Número de frames sem faces
        "media_faces_por_frame": round(media_faces_por_frame, 2),  # Média de faces por frame (2 casas decimais)
        "max_faces_em_frame": max_faces_em_frame,  # Máximo de faces em um frame
        "min_faces_em_frame": min_faces_em_frame,  # Mínimo de faces em um frame
        "tempo_processamento_segundos": round(tempo_total, 2),  # Tempo total de processamento em segundos
        "fps_processamento": round(frame_num / tempo_total, 2) if tempo_total > 0 else 0,  # FPS de processamento (frames processados por segundo)
        "detector": "yolo",  # Nome do detector usado
        "modelo": str(yolo_model_path),  # Caminho do modelo usado
        "min_conf": yolo_min_conf,  # Confiança mínima usada
        "processar_emocoes": processar_emocoes,  # Se a análise de emoções foi ativada
        "total_emocoes_detectadas": total_emocoes,  # Total de emoções detectadas
        "emocoes_detectadas": emocoes_detectadas,  # Dicionário com contagem de cada emoção
        "emocoes_percentuais": emocoes_percentuais,  # Dicionário com percentual de cada emoção
        "emocao_predominante": emocao_predominante,  # Emoção mais detectada no vídeo
    }

    # Retorna o dicionário com os resultados
    return resultado


def main():
    """Processa o vídeo da pasta src/Entrada/Activities.mp4"""
    # Define o caminho da pasta de entrada onde está o vídeo
    pasta_entrada = PROJECT_ROOT / "src" / "Entrada"
    # Define o caminho completo do vídeo de entrada
    video_entrada = pasta_entrada / "Activities.mp4"

    # Verifica se o vídeo existe
    if not video_entrada.exists():
        # Se não existir, informa e mostra os caminhos relevantes
        print(f"Vídeo não encontrado: {video_entrada}")
        print(f"Diretório atual do script: {SCRIPT_DIR}")
        print(f"Raiz do projeto detectada: {PROJECT_ROOT}")
        # Encerra a função
        return

    # Informa que o processamento está iniciando
    print("🎬 Iniciando análise de vídeo completo com YOLOv8-face e DeepFace\n")

    # Tenta processar o vídeo
    try:
        # Chama a função para detectar faces e emoções no vídeo
        resultado = detectar_faces_emocoes_video(
            caminho_video=video_entrada,  # Passa o caminho do vídeo
            yolo_min_conf=0.30,  # Define confiança mínima de 30%
            salvar_resultado=True,  # Salva o vídeo processado
            mostrar_progresso=True,  # Exibe barra de progresso
            processar_emocoes=True,  # Ativa análise de emoções
        )

        # Exibe resumo
        # Imprime linha separadora
        print("\n" + "=" * 60)
        # Título do resumo
        print("RESUMO DO PROCESSAMENTO")
        # Linha separadora
        print("=" * 60)
        # Mostra o nome do arquivo de vídeo de entrada
        print(f"Vídeo de entrada: {Path(resultado['video_entrada']).name}")
        # Mostra o nome do arquivo de vídeo de saída (ou N/A se não foi salvo)
        print(f"Vídeo de saída: {Path(resultado['video_saida']).name if resultado['video_saida'] else 'N/A'}")
        # Mostra resolução e FPS do vídeo
        print(f"Resolução: {resultado['resolucao']} | FPS: {resultado['fps']}")
        # Mostra o total de frames processados
        print(f"Total de frames processados: {resultado['total_frames']}")
        # Título da seção de estatísticas de detecção
        print(f"\n📊 Estatísticas de Detecção:")
        # Mostra o total de faces detectadas
        print(f"   Total de faces detectadas: {resultado['total_faces_detectadas']}")
        # Mostra quantos frames contêm faces
        print(f"   Frames com faces: {resultado['frames_com_faces']}")
        # Mostra quantos frames não contêm faces
        print(f"   Frames sem faces: {resultado['frames_sem_faces']}")
        # Mostra a média de faces por frame
        print(f"   Média de faces por frame: {resultado['media_faces_por_frame']}")
        # Mostra o máximo de faces em um frame
        print(f"   Máximo de faces em um frame: {resultado['max_faces_em_frame']}")
        # Mostra o mínimo de faces em um frame
        print(f"   Mínimo de faces em um frame: {resultado['min_faces_em_frame']}")

        # Se a análise de emoções foi ativada
        if resultado['processar_emocoes']:
            # Título da seção de estatísticas de emoções
            print(f"\n😊 Estatísticas de Emoções:")
            # Mostra o total de emoções detectadas
            print(f"   Total de emoções detectadas: {resultado['total_emocoes_detectadas']}")
            # Mostra a emoção predominante
            print(f"   Emoção predominante: {resultado['emocao_predominante']}")
            # Título da distribuição de emoções
            print(f"   Distribuição de emoções:")
            # Itera sobre as emoções ordenadas por frequência (mais frequente primeiro)
            for emo, count in sorted(resultado['emocoes_detectadas'].items(), key=lambda x: x[1], reverse=True):
                # Se a emoção foi detectada pelo menos uma vez
                if count > 0:
                    # Obtém o percentual desta emoção
                    percent = resultado['emocoes_percentuais'][emo]
                    # Mostra a emoção, quantidade e percentual
                    print(f"      {emo}: {count} ({percent}%)")

        # Título da seção de performance
        print(f"\n⏱️  Performance:")
        # Mostra o tempo total de processamento
        print(f"   Tempo de processamento: {resultado['tempo_processamento_segundos']}s")
        # Mostra o FPS de processamento (velocidade de processamento)
        print(f"   FPS de processamento: {resultado['fps_processamento']}")
        # Título da seção de configuração
        print(f"\n🔧 Configuração:")
        # Mostra qual detector foi usado
        print(f"   Detector: {resultado['detector']}")
        # Mostra o nome do modelo usado
        print(f"   Modelo: {Path(resultado['modelo']).name}")
        # Mostra a confiança mínima usada
        print(f"   Confiança mínima: {resultado['min_conf']}")
        # Mostra se a análise de emoções foi ativada
        print(f"   Análise de emoções: {'Sim' if resultado['processar_emocoes'] else 'Não'}")

        # Salva estatísticas em JSON
        # Define a pasta de saída
        pasta_saida = PROJECT_ROOT / "src" / "Saida" / "v4"
        # Cria a pasta de saída se não existir
        pasta_saida.mkdir(parents=True, exist_ok=True)
        # Define o caminho do arquivo JSON de estatísticas
        arquivo_stats = pasta_saida / "video_detection_emotions_stats.json"
        # Escreve o arquivo JSON com os resultados
        # json.dumps converte o dicionário em string JSON formatada
        # indent=2 formata com indentação de 2 espaços
        # ensure_ascii=False permite caracteres não-ASCII (acentos, etc)
        arquivo_stats.write_text(
            json.dumps(resultado, indent=2, ensure_ascii=False),
            encoding="utf-8"  # Especifica codificação UTF-8
        )
        # Informa onde o arquivo foi salvo
        print(f"\n💾 Estatísticas salvas em: {arquivo_stats}")

        # Se o vídeo foi salvo
        if resultado['video_saida']:
            # Informa onde o vídeo foi salvo
            print(f"✅ Vídeo processado salvo em: {resultado['video_saida']}")

    # Captura qualquer exceção que ocorra durante o processamento
    except Exception as e:
        # Informa o erro ocorrido
        print(f"\n❌ Erro ao processar vídeo: {e}")
        # Relança a exceção para que o erro seja visível
        raise


# Verifica se o script está sendo executado diretamente (não importado como módulo)
if __name__ == "__main__":
    # Se estiver sendo executado diretamente, chama a função main
    main()
