"""
Script para detectar faces, reconhecer pessoas conhecidas, analisar emoções e identificar atividades humanas (RF3).
Usa yolov8n-face.pt para detecção facial, DeepFace para reconhecimento/emoções, MediaPipe Pose para classificar atividades,
e YOLOv11 para reforçar a detecção de movimentos do corpo (modelo padrão em models/yolov11n.pt).
Carrega rostos conhecidos de src/Entrada/conhecidos para fazer reconhecimento e grava resultados em src/Saida/v7.
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
_YOLO_MOVEMENT_MODEL: Tuple[Optional[Path], Optional["YOLO"]] = (None, None)
_YOLO_POSE_MODEL: Tuple[Optional[Path], Optional["YOLO"]] = (None, None)

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
    Suporta reconhecimento de pessoas conhecidas usando nomes.
    """
    
    def __init__(self, max_distance: float = 100.0, max_frames_lost: int = 30, 
                 embedding_threshold: float = 0.6, known_faces: Optional[Dict[str, List[np.ndarray]]] = None):
        """
        Inicializa o rastreador de faces.
        
        Parâmetros
        ----------
        max_distance: distância máxima em pixels para associar faces entre frames
        max_frames_lost: número máximo de frames que uma face pode estar perdida
                         antes de ser considerada desaparecida (aumentado para permitir reaparições)
        embedding_threshold: threshold de similaridade (0-1) para reconhecimento facial.
                             Valores mais altos são mais restritivos (ex: 0.7 = 70% de similaridade)
        known_faces: dicionário com faces conhecidas {nome: [embeddings]} para reconhecimento
        """
        # Contador global de IDs (sempre incrementa, nunca diminui)
        self.next_id = 1
        # Dicionário de faces ativas: {face_id: {'box': (x, y, w, h), 'frames_lost': 0, 'embedding': array, 'nome': str}}
        self.active_faces: Dict[int, Dict[str, Any]] = {}
        # Dicionário de faces perdidas recentemente: {face_id: {'embedding': array, 'frames_lost': 0, 'nome': str}}
        self.lost_faces: Dict[int, Dict[str, Any]] = {}
        # Distância máxima para associar faces entre frames (em pixels)
        self.max_distance = max_distance
        # Número máximo de frames que uma face pode estar perdida
        self.max_frames_lost = max_frames_lost
        # Threshold de similaridade de embedding para reconhecimento facial
        self.embedding_threshold = embedding_threshold
        # Faces conhecidas para reconhecimento
        self.known_faces = known_faces if known_faces is not None else {}
    
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
               frame: Optional[np.ndarray] = None) -> List[Tuple[int, int, int, int, int, str]]:
        """
        Atualiza o rastreador com novas detecções e retorna as faces com seus IDs e nomes.
        Usa posição e reconhecimento facial por embeddings para associar faces.
        Reconhece faces conhecidas e usa seus nomes, caso contrário usa "Pessoa{id}".
        
        Parâmetros
        ----------
        boxes: lista de caixas detectadas no formato (x, y, w, h)
        frame: frame completo do vídeo (opcional, necessário para extração de embeddings)
        
        Retorno
        -------
        Lista de tuplas (x, y, w, h, face_id, nome) com as caixas, IDs únicos e nomes
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
                        'frames_lost': face_data.get('frames_lost', 0),
                        'nome': face_data.get('nome', f"Pessoa{face_id}")
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
                
                # Atualiza a face ativa com nova posição e embedding, mantendo o nome
                nome_existente = face_data.get('nome')
                # Se não tem nome e tem novo embedding, tenta reconhecer
                if not nome_existente or nome_existente.startswith('Pessoa'):
                    novo_embedding = new_embedding if new_embedding is not None else face_data.get('embedding')
                    if novo_embedding is not None and self.known_faces:
                        nome_reconhecido = _recognize_face(novo_embedding, self.known_faces, self.embedding_threshold)
                        if nome_reconhecido:
                            nome_existente = nome_reconhecido
                
                self.active_faces[face_id] = {
                    'box': new_box,
                    'frames_lost': 0,
                    'embedding': new_embedding if new_embedding is not None else face_data.get('embedding'),
                    'nome': nome_existente or face_data.get('nome', f"Pessoa{face_id}")
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
                
                # Preserva o nome existente ou tenta reconhecer
                if is_lost_face:
                    nome_existente = lost_data.get('nome')
                    # Remove das perdidas e adiciona às ativas
                    self.active_faces[best_match_id] = {
                        'box': box,
                        'frames_lost': 0,
                        'embedding': embedding,
                        'nome': nome_existente or f"Pessoa{best_match_id}"
                    }
                    del self.lost_faces[best_match_id]
                else:
                    nome_existente = active_data.get('nome')
                    # Atualiza a face ativa (não encontrada por posição, mas encontrada por embeddings)
                    self.active_faces[best_match_id] = {
                        'box': box,
                        'frames_lost': 0,
                        'embedding': embedding,
                        'nome': nome_existente or f"Pessoa{best_match_id}"
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
                            'frames_lost': 0,  # Reset ao mover para lost_faces
                            'nome': face_data.get('nome', f"Pessoa{face_id}")
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
                
                # Tenta reconhecer a face conhecida
                nome_reconhecido = None
                if embedding is not None and self.known_faces:
                    nome_reconhecido = _recognize_face(embedding, self.known_faces, self.embedding_threshold)
                
                # Se não reconheceu, usa "Pessoa{id}"
                nome = nome_reconhecido if nome_reconhecido else f"Pessoa{new_id}"
                
                # Adiciona à lista de faces ativas
                self.active_faces[new_id] = {
                    'box': box,
                    'frames_lost': 0,
                    'embedding': embedding,
                    'nome': nome
                }
                tracked_boxes[new_id] = box
        
        # Converte o dicionário em lista de tuplas (x, y, w, h, face_id, nome)
        result = []
        for face_id, (x, y, w, h) in tracked_boxes.items():
            nome = self.active_faces.get(face_id, {}).get('nome', f"Pessoa{face_id}")
            result.append((x, y, w, h, face_id, nome))
        
        # Retorna ordenado por ID para consistência
        return sorted(result, key=lambda x: x[4])


class PoseActivityTracker:
    """
    Classifica atividades humanas por pessoa usando keypoints do YOLOv11 pose.
    """

    LABELS = ["Parado", "Andando", "Correndo", "Sentado", "Gesticulando", "Sem Pose"]

    def __init__(self, conf_threshold: float = 0.3):
        self.conf_threshold = conf_threshold
        self.previous_centers: Dict[int, Tuple[float, float]] = {}
        self.previous_wrist_positions: Dict[int, Dict[str, Tuple[float, float]]] = {}

    def cleanup(self, active_face_ids: List[int]) -> None:
        active = set(active_face_ids)
        for face_id in list(self.previous_centers.keys()):
            if face_id not in active:
                del self.previous_centers[face_id]
        for face_id in list(self.previous_wrist_positions.keys()):
            if face_id not in active:
                del self.previous_wrist_positions[face_id]

    def classify(
        self,
        face_id: int,
        keypoints: Optional[List[Tuple[float, float, float]]],
        fps: float,
        frame_width: int,
        frame_height: int,
    ) -> str:
        if not keypoints:
            self.previous_centers.pop(face_id, None)
            self.previous_wrist_positions.pop(face_id, None)
            return "Sem Pose"

        hip_center = _average_position(
            [
                _get_keypoint(keypoints, 11, self.conf_threshold),
                _get_keypoint(keypoints, 12, self.conf_threshold),
            ]
        )
        if hip_center is None:
            self.previous_centers.pop(face_id, None)
            self.previous_wrist_positions.pop(face_id, None)
            return "Sem Pose"

        knee_center = _average_position(
            [
                _get_keypoint(keypoints, 13, self.conf_threshold),
                _get_keypoint(keypoints, 14, self.conf_threshold),
            ]
        )

        current_wrist_positions: Dict[str, Tuple[float, float]] = {}
        left_wrist = _get_keypoint(keypoints, 9, self.conf_threshold)
        right_wrist = _get_keypoint(keypoints, 10, self.conf_threshold)
        if left_wrist is not None:
            current_wrist_positions["left"] = left_wrist
        if right_wrist is not None:
            current_wrist_positions["right"] = right_wrist

        arm_motion = 0.0
        previous_wrist_positions = self.previous_wrist_positions.get(face_id, {})
        for label, point in current_wrist_positions.items():
            previous = previous_wrist_positions.get(label)
            if previous is not None:
                arm_motion = max(arm_motion, math.hypot(point[0] - previous[0], point[1] - previous[1]))

        speed = 0.0
        effective_fps = fps if fps > 0 else 30.0
        previous_center = self.previous_centers.get(face_id)
        if previous_center is not None:
            delta = math.hypot(hip_center[0] - previous_center[0], hip_center[1] - previous_center[1])
            speed = delta * effective_fps

        is_sitting = False
        if knee_center is not None:
            hip_knee_gap = abs(knee_center[1] - hip_center[1])
            sitting_threshold = max(30.0, frame_height * 0.06)
            is_sitting = hip_knee_gap < sitting_threshold

        self.previous_centers[face_id] = hip_center
        self.previous_wrist_positions[face_id] = current_wrist_positions

        arm_threshold = max(60.0, frame_height * 0.05)
        walking_threshold = max(50.0, frame_height * 0.06)
        running_threshold = max(160.0, frame_height * 0.12)

        if arm_motion > arm_threshold:
            return "Gesticulando"
        if is_sitting:
            return "Sentado"
        if speed > running_threshold:
            return "Correndo"
        if speed > walking_threshold:
            return "Andando"
        return "Parado"


def _get_keypoint(
    keypoints: List[Tuple[float, float, float]],
    index: int,
    conf_threshold: float,
) -> Optional[Tuple[float, float]]:
    if index >= len(keypoints):
        return None
    x, y, conf = keypoints[index]
    if conf < conf_threshold:
        return None
    return (float(x), float(y))


def _average_position(positions: List[Optional[Tuple[float, float]]]) -> Optional[Tuple[float, float]]:
    valid = [pos for pos in positions if pos is not None]
    if not valid:
        return None
    x = sum(pos[0] for pos in valid) / len(valid)
    y = sum(pos[1] for pos in valid) / len(valid)
    return (x, y)


def _draw_pose_keypoints(
    frame: np.ndarray,
    keypoints: List[Tuple[float, float, float]],
    conf_threshold: float = 0.3,
) -> None:
    connections = [
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12),
        (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]
    for x, y, conf in keypoints:
        if conf < conf_threshold:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)

    for a, b in connections:
        if a >= len(keypoints) or b >= len(keypoints):
            continue
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca < conf_threshold or cb < conf_threshold:
            continue
        cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255, 255, 255), 2)


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


def _load_yolo_movement_model(model_path: Path) -> "YOLO":  # type: ignore[name-defined]
    """
    Carrega o modelo YOLOv11 usado para detectar movimentos (pessoas) e reutiliza o cache.
    """
    global _YOLO_MOVEMENT_MODEL

    cached_path, cached_model = _YOLO_MOVEMENT_MODEL
    if cached_model is not None and cached_path == model_path:
        return cached_model

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo YOLOv11 para movimentos não encontrado em '{model_path}'."
            " Baixe o yolov11n.pt e salve nesse caminho."
        )

    try:
        model = YOLO(str(model_path))  # type: ignore[call-arg]
        print(f"Modelo YOLOv11 carregado de: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo YOLOv11: {e}")

    _YOLO_MOVEMENT_MODEL = (model_path, model)
    return model


def _load_yolo_pose_model(model_path: Path) -> "YOLO":  # type: ignore[name-defined]
    """
    Carrega o modelo YOLOv11 pose e reutiliza o cache.
    """
    global _YOLO_POSE_MODEL

    cached_path, cached_model = _YOLO_POSE_MODEL
    if cached_model is not None and cached_path == model_path:
        return cached_model

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo YOLOv11 pose nao encontrado em '{model_path}'."
            " Baixe o yolo11n-pose.pt e salve nesse caminho."
        )

    try:
        model = YOLO(str(model_path))  # type: ignore[call-arg]
        print(f"Modelo YOLOv11 pose carregado de: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo YOLOv11 pose: {e}")

    _YOLO_POSE_MODEL = (model_path, model)
    return model


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


def _detect_movements_yolo(
    image_bgr: np.ndarray,
    model_path: Path,
    min_conf: float = 0.30,
    target_classes: Optional[List[str]] = None,
) -> List[Tuple[int, int, int, int, float, str]]:
    """
    Detecta pessoas/movimentos com YOLOv11 filtrando por classes e retorna caixas com confiança.
    """
    model = _load_yolo_movement_model(model_path)
    results = model(image_bgr, verbose=False)

    target = {name.lower() for name in target_classes} if target_classes else None
    boxes: List[Tuple[int, int, int, int, float, str]] = []

    for r in results:
        if getattr(r, "boxes", None) is None:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
            if float(conf) < min_conf:
                continue

            cls_id = int(cls_id)
            label = model.names.get(cls_id, f"class{cls_id}")
            if target is not None and label.lower() not in target:
                continue

            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
            w = max(1, x2_i - x1_i)
            h = max(1, y2_i - y1_i)
            boxes.append((x1_i, y1_i, w, h, float(conf), label))

    return boxes


def _detect_pose_yolo(
    image_bgr: np.ndarray,
    model_path: Path,
    min_conf: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Detecta pose com YOLOv11 pose e retorna caixas e keypoints por pessoa.
    """
    model = _load_yolo_pose_model(model_path)
    results = model(image_bgr, verbose=False)

    detections: List[Dict[str, Any]] = []
    for r in results:
        if getattr(r, "boxes", None) is None:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()
        keypoints = getattr(r, "keypoints", None)

        for idx, ((x1, y1, x2, y2), conf, cls_id) in enumerate(zip(xyxy, confs, cls_ids)):
            if float(conf) < min_conf:
                continue

            cls_id = int(cls_id)
            label = model.names.get(cls_id, f"class{cls_id}")
            if label.lower() != "person":
                continue

            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
            w = max(1, x2_i - x1_i)
            h = max(1, y2_i - y1_i)

            kpts_list: Optional[List[Tuple[float, float, float]]] = None
            if keypoints is not None and getattr(keypoints, "xy", None) is not None:
                xy = keypoints.xy[idx].cpu().numpy()
                if getattr(keypoints, "conf", None) is not None:
                    conf_k = keypoints.conf[idx].cpu().numpy()
                else:
                    conf_k = np.ones((xy.shape[0],), dtype=float)
                kpts_list = [
                    (float(x), float(y), float(c))
                    for (x, y), c in zip(xy, conf_k)
                ]

            detections.append(
                {
                    "box": (x1_i, y1_i, w, h),
                    "conf": float(conf),
                    "label": label,
                    "keypoints": kpts_list,
                }
            )

    return detections


def _match_faces_to_poses(
    tracked_faces: List[Tuple[int, int, int, int, int, str]],
    pose_detections: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """
    Associa poses detectadas aos rostos rastreados usando o centro do rosto.
    """
    assignments: Dict[int, Dict[str, Any]] = {}
    used_pose: set[int] = set()

    for x, y, w, h, face_id, _ in tracked_faces:
        cx = x + w / 2.0
        cy = y + h / 2.0
        best_idx = None
        best_score = -1.0

        for idx, det in enumerate(pose_detections):
            if idx in used_pose:
                continue
            px, py, pw, ph = det["box"]
            if cx < px or cx > px + pw or cy < py or cy > py + ph:
                continue
            score = float(det.get("conf", 0.0))
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            used_pose.add(best_idx)
            assignments[face_id] = pose_detections[best_idx]

    return assignments


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


def _load_known_faces(pasta_conhecidos: Path, embedding_threshold: float = 0.6) -> Dict[str, List[np.ndarray]]:
    """
    Carrega faces conhecidas de uma pasta e extrai seus embeddings.
    
    Parâmetros
    ----------
    pasta_conhecidos: caminho para a pasta contendo fotos de pessoas conhecidas
    embedding_threshold: threshold de similaridade para reconhecimento
    
    Retorno
    -------
    Dicionário {nome_pessoa: [lista_de_embeddings]} com as faces conhecidas
    """
    known_faces: Dict[str, List[np.ndarray]] = {}
    
    # Verifica se a pasta existe
    if not pasta_conhecidos.exists():
        print(f"⚠️  Pasta de conhecidos não encontrada: {pasta_conhecidos}")
        return known_faces
    
    # Extensões de imagem suportadas
    extensoes = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Lista todos os arquivos de imagem na pasta
    arquivos_imagem = []
    for ext in extensoes:
        # Adiciona extensões em minúscula e maiúscula, evitando duplicatas
        arquivos_imagem.extend(list(pasta_conhecidos.glob(f"*{ext}")))
        arquivos_imagem.extend(list(pasta_conhecidos.glob(f"*{ext.upper()}")))
    
    # Remove duplicatas (caso existam)
    arquivos_imagem = list(set(arquivos_imagem))
    
    if not arquivos_imagem:
        print(f"⚠️  Nenhuma imagem encontrada em: {pasta_conhecidos}")
        return known_faces
    
    print(f"📚 Carregando {len(arquivos_imagem)} foto(s) de pessoas conhecidas...")
    
    # Processa cada imagem
    for arquivo in arquivos_imagem:
        # Extrai o nome da pessoa do nome do arquivo (antes do underscore ou ponto)
        nome_base = arquivo.stem  # Nome sem extensão
        
        # Remove números e underscores do final (ex: "Bela_4" -> "Bela")
        partes = nome_base.split('_')
        if len(partes) > 1:
            # Tenta usar apenas a primeira parte como nome
            nome_pessoa = partes[0]
        else:
            # Se não tem underscore, usa o nome completo
            nome_pessoa = nome_base
        
        # Lê a imagem
        try:
            img = cv2.imread(str(arquivo))
            if img is None:
                print(f"   ⚠️  Não foi possível ler: {arquivo.name}")
                continue
            
            # Detecta faces na imagem
            # Usa DeepFace para detectar e extrair embedding
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Extrai embedding usando DeepFace
                # Usa enforce_detection=False para permitir processamento mesmo se a detecção não for perfeita
                result = DeepFace.represent(
                    img_rgb,
                    model_name="VGG-Face",
                    enforce_detection=False,  # Permite processar mesmo se a detecção não for perfeita
                )
                
                # DeepFace pode retornar lista ou dict
                if isinstance(result, list):
                    if len(result) == 0:
                        print(f"   ⚠️  Nenhuma face detectada em: {arquivo.name}")
                        continue
                    result = result[0]
                
                # Verifica se result é um dicionário
                if not isinstance(result, dict):
                    print(f"   ⚠️  Formato inesperado do resultado para: {arquivo.name}")
                    continue
                
                embedding = result.get("embedding")
                if embedding is not None:
                    embedding_array = np.array(embedding)
                    
                    # Verifica se o embedding tem tamanho válido
                    if embedding_array.size > 0:
                        # Adiciona à lista de embeddings desta pessoa
                        if nome_pessoa not in known_faces:
                            known_faces[nome_pessoa] = []
                        known_faces[nome_pessoa].append(embedding_array)
                    else:
                        print(f"   ⚠️  Embedding vazio para: {arquivo.name}")
                else:
                    print(f"   ⚠️  Não foi possível extrair embedding de: {arquivo.name}")
                    
            except Exception as e:
                # Se não conseguiu detectar face, mostra erro completo
                error_msg = str(e)
                print(f"   ⚠️  Erro ao processar {arquivo.name}: {error_msg[:100]}")
                continue
                
        except Exception as e:
            print(f"   ⚠️  Erro ao processar {arquivo.name}: {e}")
            continue
    
    # Resumo
    total_pessoas = len(known_faces)
    total_fotos = sum(len(embeddings) for embeddings in known_faces.values())
    
    print(f"✅ Carregadas {total_fotos} foto(s) de {total_pessoas} pessoa(s) conhecida(s)")
    
    if total_pessoas > 0:
        print(f"   Pessoas conhecidas: {', '.join(sorted(known_faces.keys()))}")
    else:
        print(f"\n❌ ERRO: Não foi possível carregar nenhuma face conhecida!")
        print(f"   Verifique se:")
        print(f"   - A pasta {pasta_conhecidos} existe")
        print(f"   - As imagens contêm rostos detectáveis")
        print(f"   - Os arquivos têm extensões .jpg, .jpeg, .png ou .bmp")
        raise RuntimeError(f"Não foi possível carregar faces conhecidas de: {pasta_conhecidos}")
    
    return known_faces


def _recognize_face(embedding: np.ndarray, known_faces: Dict[str, List[np.ndarray]], 
                   threshold: float = 0.6) -> Optional[str]:
    """
    Reconhece uma face comparando seu embedding com faces conhecidas.
    
    Parâmetros
    ----------
    embedding: embedding da face a ser reconhecida
    known_faces: dicionário com faces conhecidas {nome: [embeddings]}
    threshold: threshold de similaridade mínimo para reconhecimento
    
    Retorno
    -------
    Nome da pessoa reconhecida, ou None se não reconhecer
    """
    if embedding is None or not known_faces:
        return None
    
    best_name = None
    best_similarity = 0.0
    
    # Compara com todas as faces conhecidas
    for nome, embeddings_list in known_faces.items():
        # Compara com todos os embeddings desta pessoa (pode ter múltiplas fotos)
        for known_embedding in embeddings_list:
            similarity = _calculate_embedding_similarity(embedding, known_embedding)
            
            # Se encontrou uma correspondência melhor
            if similarity > best_similarity:
                best_similarity = similarity
                best_name = nome
    
    # Se a similaridade é maior que o threshold, retorna o nome
    if best_similarity >= threshold:
        return best_name
    
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
    yolo_model_path: Optional[Path] = None,  # Caminho opcional para o modelo YOLOv8-face
    yolo_min_conf: float = 0.30,  # Confiança mínima para YOLOv8-face
    yolo_movement_model_path: Optional[Path] = None,  # Caminho para o YOLOv11 de movimentos
    yolo_movement_min_conf: float = 0.30,  # Confiança mínima para YOLOv11
    yolo_movement_classes: Optional[List[str]] = None,  # Classes alvo para YOLOv11
    yolo_pose_model_path: Optional[Path] = None,  # Caminho para o YOLOv11 pose
    yolo_pose_min_conf: float = 0.25,  # Confianca minima para YOLOv11 pose
    salvar_resultado: bool = True,  # Se True, salva o video processado
    mostrar_progresso: bool = True,  # Se True, exibe barra de progresso
    processar_emocoes: bool = True,  # Se True, analisa emoções das faces detectadas
) -> Dict[str, Any]:  # Retorna dicionário com informações sobre o processamento
    """
    Detecta faces e emoções em um vídeo completo usando YOLOv8-face e DeepFace,
    e acrescenta detecção de movimentos com YOLOv11/ultralytics.

    Parâmetros
    ----------
    caminho_video: caminho para o vídeo de entrada
    caminho_saida: caminho para o vídeo de saída (padrão: src/Saida/video_detectado.mp4)
    yolo_model_path: caminho para o modelo YOLOv8-face (.pt)
                     (padrão: PROJECT_ROOT/models/yolov8n-face.pt)
    yolo_min_conf: confiança mínima para aceitar detecções do YOLOv8-face
    yolo_movement_model_path: caminho para o modelo YOLOv11 de movimentos
                              (padrão: PROJECT_ROOT/models/yolov11n.pt)
    yolo_movement_min_conf: confiança mínima para aceitar detecções do YOLOv11
    yolo_movement_classes: lista de classes (p.ex. ["person"]) filtradas pelo YOLOv11
    yolo_pose_model_path: caminho para o modelo YOLOv11 pose
                           (padrao: PROJECT_ROOT/models/yolo11n-pose.pt)
    yolo_pose_min_conf: confianca minima para aceitar deteccoes do YOLOv11 pose
    salvar_resultado: se True, salva o video processado
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
        # Define o caminho padrão do modelo YOLOv8-face
        yolo_model_path = PROJECT_ROOT / "models" / "yolov8n-face.pt"

    if yolo_movement_model_path is None:
        yolo_movement_model_path = PROJECT_ROOT / "models" / "yolov11n.pt"
    if yolo_movement_classes is None:
        yolo_movement_classes = ["person"]

    if yolo_pose_model_path is None:
        yolo_pose_model_path = PROJECT_ROOT / "models" / "yolo11n-pose.pt"

    # Define caminho de saída
    # Se o caminho de saída não foi especificado, cria um padrão
    if caminho_saida is None:
        # Define a pasta de saída padrão (v7 agora guarda atividades)
        pasta_saida = PROJECT_ROOT / "src" / "Saida" / "v7"
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

    # Carrega faces conhecidas da pasta de conhecidos
    pasta_conhecidos = PROJECT_ROOT / "src" / "Entrada" / "conhecidos"
    known_faces = _load_known_faces(pasta_conhecidos, embedding_threshold=0.6)
    pose_activity_tracker = PoseActivityTracker()
    atividades_detectadas: Dict[str, int] = {label: 0 for label in PoseActivityTracker.LABELS}
    total_movements_detected = 0
    movements_por_frame: List[int] = []
    
    # Inicializa o rastreador de faces para manter IDs únicos entre frames
    # Usa reconhecimento facial por embeddings para identificar a mesma pessoa
    # mesmo quando ela reaparece após passar atrás de objetos ou virar de costas
    face_tracker = FaceTracker(
        max_distance=100.0,  # Distância máxima em pixels para associação por posição
        max_frames_lost=30,  # Frames que uma face pode estar perdida antes de ser removida (aumentado)
        embedding_threshold=0.6,  # Threshold de similaridade para reconhecimento facial (60%)
        known_faces=known_faces  # Faces conhecidas para reconhecimento
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
        pose_detections = _detect_pose_yolo(
            frame,
            model_path=yolo_pose_model_path,
            min_conf=yolo_pose_min_conf,
        )
        pose_by_face = _match_faces_to_poses(tracked_faces, pose_detections)

        active_face_ids = [face_id for _, _, _, _, face_id, _ in tracked_faces]
        pose_activity_tracker.cleanup(active_face_ids)

        activity_by_face: Dict[int, str] = {}
        for _, _, _, _, face_id, _ in tracked_faces:
            pose_det = pose_by_face.get(face_id)
            keypoints = pose_det.get("keypoints") if pose_det else None
            label = pose_activity_tracker.classify(face_id, keypoints, fps, largura, altura)
            activity_by_face[face_id] = label
            if keypoints:
                _draw_pose_keypoints(frame, keypoints)

        if activity_by_face:
            label_counts: Dict[str, int] = {}
            for label in activity_by_face.values():
                label_counts[label] = label_counts.get(label, 0) + 1
                atividades_detectadas[label] = atividades_detectadas.get(label, 0) + 1
            activity_label = max(label_counts.items(), key=lambda x: x[1])[0]
        else:
            activity_label = "Sem Pose"
            atividades_detectadas[activity_label] = atividades_detectadas.get(activity_label, 0) + 1

        cv2.putText(
            frame,
            f"Atividade: {activity_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # Detecta movimentos usando YOLOv11 e desenha caixas/labels
        movement_boxes = _detect_movements_yolo(
            frame,
            model_path=yolo_movement_model_path,
            min_conf=yolo_movement_min_conf,
            target_classes=yolo_movement_classes,
        )
        total_movements_detected += len(movement_boxes)
        movements_por_frame.append(len(movement_boxes))

        movement_color = (0, 165, 255)  # laranja médio
        for mx, my, mw, mh, mconf, mlabel in movement_boxes:
            mx = max(0, mx)
            my = max(0, my)
            mw = min(mw, largura - mx)
            mh = min(mh, altura - my)
            if mw <= 0 or mh <= 0:
                continue

            cv2.rectangle(
                frame,
                (mx, my),
                (mx + mw, my + mh),
                movement_color,
                2,
            )
            cv2.putText(
                frame,
                f"{mlabel} {int(mconf * 100)}%",
                (mx, max(20, my - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                movement_color,
                2,
                cv2.LINE_AA,
            )
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
        for x, y, w, h, face_id, nome in tracked_faces:
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

            # Desenha label com nome da pessoa (reconhecida ou "Pessoa{id}")
            # Calcula a posição y do label (acima do retângulo, mínimo y=20)
            label_y = max(y - 10, 20)
            # Adiciona texto com o nome da pessoa acima do retângulo
            cv2.putText(
                frame,  # Frame onde desenhar
                nome,  # Texto a ser desenhado (nome da pessoa ou "Pessoa{id}")
                (x, label_y),  # Posição do texto
                cv2.FONT_HERSHEY_SIMPLEX,  # Fonte a ser usada
                0.6,  # Escala da fonte
                cor,  # Cor do texto (mesma do retângulo)
                2,  # Espessura do texto
            )

            # Calcula a posição base para mensagens abaixo do rosto
            text_y = y + h + 25
            activity_text = f"Atividade: {activity_by_face.get(face_id, "Sem Pose")}"

            if emotion_text:
                cv2.putText(
                    frame,
                    emotion_text,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )
                activity_y = text_y + 20
            else:
                activity_y = text_y

            activity_y = min(activity_y, altura - 5)
            cv2.putText(
                frame,
                activity_text,
                (x, activity_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
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

    total_atividades = sum(atividades_detectadas.values())
    atividades_percentuais = {
        label: round((count / total_atividades * 100), 2) if total_atividades > 0 else 0.0
        for label, count in atividades_detectadas.items()
    }
    atividade_predominante = (
        max(atividades_detectadas.items(), key=lambda x: x[1])[0]
        if total_atividades > 0
        else "N/A"
    )

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
        "atividades_detectadas": dict(atividades_detectadas),  # Quantos frames caíram em cada atividade
        "atividades_percentuais": atividades_percentuais,  # Percentual de cada atividade
        "atividade_predominante": atividade_predominante,  # Atividade mais comum
        "total_movimentos_detectados": total_movements_detected,
        "media_movimentos_por_frame": round(total_movements_detected / frame_num, 2) if frame_num > 0 else 0,
        "movimentos_por_frame": movements_por_frame,
        "yolo_movement_model": str(yolo_movement_model_path),  # Modelo YOLOv11 usado
        "yolo_pose_model": str(yolo_pose_model_path),
        "yolo_pose_min_conf": yolo_pose_min_conf,
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

        print(f"\n??  Estatísticas de Atividades:")
        print(f"   Atividade predominante: {resultado['atividade_predominante']}")
        print(f"   Distribuição de atividades:")
        for label, count in sorted(resultado['atividades_detectadas'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percent = resultado['atividades_percentuais'].get(label, 0.0)
                print(f"      {label}: {count} ({percent}%)")

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
        pasta_saida = PROJECT_ROOT / "src" / "Saida" / "v7"
        # Cria a pasta de saída se não existir
        pasta_saida.mkdir(parents=True, exist_ok=True)
        # Define o caminho do arquivo JSON de estatísticas
        arquivo_stats = pasta_saida / "video_detection_activities_stats.json"
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
