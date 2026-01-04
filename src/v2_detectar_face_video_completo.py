"""
Script para detectar faces em vídeo completo usando YOLOv8-face.
Usa apenas yolov8n-face.pt, sem fallback.
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

# Determina o diretório base do projeto (pasta que contém src/)
# Obtém o diretório absoluto onde este script está localizado
SCRIPT_DIR = Path(__file__).parent.absolute()
# Se o diretório do script se chama "src", o projeto root é o pai, senão é o próprio diretório
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

# Cache global para o modelo YOLO (evita recarregar)
# Variável global que armazena o modelo YOLO carregado para reutilização entre frames
_YOLO_MODEL = None  # type: ignore[var-annotated]


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


def detectar_faces_video(
    caminho_video: Path,  # Caminho para o arquivo de vídeo de entrada
    caminho_saida: Optional[Path] = None,  # Caminho opcional para o vídeo de saída
    yolo_model_path: Optional[Path] = None,  # Caminho opcional para o modelo YOLO
    yolo_min_conf: float = 0.30,  # Confiança mínima para YOLO (30%)
    salvar_resultado: bool = True,  # Se True, salva o vídeo processado
    mostrar_progresso: bool = True,  # Se True, exibe barra de progresso
) -> Dict[str, Any]:  # Retorna dicionário com informações sobre o processamento
    """
    Detecta faces em um vídeo completo usando YOLOv8-face.

    Parâmetros
    ----------
    caminho_video: caminho para o vídeo de entrada
    caminho_saida: caminho para o vídeo de saída (padrão: src/Saida/v2/video_detectado.mp4)
    yolo_model_path: caminho para o modelo YOLOv8-face (.pt)
                     (padrão: PROJECT_ROOT/models/yolov8n-face.pt)
    yolo_min_conf: confiança mínima para aceitar detecções do YOLO
    salvar_resultado: se True, salva o vídeo processado
    mostrar_progresso: se True, exibe barra de progresso

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
        pasta_saida = PROJECT_ROOT / "src" / "Saida" / "v2"
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
    # Registra o tempo de início do processamento
    tempo_inicio = time.time()

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

        # Ordena as caixas para nomear as faces de forma consistente
        # Ordena as caixas primeiro por coordenada x (esquerda para direita) e depois por y (cima para baixo)
        boxes = sorted(boxes, key=lambda b: (b[0], b[1]))

        # Atualiza estatísticas
        # Conta quantas faces foram detectadas neste frame
        num_faces = len(boxes)
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

        # Desenha retângulos nas faces detectadas
        # Define a cor verde para os retângulos (formato BGR do OpenCV)
        cor = (0, 255, 0)  # verde para YOLO
        # Itera sobre cada face detectada para desenhar retângulos e labels
        for idx, (x, y, w, h) in enumerate(boxes, start=1):
            # Desenha um retângulo ao redor da face detectada
            # Parâmetros: frame, ponto inicial (x,y), ponto final (x+w, y+h), cor, espessura
            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)
            # Adiciona texto com o número da face acima do retângulo
            cv2.putText(
                frame,  # Frame onde desenhar
                f"Face {idx}",  # Texto a ser desenhado (número da face)
                (x, max(y - 10, 20)),  # Posição do texto (acima do retângulo, mínimo y=20)
                cv2.FONT_HERSHEY_SIMPLEX,  # Fonte a ser usada
                0.6,  # Escala da fonte
                cor,  # Cor do texto (mesma do retângulo)
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
    print("🎬 Iniciando análise de vídeo completo com YOLOv8-face\n")

    # Tenta processar o vídeo
    try:
        # Chama a função para detectar faces no vídeo
        resultado = detectar_faces_video(
            caminho_video=video_entrada,  # Passa o caminho do vídeo
            yolo_min_conf=0.30,  # Define confiança mínima de 30%
            salvar_resultado=True,  # Salva o vídeo processado
            mostrar_progresso=True,  # Exibe barra de progresso
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

        # Salva estatísticas em JSON
        # Define a pasta de saída
        pasta_saida = PROJECT_ROOT / "src" / "Saida" / "v2"
        # Cria a pasta de saída se não existir
        pasta_saida.mkdir(parents=True, exist_ok=True)
        # Define o caminho do arquivo JSON de estatísticas
        arquivo_stats = pasta_saida / "video_detection_stats.json"
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
