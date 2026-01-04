"""
Script para detectar faces em imagens da pasta src/Entrada/calibrar
Agora com suporte a YOLOv8-face como detector principal.
"""

# Importa Path para manipulação de caminhos de arquivos de forma multiplataforma
from pathlib import Path
# Importa tipos de anotação para melhor documentação e verificação de tipos
from typing import List, Tuple, Optional

# Importa OpenCV para processamento de imagens e detecção de faces
import cv2
# Importa NumPy para manipulação de arrays e operações matemáticas
import numpy as np

# Tentativa de import do YOLO (ultralytics) - biblioteca para detecção de objetos usando YOLOv8
try:
    # Tenta importar a classe YOLO do pacote ultralytics
    from ultralytics import YOLO  # type: ignore
except ImportError:  # Se o pacote ultralytics não estiver instalado
    # Define YOLO como None para indicar que não está disponível
    YOLO = None  # type: ignore[assignment]

# Importa funções auxiliares do módulo face_detection para detecção alternativa de faces
from face_detection import (
    _try_import_mediapipe,  # Função para tentar importar MediaPipe
    _detect_faces_mediapipe,  # Função para detectar faces usando MediaPipe
    _detect_faces_haar,  # Função para detectar faces usando Haar Cascade
    _load_haar_classifiers,  # Função para carregar classificadores Haar
)

# Determina o diretório base do projeto (pasta que contém src/)
# Obtém o diretório absoluto onde este script está localizado
SCRIPT_DIR = Path(__file__).parent.absolute()
# Se o diretório do script se chama "src", o projeto root é o pai, senão é o próprio diretório
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR

# Cache global para o modelo YOLO (evita recarregar a cada imagem)
# Variável global que armazena o modelo YOLO carregado para reutilização
_YOLO_MODEL = None  # type: ignore[var-annotated]


def _try_load_yolo_model(model_path: Path) -> Optional["YOLO"]:  # type: ignore[name-defined]
    """
    Tenta carregar o modelo YOLO apenas uma vez e reaproveita nas próximas chamadas.
    """
    # Declara que vamos usar a variável global _YOLO_MODEL
    global _YOLO_MODEL

    # Se o modelo já foi carregado anteriormente, retorna o modelo em cache
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    # Verifica se o pacote YOLO está disponível (foi importado com sucesso)
    if YOLO is None:
        # Se não estiver disponível, imprime mensagem de aviso
        print(
            "Aviso: pacote 'ultralytics' não está instalado. "
            "Instale com: pip install ultralytics"
        )
        # Retorna None indicando que não foi possível carregar
        return None

    # Verifica se o arquivo do modelo existe no caminho especificado
    if not model_path.exists():
        # Se não existir, imprime mensagem de aviso
        print(
            f"Aviso: modelo YOLO não encontrado em '{model_path}'. "
            f"Baixe o yolov8-face e salve nesse caminho ou ajuste no código."
        )
        # Retorna None indicando que o modelo não foi encontrado
        return None

    # Tenta carregar o modelo YOLO do arquivo
    try:
        # Carrega o modelo YOLO a partir do caminho do arquivo (converte Path para string)
        _YOLO_MODEL = YOLO(str(model_path))  # type: ignore[call-arg]
        # Informa que o modelo foi carregado com sucesso
        print(f"Modelo YOLO carregado de: {model_path}")
    except Exception as e:  # Captura qualquer exceção durante o carregamento
        # Imprime mensagem de erro
        print(f"Erro ao carregar modelo YOLO: {e}")
        # Define o modelo como None para indicar falha
        _YOLO_MODEL = None

    # Retorna o modelo carregado (ou None se houve erro)
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
    model = _try_load_yolo_model(model_path)
    # Se o modelo não foi carregado, retorna lista vazia
    if model is None:
        return []

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


def detectar_faces_imagem(
    caminho_imagem: Path,  # Caminho para o arquivo de imagem
    detector: str = "auto",  # Tipo de detector: "auto", "yolo", "mediapipe", ou "haar"
    mp_min_conf: float = 0.6,  # Confiança mínima para MediaPipe (60%)
    haar_scale_factor: float = 1.05,  # Fator de escala para Haar Cascade
    haar_min_neighbors: int = 3,  # Mínimo de vizinhos para Haar Cascade
    haar_min_rel_size: float = 0.02,  # Tamanho mínimo relativo para Haar Cascade
    salvar_resultado: bool = True,  # Se True, salva a imagem com faces marcadas
    mostrar_resultado: bool = True,  # Se True, exibe a imagem na tela
    yolo_model_path: Optional[Path] = None,  # Caminho opcional para o modelo YOLO
    yolo_min_conf: float = 0.30,  # Confiança mínima para YOLO (30%)
) -> dict:  # Retorna dicionário com informações sobre as faces detectadas
    """
    Detecta faces em uma única imagem.

    Parâmetros
    ----------
    caminho_imagem: caminho para a imagem
    detector:
        "auto"  -> tenta YOLO, depois MediaPipe, depois Haar
        "yolo"  -> usa apenas YOLO
        "mediapipe" -> usa apenas MediaPipe
        "haar"  -> usa apenas Haar
    mp_min_conf: confiança mínima do MediaPipe
    haar_scale_factor, haar_min_neighbors, haar_min_rel_size: parâmetros do Haar
    salvar_resultado: se True, salva a imagem com as faces detectadas
    mostrar_resultado: se True, exibe a imagem na tela
    yolo_model_path: caminho para o modelo YOLOv8-face (.pt)
                     (padrão: PROJECT_ROOT/models/yolov8n-face.pt)
    yolo_min_conf: confiança mínima para aceitar detecções do YOLO

    Retorno
    -------
    dict com informações sobre as faces detectadas
    """
    # Verifica se o arquivo de imagem existe
    if not caminho_imagem.exists():
        # Se não existir, lança exceção
        raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

    # Carrega a imagem (usando numpy para lidar com caracteres especiais no Windows)
    # Abre o arquivo em modo binário para leitura
    with open(caminho_imagem, "rb") as f:
        # Lê todos os bytes do arquivo
        image_bytes = f.read()
    # Converte os bytes em um array numpy de inteiros sem sinal de 8 bits
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decodifica a imagem usando OpenCV a partir do array de bytes
    imagem = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # Verifica se a decodificação foi bem-sucedida
    if imagem is None:
        # Se falhou, lança exceção
        raise ValueError(f"Não foi possível carregar a imagem: {caminho_imagem}")

    # Converte o nome do detector para minúsculas para comparação case-insensitive
    detector = detector.lower()
    # Inicializa lista vazia para armazenar as caixas das faces detectadas
    boxes: List[Tuple[int, int, int, int]] = []
    # Inicializa variável para rastrear qual detector foi usado
    detector_used = "none"

    # ---------------------------------------------------------
    # 1) YOLOv8-face (prioritário para uso profissional)
    # ---------------------------------------------------------
    # Verifica se deve usar YOLO (modo auto ou yolo explícito)
    if detector in ("auto", "yolo"):
        # Se o caminho do modelo não foi especificado, usa o padrão
        if yolo_model_path is None:
            # Define o caminho padrão do modelo YOLO
            yolo_model_path = PROJECT_ROOT / "models" / "yolov8n-face.pt"

        # Chama a função para detectar faces usando YOLO
        boxes = _detect_faces_yolo(
            imagem,  # Passa a imagem
            model_path=yolo_model_path,  # Passa o caminho do modelo
            min_conf=yolo_min_conf,  # Passa a confiança mínima
        )
        # Se encontrou faces, marca que usou YOLO
        if boxes:
            detector_used = "yolo"
        # Se o usuário pediu explicitamente YOLO, marca como usado mesmo sem detecções
        elif detector == "yolo":
            # Se o usuário pediu explicitamente YOLO, não faz fallback
            detector_used = "yolo"

    # ---------------------------------------------------------
    # 2) MediaPipe (fallback caso YOLO não ache nada em modo auto)
    # ---------------------------------------------------------
    # Se ainda não encontrou faces e está em modo auto ou mediapipe explícito
    if detector_used == "none" and detector in ("auto", "mediapipe"):
        # Tenta importar MediaPipe
        mp = _try_import_mediapipe()
        # Se MediaPipe está disponível
        if mp is not None:
            # Detecta faces usando MediaPipe
            boxes = _detect_faces_mediapipe(imagem, mp, min_conf=mp_min_conf)
            # Se encontrou faces, marca que usou MediaPipe
            if boxes:
                detector_used = "mediapipe"
            # Se o usuário pediu explicitamente MediaPipe, marca como usado
            elif detector == "mediapipe":
                detector_used = "mediapipe"

    # ---------------------------------------------------------
    # 3) Haar Cascade (último fallback)
    # ---------------------------------------------------------
    # Se ainda não encontrou faces e está em modo auto ou haar explícito
    if detector_used == "none" and detector in ("auto", "haar"):
        # Carrega os classificadores Haar (frontal e perfil)
        frontal, profile = _load_haar_classifiers()
        # Detecta faces usando Haar Cascade
        boxes = _detect_faces_haar(
            imagem,  # Passa a imagem
            frontal=frontal,  # Passa o classificador frontal
            profile=profile,  # Passa o classificador de perfil
            scale_factor=haar_scale_factor,  # Passa o fator de escala
            min_neighbors=haar_min_neighbors,  # Passa o mínimo de vizinhos
            min_rel_size=haar_min_rel_size,  # Passa o tamanho mínimo relativo
            try_rotations=True,  # Tenta detectar faces rotacionadas
        )
        # Marca que usou Haar (sempre marca, mesmo sem detecções)
        detector_used = "haar"

    # Ordena as caixas para nomear as faces de forma consistente (esquerda -> direita, cima -> baixo)
    # Ordena as caixas primeiro por coordenada x (esquerda para direita) e depois por y (cima para baixo)
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))

    # Desenha retângulos nas faces detectadas
    # Cria uma cópia da imagem original para não modificar a original
    imagem_resultado = imagem.copy()

    # Define um mapa de cores para cada tipo de detector
    color_map = {
        "yolo": (0, 255, 0),       # verde para YOLO
        "mediapipe": (255, 0, 0),  # azul para MediaPipe
        "haar": (0, 0, 255),       # vermelho para Haar
    }
    # Obtém a cor correspondente ao detector usado, ou vermelho como padrão
    cor = color_map.get(detector_used, (0, 0, 255))

    # Itera sobre cada face detectada para desenhar retângulos e labels
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        # Desenha um retângulo ao redor da face detectada
        # Parâmetros: imagem, ponto inicial (x,y), ponto final (x+w, y+h), cor, espessura
        cv2.rectangle(imagem_resultado, (x, y), (x + w, y + h), cor, 2)
        # Adiciona texto com o número da face acima do retângulo
        cv2.putText(
            imagem_resultado,  # Imagem onde desenhar
            f"Face {idx}",  # Texto a ser desenhado
            (x, max(y - 10, 20)),  # Posição do texto (acima do retângulo, mínimo y=20)
            cv2.FONT_HERSHEY_SIMPLEX,  # Fonte a ser usada
            0.6,  # Escala da fonte
            cor,  # Cor do texto (mesma do retângulo)
            2,  # Espessura do texto
        )

    # Salva resultado se solicitado
    if salvar_resultado:
        # Define o caminho da pasta de saída
        pasta_saida = PROJECT_ROOT / "src" / "Saida" / "v1"
        # Cria a pasta de saída (e pastas pai se necessário) se não existir
        pasta_saida.mkdir(parents=True, exist_ok=True)
        # Cria o nome do arquivo de saída adicionando "_detectado" antes da extensão
        nome_arquivo = caminho_imagem.stem + "_detectado" + caminho_imagem.suffix
        # Define o caminho completo do arquivo de saída
        caminho_saida = pasta_saida / nome_arquivo

        # Usa imencode para salvar com suporte a caracteres especiais no Windows
        # Obtém a extensão do arquivo em minúsculas
        ext = caminho_saida.suffix.lower()
        # Determina a extensão de codificação baseada na extensão do arquivo
        if ext in (".jpg", ".jpeg"):
            # Se for JPG ou JPEG, usa codificação JPG
            encode_ext = ".jpg"
        elif ext == ".png":
            # Se for PNG, usa codificação PNG
            encode_ext = ".png"
        else:
            # Para outras extensões, usa a própria extensão ou PNG como padrão
            encode_ext = ext if ext else ".png"

        # Codifica a imagem no formato especificado
        success, encoded_img = cv2.imencode(encode_ext, imagem_resultado)
        # Verifica se a codificação foi bem-sucedida
        if success:
            # Abre o arquivo de saída em modo binário para escrita
            with open(caminho_saida, "wb") as f:
                # Escreve os bytes da imagem codificada no arquivo
                f.write(encoded_img.tobytes())
            # Informa que o arquivo foi salvo
            print(f"Resultado salvo em: {caminho_saida}")
        else:
            # Se falhou, informa o erro
            print(f"Erro ao salvar imagem: {caminho_saida}")

    # Mostra resultado se solicitado
    if mostrar_resultado:
        # Obtém as dimensões da imagem (altura, largura)
        altura, largura = imagem_resultado.shape[:2]
        # Define a largura máxima para exibição (para não ocupar tela toda)
        max_largura = 1200
        # Verifica se a imagem é maior que a largura máxima
        if largura > max_largura:
            # Calcula o fator de escala para redimensionar
            escala = max_largura / largura
            # Calcula a nova largura mantendo a proporção
            nova_largura = int(largura * escala)
            # Calcula a nova altura mantendo a proporção
            nova_altura = int(altura * escala)
            # Redimensiona a imagem para o novo tamanho
            imagem_resultado = cv2.resize(imagem_resultado, (nova_largura, nova_altura))

        # Exibe a imagem em uma janela com o nome do arquivo
        cv2.imshow(f"Faces detectadas - {caminho_imagem.name}", imagem_resultado)
        # Informa ao usuário para pressionar uma tecla
        print("Pressione qualquer tecla para continuar...")
        # Aguarda o usuário pressionar uma tecla (0 = espera indefinidamente)
        cv2.waitKey(0)
        # Fecha todas as janelas abertas pelo OpenCV
        cv2.destroyAllWindows()

    # Retorna um dicionário com informações sobre o processamento
    return {
        "imagem": str(caminho_imagem),  # Caminho da imagem processada
        "faces_detectadas": len(boxes),  # Número de faces detectadas
        "boxes": boxes,  # Lista de caixas das faces detectadas
        "detector": detector_used,  # Nome do detector usado
    }


def main():
    """Processa todas as imagens da pasta src/Entrada/calibrar"""
    # Define o caminho da pasta de entrada onde estão as imagens
    pasta_entrada = PROJECT_ROOT / "src" / "Entrada" / "calibrar"

    # Verifica se a pasta de entrada existe
    if not pasta_entrada.exists():
        # Se não existir, informa e mostra os caminhos relevantes
        print(f"Pasta não encontrada: {pasta_entrada}")
        print(f"Diretório atual do script: {SCRIPT_DIR}")
        print(f"Raiz do projeto detectada: {PROJECT_ROOT}")
        # Encerra a função
        return

    # Extensões de imagem suportadas pelo script
    extensoes = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # Lista todas as imagens na pasta de entrada que têm extensões suportadas
    # Itera sobre todos os arquivos na pasta e filtra pelos que têm extensão válida
    imagens = [f for f in pasta_entrada.iterdir() if f.suffix.lower() in extensoes]

    # Verifica se encontrou alguma imagem
    if not imagens:
        # Se não encontrou, informa e encerra
        print(f"Nenhuma imagem encontrada em: {pasta_entrada}")
        return

    # Informa quantas imagens foram encontradas
    print(f"Encontradas {len(imagens)} imagem(ns) para processar\n")

    # Inicializa lista para armazenar os resultados do processamento
    resultados = []
    # Itera sobre cada imagem encontrada
    for i, imagem_path in enumerate(imagens, 1):
        # Mostra qual imagem está sendo processada (número atual / total)
        print(f"[{i}/{len(imagens)}] Processando: {imagem_path.name}")
        # Tenta processar a imagem
        try:
            # Chama a função para detectar faces na imagem
            resultado = detectar_faces_imagem(
                imagem_path,  # Passa o caminho da imagem
                detector="auto",   # Usa modo auto (tenta YOLO -> MediaPipe -> Haar)
                salvar_resultado=True,  # Salva a imagem processada
                mostrar_resultado=False,  # Nao exibe a imagem para evitar bloqueio
            )
            # Adiciona o resultado à lista de resultados
            resultados.append(resultado)
            # Mostra quantas faces foram detectadas e qual detector foi usado
            print(
                f"  ✓ {resultado['faces_detectadas']} face(s) "
                f"detectada(s) (detector: {resultado['detector']})\n"
            )
        # Captura qualquer exceção que ocorra durante o processamento
        except Exception as e:
            # Informa o erro ocorrido
            print(f"  ✗ Erro ao processar: {e}\n")
            # Adiciona um resultado de erro à lista
            resultados.append(
                {
                    "imagem": str(imagem_path),  # Caminho da imagem que falhou
                    "faces_detectadas": 0,  # Zero faces detectadas (erro)
                    "erro": str(e),  # Mensagem de erro
                }
            )

    # Resumo final
    # Imprime linha separadora
    print("\n" + "=" * 50)
    # Título do resumo
    print("RESUMO DO PROCESSAMENTO")
    # Linha separadora
    print("=" * 50)
    # Calcula o total de faces detectadas em todas as imagens
    total_faces = sum(r.get("faces_detectadas", 0) for r in resultados)
    # Mostra o total de imagens processadas
    print(f"Total de imagens processadas: {len(imagens)}")
    # Mostra o total de faces detectadas
    print(f"Total de faces detectadas: {total_faces}")
    # Calcula e mostra a média de faces por imagem
    print(f"Média de faces por imagem: {total_faces / len(imagens):.2f}")
    # Título da seção de detalhes
    print("\nDetalhes por imagem:")
    # Itera sobre cada resultado para mostrar detalhes
    for r in resultados:
        # Define símbolo de status: ✓ se encontrou faces, ✗ se não encontrou
        status = "✓" if r.get("faces_detectadas", 0) > 0 else "✗"
        # Mostra o status, nome do arquivo, número de faces e detector usado
        print(
            f"  {status} {Path(r['imagem']).name}: "
            f"{r.get('faces_detectadas', 0)} face(s) "
            f"(detector: {r.get('detector', 'n/a')})"
        )


# Verifica se o script está sendo executado diretamente (não importado como módulo)
if __name__ == "__main__":
    # Se estiver sendo executado diretamente, chama a função main
    main()
