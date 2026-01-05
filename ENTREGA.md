# Entrega - Tech Challenge Fase 4

Este documento organiza o que precisa ser apresentado na entrega e como
reproduzir os resultados do projeto.

## Objetivo do projeto
Aplicação de análise de vídeo com:
- detecção/reconhecimento facial
- análise de expressões emocionais
- detecção de atividades
- geração de resumo automático

## Como executar (gerar a saída final)
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/download_yolo_model.py
python src/v9_final.py
```

## Arquivos de entrada
- Vídeo de entrada: `src/Entrada/Activities.mp4`
- Faces conhecidas: `src/Entrada/conhecidos/*.jpg`

## Saídas geradas (entrega técnica)
- Vídeo processado: `src/Saida/v9/Activities_detectado.mp4`
- Estatísticas de anomalias: `src/Saida/v9/video_detection_anomalias_stats.json`
- Relatório automático: `src/Saida/v9/relatorio_resumo.txt`

## Relatório automático (resumo)
O relatório em `src/Saida/v9/relatorio_resumo.txt` inclui:
- total de frames analisados
- número de anomalias detectadas
- distribuição de atividades e emoções
- resumo por pessoa (intervalo no vídeo, emoção e atividade predominantes)

## Demonstração em vídeo
Gravar um vídeo de até 10 minutos mostrando:
- execução do script final `src/v9_final.py`
- vídeo de saída com boxes, nomes e emoções
- relatório gerado

## Links para entrega
Preencher com os links finais:
- Vídeo (YouTube): https://www.youtube.com/watch?v=uBs0xYTHJaU
- Repositório (GitHub): https://github.com/GustavoMaltaGuimaraesBrasil/TC4_Visao
