# Entrega - Tech Challenge Fase 4

Este documento organiza o que precisa ser apresentado na entrega e como
reproduzir os resultados do projeto.

## Objetivo do projeto
Aplicacao de analise de video com:
- reconhecimento facial
- analise de expressoes emocionais
- deteccao de atividades
- geracao de resumo automatico

## Como executar (gerar a saida final)
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/download_yolo_model.py
python src/v9_final.py
```

## Arquivos de entrada
- Video de entrada: `src/Entrada/Activities.mp4`
- Faces conhecidas: `src/Entrada/conhecidos/*.jpg`

## Saidas geradas (entrega tecnica)
- Video processado: `src/Saida/v9/Activities_detectado.mp4`
- Estatisticas de anomalias: `src/Saida/v9/video_detection_anomalias_stats.json`
- Relatorio automatico: `src/Saida/v9/relatorio_resumo.txt`

## Relatorio automatico (resumo)
O relatorio em `src/Saida/v9/relatorio_resumo.txt` inclui:
- total de frames analisados
- numero de anomalias detectadas
- distribuicao de atividades e emocoes
- resumo por pessoa (intervalo no video, emocao e atividade predominantes)

## Demonstracao em video
Gravar um video de ate 10 minutos mostrando:
- execucao do script final `src/v9_final.py`
- video de saida com boxes, nomes e emocoes
- relatorio gerado

## Links para entrega
Preencher com os links finais:
- Video (YouTube): 
- Repositorio (GitHub): https://github.com/GustavoMaltaGuimaraesBrasil/TC4_Visao
