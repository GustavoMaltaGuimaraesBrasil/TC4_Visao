# Documento completo para estudo (10 min)

## 1. Introdução
Este projeto trata de detecção facial em vídeo e análise temporal. A meta principal é localizar rostos ao longo do tempo e produzir um relatório que indique presença e eventos relevantes. A apresentação segue a ordem: requisitos do PDF, estratégia por versões, funcionalidades por fase, ferramentas usadas, dificuldades e conclusões.

## 2. O que o PDF pedia
O PDF de especificação exigia:
- detecção facial no vídeo;
- identificação de presença/ausência ao longo do tempo;
- relatório com eventos e descrição temporal.

Minha abordagem foi cumprir esses requisitos e, ao final, acrescentar melhorias que agregam valor.

## 3. Linha de raciocínio
O vídeo é um ambiente real, com variações de luz, ângulos, zoom, pessoas de costas e oclusões. Por isso, adotei uma linha de raciocínio incremental: construir por versões, validando cada etapa antes de avançar. Esse método reduz riscos e dá previsibilidade aos resultados.

## 4. Por que trabalhar por versões
A escolha por versões incrementais foi estratégica:
- facilita depuração (eu sei exatamente em que fase surgiu um problema);
- permite comparar ganhos de desempenho;
- preserva uma base funcional estável.

Essa organização também ajuda a explicar o projeto de forma clara para avaliação.

## 5. Funcionalidades por fase
### v1 - Detecção em imagens estáticas
- objetivo: validar o pipeline de detecção;
- funcionalidade: localizar faces e desenhar caixas;
- saída: imagens com faces marcadas;
- ferramentas: OpenCV + Haar Cascade.

### v2 - Processamento de vídeo por frames
- objetivo: aplicar detecção ao vídeo inteiro;
- funcionalidade: extrair e analisar frames;
- saída: métricas básicas por frame;
- ferramentas: OpenCV e NumPy.

### v3 - Detecção mais robusta
- objetivo: melhorar precisão;
- funcionalidade: usar MediaPipe/YOLO com fallback para Haar;
- saída: detecções mais confiáveis;
- ferramentas: MediaPipe, YOLOv8-face, OpenCV.

### v4 - Análise temporal e relatório
- objetivo: traduzir detecções em eventos humanos;
- funcionalidade: agregação temporal, "Fulano do minuto X ao Y";
- saída: relatório final com presença e intervalos;
- ferramentas: Python, OpenCV, estruturação de dados.

## 6. O que fiz a mais
O PDF pedia apenas detecção facial. Eu também implementei reconhecimento simples, salvando prints de rosto para identificar quem apareceu. Além disso, acrescentei uma descrição humana no relatório, com frases como: "Fulano está do minuto X ao Y fazendo.". Isso tornou a entrega mais interpretável.

## 7. Ferramentas e bibliotecas por fase
- v1: OpenCV, Haar Cascade;
- v2: OpenCV, NumPy;
- v3: MediaPipe, YOLOv8-face, OpenCV;
- v4: Python, OpenCV, estruturação de dados.

## 8. Exemplos de comandos
```powershell
python src/v1_detectar_face_imagens.py
python src/v2_processar_video.py
python src/v4_relatorio_timeline.py
```

## 9. Dificuldades encontradas
- O vídeo possui zoom excessivo em alguns momentos, dificultando detecção de movimento.
- Não consegui detectar a Bela, porque ela aparece deitada e quase só o rosto.
- Pessoas aparecem de costas e há oclusões frequentes.
- É um vídeo complexo, com pouca previsibilidade visual.

Esses fatores afetam tanto a detecção quanto a identificação contínua.

## 10. O que são anomalias
Anomalias são eventos fora do padrão esperado, como:
- ausência inesperada de alguém;
- mudança brusca de comportamento;
- movimentação atípica em um intervalo curto.

## 11. Técnicas para reduzir falso positivo
- análise de alguns frames antes de confirmar uma identificação;
- uso de janela temporal para validar presença contínua;
- fallback de detectores quando um falha.

Essas técnicas evitam que ruídos isolados sejam interpretados como eventos reais.

## 12. Conclusão
O projeto atendeu todos os requisitos do PDF e foi além com reconhecimento simples e descrição temporal. A estratégia por versões tornou o desenvolvimento mais seguro, compreensível e robusto. Mesmo com um vídeo difícil, foi possível gerar resultados úteis e interpretáveis.
