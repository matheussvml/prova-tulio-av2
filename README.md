# Estudo para AV2 — Machine Learning (Prof. Túlio Ribeiro)
> Unifor | Material progressivo — coberto PDF por PDF

---

## Seção 1 — Introdução ao Aprendizado de Máquina e Regressão Linear

---

### 1.1 O que é Aprendizado de Máquina?

**Definição**: Aprendizado de Máquina (AM) é um subcampo da IA onde o sistema **aprende padrões a partir de dados**, sem ser explicitamente programado para cada regra.

> "Em vez de escrever regras → você fornece dados e o modelo descobre as regras."

---

### 1.2 Tipos de Aprendizado

| Tipo | Dados de entrada | Objetivo | Exemplos |
|------|-----------------|----------|----------|
| **Supervisionado** | Dados com rótulos (x, y) | Aprender f(x) ≈ y | Prever preço de imóvel, classificar spam |
| **Não-supervisionado** | Dados SEM rótulos (só x) | Descobrir estrutura oculta | Agrupar clientes, detectar anomalias |
| **Semi-supervisionado** | Poucos rótulos + muitos sem rótulo | Aproveitar dados não rotulados | Classificação de imagens com poucas labels |

#### Regra de ouro para identificar o tipo:
- **Tem y (resposta correta) no treino?** → Supervisionado
- **Não tem y?** → Não-supervisionado
- **Tem y só para uma parte dos dados?** → Semi-supervisionado

---

### 1.3 Aprendizado Supervisionado: Regressão vs Classificação

| Critério | Regressão | Classificação |
|----------|-----------|---------------|
| **Saída (y)** | Valor contínuo (número real) | Categoria discreta (classe) |
| **Pergunta** | "Quanto?" / "Qual valor?" | "Qual categoria?" |
| **Exemplos** | Preço, temperatura, salário | Spam/não-spam, gato/cachorro |

---

### 1.4 Terminologia Fundamental

| Termo | Símbolo | Significado |
|-------|---------|-------------|
| **Feature** (atributo) | x | Variável de entrada (ex: tamanho do imóvel em m²) |
| **Label** (rótulo/alvo) | y | Variável de saída que queremos prever (ex: preço) |
| **Training set** | — | Dados usados para treinar o modelo |
| **Test set** | — | Dados novos para avaliar o modelo |
| **Número de exemplos** | m | Quantidade de linhas no dataset |
| **i-ésimo exemplo** | x⁽ⁱ⁾, y⁽ⁱ⁾ | O par de dados na linha i |

---

### 1.5 Modelo de Regressão Linear Simples

O modelo tenta aprender uma reta que melhor representa os dados:

```
f_{w,b}(x) = wx + b
```

| Parâmetro | Nome | Papel |
|-----------|------|-------|
| **w** | peso (weight) | Inclinação da reta — quanto y muda por unidade de x |
| **b** | viés (bias) ou intercepto | Valor de y quando x = 0 |

**Exemplo concreto:**
- Se w = 0.5 e b = 10, para um imóvel de 80 m²:
  - f(80) = 0.5 × 80 + 10 = **50 mil reais**

> O treinamento consiste em encontrar os valores ideais de **w** e **b** que minimizam o erro.

---

### 1.6 Função de Custo — MSE (Erro Quadrático Médio)

Para saber quão "ruim" está o modelo, calculamos o erro entre a predição e o valor real:

```
J(w, b) = (1 / 2m) × Σᵢ₌₁ᵐ (f(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

**Desmontando a fórmula:**

| Parte | Significado |
|-------|-------------|
| `f(x⁽ⁱ⁾)` | Predição do modelo para o exemplo i |
| `y⁽ⁱ⁾` | Valor real para o exemplo i |
| `(f - y)²` | Erro ao quadrado (penaliza erros grandes; elimina sinal negativo) |
| `Σ / m` | Média dos erros |
| `1/2` | Fator para simplificar a derivada no gradiente descendente |

**Objetivo**: Encontrar w e b que **minimizem J(w, b)**.

- J grande → modelo ruim (predições longe dos valores reais)
- J próximo de 0 → modelo bom

---

### 1.7 Código Python — Regressão Linear com Scikit-learn

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dados: tamanho do imóvel (m²) e preço (mil R$)
X = np.array([[40], [60], [80], [100], [120]])
y = np.array([25, 35, 50, 62, 75])

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Parâmetros aprendidos
print(f"w (coeficiente): {modelo.coef_[0]:.2f}")
print(f"b (intercepto): {modelo.intercept_:.2f}")

# Predição para um imóvel de 90 m²
pred = modelo.predict([[90]])
print(f"Preço previsto para 90m²: R$ {pred[0]:.1f} mil")

# Visualizar
X_plot = np.linspace(30, 130, 100).reshape(-1, 1)
plt.scatter(X, y, color='blue', label='Dados reais')
plt.plot(X_plot, modelo.predict(X_plot), color='red', label='Modelo')
plt.xlabel("Tamanho (m²)")
plt.ylabel("Preço (mil R$)")
plt.legend()
plt.title("Regressão Linear — Preço de Imóveis")
plt.show()
```

---

### 1.8 Questões no Estilo da Prova

---

**Questão 1** — Uma construtora quer desenvolver um sistema para estimar o **custo de construção** de um edifício com base no número de andares, área do terreno e tipo de acabamento. A equipe de TI possui um histórico completo de 500 obras anteriores com seus respectivos custos registrados. Qual é o tipo de aprendizado e o modelo mais adequado para esta situação?

a) Aprendizado não-supervisionado com K-Means, pois há muitas variáveis de entrada  
b) Aprendizado supervisionado de regressão, pois a saída é um valor contínuo (custo) e há rótulos disponíveis  
c) Aprendizado semi-supervisionado com SVM, pois parte dos dados não possui rótulo  
d) Aprendizado supervisionado de classificação, pois o custo pode ser categorizado em faixas  

> ✅ **Resposta: B**
> **Por quê**: Há dados históricos com resposta correta (custo = y) → supervisionado. A saída é um valor numérico contínuo (custo em R$) → regressão. Opção D é pegadinha: classificar em "faixas de custo" seria possível, mas o enunciado pede para *estimar o custo*, não categorizar.

---

**Questão 2** — Um analista treinou um modelo de regressão linear simples com os parâmetros `w = 1.200` e `b = 5.000` para prever o salário mensal (em R$) de um funcionário com base nos anos de experiência. Se a função de custo `J(w,b)` calculada no conjunto de treino foi de **320.000**, o que isso indica e qual deve ser a próxima ação?

a) O modelo está perfeito; J = 320.000 representa a soma das predições corretas  
b) O modelo apresenta erros significativos; deve-se ajustar w e b para reduzir J  
c) J = 320.000 indica underfitting garantido; deve-se adicionar mais features  
d) O valor de J é irrelevante para regressão linear; apenas R² importa  

> ✅ **Resposta: B**
> **Por quê**: J(w,b) é a função de custo MSE. Um valor alto de J significa que as predições estão distantes dos valores reais. A solução é ajustar w e b (via gradiente descendente, seção 2) para minimizar J. Opção C é incorreta: underfitting não é diagnosticado pelo valor absoluto de J, mas sim pela comparação com o erro esperado.

---

**Questão 3** — Uma startup de e-commerce possui dados de **10.000 clientes** com histórico de compras, mas apenas **200 desses clientes** responderam a uma pesquisa indicando se são "clientes fiéis" ou "clientes ocasionais". A equipe deseja usar esses dados para classificar os demais clientes. Qual abordagem é mais adequada?

a) Aprendizado supervisionado, usando apenas os 200 rotulados no treino  
b) Aprendizado não-supervisionado com clustering para agrupar todos os 10.000  
c) Aprendizado semi-supervisionado, aproveitando os 200 rótulos e os 9.800 não rotulados  
d) Não é possível treinar com tão poucos rótulos; é necessário rotular todos os 10.000  

> ✅ **Resposta: C**
> **Por quê**: Quando há poucos rótulos disponíveis mas muitos dados sem rótulo, o aprendizado **semi-supervisionado** é ideal pois aproveita a estrutura dos dados não rotulados para melhorar a classificação. Opção A desperdiça 98% dos dados; opção D é impraticável.

---

**Questão 4** — Dado o conjunto de treinamento abaixo para um modelo de regressão linear `f(x) = wx + b` com `w = 2` e `b = 1`:

| x⁽ⁱ⁾ | y⁽ⁱ⁾ | f(x⁽ⁱ⁾) |
|------|------|---------|
| 1    | 3    | 3       |
| 2    | 5    | 5       |
| 3    | 8    | 7       |

Qual é o valor de `J(w, b)`?

a) J = 0  
b) J = 1/3  
c) J = 2/3  
d) J = 4/3  

> ✅ **Resposta: B**
> **Cálculo**:
> - Erro i=1: (3 - 3)² = 0
> - Erro i=2: (5 - 5)² = 0
> - Erro i=3: (7 - 8)² = 1
> - J = (1/2×3) × (0 + 0 + 1) = (1/6) × 1 = **1/6 ≈ 0.167** ≈ 1/6
>
> *Nota*: a opção mais próxima é B = 1/3, que seria o resultado sem o fator 1/2. Fique atento se a prova usa MSE com ou sem o 1/2. Com 1/2m: J = 1/6. Sem 1/2m (MSE padrão): J = 1/3.

---

*Fim da Seção 1 — próxima seção: Gradiente Descendente*

---

## Seção 2 — Gradiente Descendente

---

### 2.1 O que é o Gradiente Descendente?

**Definição**: Algoritmo de otimização iterativo que encontra o **mínimo de uma função diferenciável** dando passos na direção oposta ao gradiente (derivada).

> Intuição: imagine estar no topo de uma montanha com nevoeiro. A cada passo, você sente a inclinação do chão e caminha para onde está mais íngreme para baixo. O gradiente descendente faz isso matematicamente com J(w,b).

**Objetivo aplicado**: encontrar os valores de w e b que **minimizam J(w,b)**.

---

### 2.2 Fórmulas de Atualização (o coração do algoritmo)

Repetir até convergir:

```
w = w - α × (∂/∂w) J(w,b)    →    w = w - α × (1/m) Σ (f(x⁽ⁱ⁾) - y⁽ⁱ⁾) × x⁽ⁱ⁾
b = b - α × (∂/∂b) J(w,b)    →    b = b - α × (1/m) Σ (f(x⁽ⁱ⁾) - y⁽ⁱ⁾)
```

**Desmontando a fórmula de w:**

| Parte | Significado |
|-------|-------------|
| `α` | Taxa de aprendizado — tamanho do passo |
| `∂/∂w J` | Derivada parcial de J em relação a w — indica a direção e inclinação |
| `(f(x⁽ⁱ⁾) - y⁽ⁱ⁾)` | Erro da predição no exemplo i |
| `× x⁽ⁱ⁾` | Vem da regra da cadeia (derivar wx⁽ⁱ⁾+b em relação a w = x⁽ⁱ⁾) |

**Regra crítica — atualização simultânea:**
```python
# CORRETO: calcular tmp primeiro, depois atribuir
tmp_w = w - alpha * grad_w
tmp_b = b - alpha * grad_b
w = tmp_w
b = tmp_b

# ERRADO: atualizar w antes de calcular grad_b com o w antigo
w = w - alpha * grad_w   # w já mudou!
b = b - alpha * grad_b   # usa w novo, mas deveria ser o antigo
```

---

### 2.3 Intuição da Derivada — Por que funciona?

A derivada ∂J/∂w indica a **inclinação da curva de custo** naquele ponto:

| Situação | Derivada | w após update | Efeito |
|----------|----------|---------------|--------|
| Estou à direita do mínimo | > 0 (positiva) | w - α×(positivo) → w **diminui** | Caminha para esquerda |
| Estou à esquerda do mínimo | < 0 (negativa) | w - α×(negativo) → w **aumenta** | Caminha para direita |
| Estou no mínimo | = 0 | w não muda | Convergiu |

> O sinal de menos na fórmula garante que sempre caminhamos na direção de **descida**.

---

### 2.4 Taxa de Aprendizado (α) — O hiperparâmetro mais crítico

| Valor de α | Comportamento | Problema |
|------------|---------------|---------|
| Muito pequeno (ex: 0.0001) | Passos minúsculos | Convergência extremamente lenta |
| Adequado (ex: 0.01) | Desce suavemente até o mínimo | Ideal |
| Alto (ex: 0.5) | Oscila ao redor do mínimo | Pode não convergir precisamente |
| Muito alto (ex: 5.0) | Salta além do mínimo para cima | **Diverge** — J aumenta em vez de diminuir |

**Regra prática**: se J está **aumentando** durante o treino → α muito alto. Se está **diminuindo muito devagar** → α muito pequeno.

---

### 2.5 Critérios de Parada

| Abordagem | Descrição | Quando usar |
|-----------|-----------|-------------|
| **Número fixo de épocas** | Define 1000 iterações e para | Simples, comum na prática |
| **Tolerância ε** | Para quando \|J_atual - J_anterior\| ≤ ε | Mais preciso, evita paradas prematuras |

---

### 2.6 Mínimo Local vs. Global — Armadilha importante

- **Função convexa** (formato de tigela): **um único mínimo global** → GD sempre converge para ele
  - A regressão linear com MSE é **convexa** — não há risco de mínimo local
- **Função não-convexa**: múltiplos mínimos locais → GD pode ficar "preso" num mínimo local dependendo do ponto de partida

> Ponto de sela (*saddle point*): parece mínimo num eixo mas é máximo em outro.

---

### 2.7 Feature Scaling — Por que normalizar as features?

Quando as features têm escalas muito diferentes (ex: x₁ = número de quartos [1-5] vs. x₂ = preço [100.000-500.000]):

- A superfície de J(w,b) se torna uma **elipse alongada**
- O gradiente descendente oscila de um lado para o outro, convergindo lentamente

**Solução**: padronizar as features (Z-Score / StandardScaler):

```
x_scaled = (x - μ) / σ
```

Resultado: superfície de custo vira uma **tigela simétrica** → GD desce em linha reta e rápida.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)  # treino: fit + transform
X_test_scaled = scaler.transform(X_test)  # teste: só transform (com stats do treino)
```

---

### 2.8 Tipos de Gradiente Descendente

| Tipo | Dados por iteração | Caminho | Velocidade | Uso |
|------|-------------------|---------|-----------|-----|
| **Batch GD** | Todos os m exemplos | Suave, direto | Lento por iteração | Datasets pequenos |
| **Stochastic GD (SGD)** | 1 exemplo aleatório | Muito ruidoso | Rápido por iteração | Online learning |
| **Mini-Batch GD** | Pequeno lote (32-256) | Ligeiramente ruidoso | Equilibrado | **Padrão em DL** |

---

### 2.9 Otimizadores Avançados

O GD básico pode ser melhorado com otimizadores que adaptam α automaticamente:

| Otimizador | Diferencial |
|------------|-------------|
| **Momentum** | Acumula "embalo" da direção anterior — ajuda a sair de mínimos locais |
| **RMSprop** | Taxa de aprendizado adaptativa por variável |
| **ADAM** | Une Momentum + RMSprop — **estado da arte**, padrão no TensorFlow/PyTorch |

**ADAM** (*Adaptive Moment Estimation*):
- Exige menos calibração manual de α
- Converge muito mais rápido que GD simples
- É a **primeira escolha** em projetos reais de ML/DL

```python
# Usando ADAM no Keras/TensorFlow
model.compile(optimizer='adam', loss='mse')

# Usando SGD no scikit-learn
from sklearn.linear_model import SGDRegressor
modelo = SGDRegressor(learning_rate='adaptive', eta0=0.01, max_iter=1000)
```

---

### 2.10 Gradiente Descendente para Regressão Linear — Fórmulas Expandidas

Para f(x) = wx + b e J(w,b) = (1/2m)Σ(wx⁽ⁱ⁾ + b - y⁽ⁱ⁾)², as derivadas resolvidas são:

```
∂J/∂w = (1/m) × Σ (f(x⁽ⁱ⁾) - y⁽ⁱ⁾) × x⁽ⁱ⁾
∂J/∂b = (1/m) × Σ (f(x⁽ⁱ⁾) - y⁽ⁱ⁾)
```

**Por que x⁽ⁱ⁾ aparece na atualização de w?** Regra da cadeia: derivar (wx⁽ⁱ⁾ + b - y⁽ⁱ⁾)² em relação a w resulta em 2(wx⁽ⁱ⁾+b-y⁽ⁱ⁾) × x⁽ⁱ⁾ (o fator 2 cancela com o 1/2m).

**Propriedade especial do MSE**: como a função de custo é quadrática (convexa), **o gradiente descendente para regressão linear SEMPRE converge para o mínimo global**, independente do ponto inicial.

---

### 2.11 Código Python — Gradiente Descendente do Zero

```python
import numpy as np

def compute_cost(X, y, w, b):
    m = len(y)
    predictions = X * w + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, w_init=0, b_init=0, alpha=0.01, epochs=1000, tol=1e-6):
    w, b = w_init, b_init
    m = len(y)
    history = []

    for epoch in range(epochs):
        pred = X * w + b
        error = pred - y

        # gradientes (derivadas)
        dw = (1/m) * np.sum(error * X)
        db = (1/m) * np.sum(error)

        # atualização simultânea
        w = w - alpha * dw
        b = b - alpha * db

        cost = compute_cost(X, y, w, b)
        history.append(cost)

        # critério de parada por tolerância
        if epoch > 0 and abs(history[-2] - history[-1]) < tol:
            print(f"Convergiu na época {epoch}")
            break

    return w, b, history

# Exemplo
X = np.array([40, 60, 80, 100, 120], dtype=float)
y = np.array([25, 35, 50, 62, 75], dtype=float)

w, b, hist = gradient_descent(X, y, alpha=0.0001, epochs=10000)
print(f"w={w:.4f}, b={b:.4f}")
```

---

### 2.12 Questões no Estilo da Prova

---

**Questão 1** — Um cientista de dados está treinando um modelo de regressão linear para prever o consumo de energia elétrica (kWh) de um edifício com base na temperatura externa. Após 100 épocas de treinamento com taxa de aprendizado α = 2.0, ele observa que a função de custo J(w,b) está **aumentando** a cada iteração em vez de diminuir. Qual é o diagnóstico e a solução correta?

a) O modelo está com underfitting; deve-se adicionar mais features ao modelo  
b) A taxa de aprendizado está muito pequena; o algoritmo precisa de mais épocas para convergir  
c) A taxa de aprendizado está muito alta; o gradiente está ultrapassando o mínimo e divergindo  
d) Os dados não têm relação linear; deve-se usar um modelo de classificação  

> ✅ **Resposta: C**
> **Por quê**: Quando J aumenta com as iterações, o passo (α) é grande demais — o algoritmo "salta" para além do mínimo e sobe no outro lado, divergindo. A solução é reduzir α (ex: tentar 0.1, 0.01). Opção B é o diagnóstico oposto (J diminui mas devagar).

---

**Questão 2** — Uma engenheira treina um modelo de regressão linear múltipla com duas features: `x₁ = número de funcionários` (valores entre 10 e 500) e `x₂ = anos de experiência do gerente` (valores entre 1 e 15). O gradiente descendente está oscilando muito e demorando para convergir. Qual é a causa mais provável e o tratamento adequado?

a) O modelo está sofrendo overfitting; deve-se usar regularização L2  
b) As features têm escalas muito diferentes, causando superfície de custo elongada; deve-se aplicar Feature Scaling (ex: StandardScaler)  
c) A função de custo tem múltiplos mínimos locais; deve-se usar o otimizador ADAM  
d) O batch size está muito pequeno; deve-se usar Batch Gradient Descent com todos os dados  

> ✅ **Resposta: B**
> **Por quê**: x₁ varia de 10-500 e x₂ de 1-15 — escalas muito diferentes distorcem a superfície de J em elipse alongada, fazendo o GD oscilar. StandardScaler transforma cada feature para média 0 e desvio padrão 1, restaurando a superfície simétrica. Opção C é um distrator: a regressão linear com MSE **não tem mínimos locais** (função convexa).

---

**Questão 3** — Um time de ML precisa treinar um modelo em um dataset com **10 milhões de registros**. Usar o Batch Gradient Descent clássico (que calcula o gradiente com todos os dados) está se mostrando computacionalmente inviável. Qual variante do gradiente descendente é mais indicada para este cenário e por quê?

a) Stochastic GD (SGD), pois usa um único exemplo por vez, sendo o mais rápido por iteração e adequado para datasets enormes  
b) Mini-Batch GD, pois equilibra eficiência computacional (não carrega todos os dados na memória) com estabilidade de convergência  
c) Batch GD com mais épocas, pois mesmo sendo lento, garante convergência ao mínimo global  
d) O otimizador ADAM não é adequado para datasets grandes; deve-se usar apenas Momentum  

> ✅ **Resposta: B**
> **Por quê**: Mini-Batch GD é o **padrão em Deep Learning** para datasets grandes. Divide os dados em lotes (ex: 256 exemplos), atualizando os parâmetros a cada lote — mais rápido que Batch GD e mais estável que SGD. ADAM (opção D é pegadinha) é um otimizador que **pode ser usado com Mini-Batch** e é altamente recomendado.

---

**Questão 4** — Analise o pseudocódigo abaixo de implementação do gradiente descendente:

```
w = w - alpha * dw    # linha 1
b = b - alpha * db    # linha 2 (db calculado ANTES com o w original)
```

Este código apresenta um problema conceitual. Qual é ele?

a) A taxa de aprendizado alpha está sendo multiplicada incorretamente  
b) A atualização não é simultânea: w é modificado na linha 1, mas db foi calculado com o w original, gerando inconsistência  
c) As derivadas dw e db devem ser calculadas em sequência, não simultaneamente  
d) Não há problema; este é o modo correto de implementar o gradiente descendente  

> ✅ **Resposta: B** *(com ressalva)*
> **Por quê**: O correto é calcular tmp_w e tmp_b **antes** de atualizar qualquer parâmetro (atualização simultânea). Se w já foi atualizado antes de b ser recalculado, b usaria um w novo em vez do w antigo — matematicamente incorreto. Na prática com numpy, calcular `dw` e `db` no mesmo passo vetorizado resolve isso automaticamente.

---

*Fim da Seção 2 — próxima seção: Redução de Dimensionalidade*

---

## Seção 3 — Redução de Dimensionalidade

---

### 3.1 O que é Dimensionalidade?

**Dimensionalidade** = número de features (colunas) que descrevem cada amostra. Um dataset com altura, peso, idade e pressão arterial tem 4 dimensões (4D).

**Problema central:** dados reais (genômica, imagens, finanças) têm milhares de dimensões. Isso gera dois problemas sérios:

#### Maldição da Dimensionalidade (Curse of Dimensionality)

| Problema | Explicação |
|----------|-----------|
| **Esparsidade** | Em alta dimensão, os pontos ficam cada vez mais distantes entre si — padrões ficam ocultos |
| **Custo computacional** | Matrizes gigantescas, treino lento, modelos pesados |
| **Redundância** | Duas features podem medir "quase a mesma coisa" — informação duplicada |

#### Aplicações da Redução de Dimensionalidade

| Aplicação | Para que serve |
|-----------|---------------|
| **Redução de Ruído** | Remove features irrelevantes/ruidosas |
| **Visualização** | Projeta dados complexos em 2D ou 3D |
| **Pré-processamento** | Acelera e otimiza treino de modelos de ML |
| **Análise de Cluster** | Destaca as features essenciais para agrupamento |

---

### 3.2 SVD — Decomposição em Valores Singulares

**SVD** é a ferramenta de álgebra linear que decompõe qualquer matriz em três componentes. É o "motor matemático" do PCA e de técnicas de compressão.

#### Fórmula fundamental

```
A = U Σ V^T
```

#### Dissecando cada componente

| Símbolo | Nome | O que representa | Dimensões |
|---------|------|-----------------|-----------|
| **A** | Matriz original | Dados brutos (m amostras × n features) | m × n |
| **U** | Autovetores esquerdos | Como as **amostras (linhas)** se relacionam | m × m |
| **Σ (Sigma)** | Valores singulares | "Força"/importância de cada componente (diagonal) | m × n |
| **V^T** | Autovetores direitos (transposta) | Como as **variáveis (colunas)** se relacionam | n × n |

> **Regra de ouro:** Os valores em Σ estão em ordem decrescente. O maior = componente mais importante. Descartar os menores = compressão.

#### Onde e como usar SVD

- **Compressão de imagens:** aplicar SVD à matriz de pixels → manter só os k maiores valores singulares → reconstruir imagem com menos dados. Ex do PDF: k=50 ≈ qualidade original; k=5 = imagem borrada.
- **Base teórica do PCA** (scikit-learn usa SVD internamente).

#### Vantagens e Desvantagens

| | SVD |
|--|-----|
| **Vantagem** | Garantia matemática: sempre encontra a MELHOR aproximação linear (Teorema de Eckart-Young). Funciona com qualquer matriz (não precisa ser quadrada) |
| **Desvantagem** | Custoso para matrizes gigantescas. Componentes gerados são matemáticos — difíceis de interpretar humanamente |

---

### 3.3 PCA — Análise de Componentes Principais

**Ideia central:** encontrar novas direções (eixos) no espaço que capturem a **maior variância** dos dados. O primeiro componente (PC1) explica o máximo de "espalhamento"; o segundo (PC2) explica o restante de forma ortogonal; e assim por diante.

> **Analogia do professor:** Imagine iluminar uma xícara 3D com uma lanterna contra a parede. O PCA gira essa xícara matematicamente até que a "sombra" (projeção 2D) seja a maior e mais detalhada possível.

#### Fórmulas matemáticas

**1. Matriz de Covariância:**
```
C = (1 / n-1) * X^T * X
```

| Símbolo | Significado |
|---------|-------------|
| **C** | Matriz de covariância — captura como as variáveis variam juntas |
| **n** | Número de amostras |
| **X** | Matriz de dados JÁ CENTRALIZADA (média subtraída de cada coluna) |
| **X^T** | Matriz X transposta |

**2. Autodecomposição (Eigendecomposition):**
```
C * v = λ * v
```

| Símbolo | Significado |
|---------|-------------|
| **v** | Autovetor = nova direção (o Componente Principal) |
| **λ (lambda)** | Autovalor = quanta variância aquele eixo captura |

**3. Projeção final:**
```
Y = X · W
```
Onde W = matriz com os autovetores selecionados (colunas = componentes escolhidos).

#### Passo a Passo do PCA

```
1. CENTRALIZAR    → subtrair a média de cada coluna (leva dados à origem)
2. COVARIÂNCIA    → calcular C = (1/n-1) X^T X (mede redundâncias)
3. EIGENDECOMP.   → resolver Cv = λv (encontrar eixos e suas importâncias)
4. PROJEÇÃO       → Y = X · W (multiplicar pelos autovetores escolhidos)
```

#### PCA via SVD (como funciona no scikit-learn)

O PCA clássico (calcular a covariância X^T X para datasets com milhares de dimensões) é lento e propenso a erros de arredondamento. O scikit-learn **usa SVD internamente**:

- Centraliza X (subtrai a média).
- Aplica SVD: X = U Σ V^T.
- As colunas de **V** (autovetores direitos do SVD) são exatamente os Componentes Principais.

#### Variância Explicada — Como escolher quantos componentes manter?

```python
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)         # % de variância de cada PC
print(pca.explained_variance_ratio_.cumsum()) # acumulado
```

**Regras práticas:**
- **Limiar:** manter componentes até acumular **70% a 95%** de variância explicada.
- **Método do Cotovelo (Elbow):** plotar variância acumulada vs. número de componentes e escolher onde o gráfico "dobra" (o ganho de informação começa a diminuir drasticamente).

> Exemplo do PDF: dataset MNIST (784 dimensões) — apenas **83 componentes** explicam 90% dos dados.

#### Vantagens e Desvantagens do PCA

| | PCA |
|--|-----|
| **Vantagem** | Determinístico (resultado sempre igual). Ortogonalidade (sem redundância entre eixos). Extremamente rápido |
| **Desvantagem** | Só captura relações **lineares** (falha em dados curvos/espirais). Extremamente sensível a outliers (dados anômalos puxam o eixo) |

---

### 3.4 LDA — Análise Discriminante Linear

**Diferença fundamental em relação ao PCA:** o LDA é **supervisionado** — usa os rótulos (classes) dos dados. Objetivo: projetar os dados de forma que pontos da mesma classe fiquem **juntos** e pontos de classes diferentes fiquem o **mais separados possível**.

> **Analogia:** O PCA quer a "maior sombra". O LDA gira a lanterna buscando o ângulo onde a sombra do Vinho Tinto fique completamente separada da sombra do Vinho Branco. **Separação > formato.**

#### Fórmula: Critério de Fisher

```
J(w) = (w^T * S_B * w) / (w^T * S_W * w)
```

#### Dissecando a fórmula

| Símbolo | Nome | Significado |
|---------|------|-------------|
| **J(w)** | Função objetivo | O LDA **maximiza** este valor para encontrar a melhor separação |
| **w** | Vetor de projeção | O novo eixo/dimensão para onde os dados vão |
| **S_B** | Between-class scatter | Mede o quão **distantes** estão os centros (médias) de cada classe |
| **S_W** | Within-class scatter | Mede o quão **espalhados** estão os dados dentro de cada classe |

**Intuição:** queremos maximizar S_B (classes separadas) e minimizar S_W (dados compactos dentro da classe) → razão S_B/S_W máxima.

#### Limitação crítica de dimensões

```
Número máximo de dimensões = N_classes - 1
```

Exemplos:
- 2 classes (gato/cachorro) → máximo **1 dimensão**.
- 3 classes (vinho tinto/branco/rosé) → máximo **2 dimensões**.

#### Vantagens e Desvantagens do LDA

| | LDA |
|--|-----|
| **Vantagem** | Supervisionado: força separação das classes. Ideal como pré-processamento antes de classificadores |
| **Desvantagem** | Exige dados rotulados. Limitado a N_classes - 1 dimensões |

---

### 3.5 t-SNE — Visualização Não-Linear

**t-SNE** (t-distributed Stochastic Neighbor Embedding) é uma técnica **não-linear** focada em **visualização**. Preserva a vizinhança local: pontos próximos no espaço original devem ficar próximos na projeção 2D/3D.

> **Analogia:** O t-SNE vê os dados como pessoas de mãos dadas formando um "S". Ele desenrola esse "S" amassando e esticando o espaço — com uma regra: ninguém pode soltar a mão do vizinho mais próximo.

#### Fórmula: Divergência de Kullback-Leibler

```
C = Σ_i Σ_j p_{j|i} * log(p_{j|i} / q_{j|i})
```

#### Dissecando a fórmula

| Símbolo | Significado |
|---------|-------------|
| **C** | Função de custo total (o algoritmo minimiza C) |
| **p_{j\|i}** | Probabilidade de j ser vizinho de i no espaço **original** (alta dimensão) — usa distribuição **Gaussiana** |
| **q_{j\|i}** | Probabilidade de j ser vizinho de i no espaço **reduzido** (baixa dimensão) — usa distribuição **t-Student** |

#### Por que t-Student em vez de Gaussiana? — O Problema do Amontoamento (Crowding)

| Distribuição | Comportamento | Problema |
|-------------|--------------|---------|
| **Gaussiana** | Caudas curtas — decai rápido | Pontos moderadamente distantes ficam com probabilidade ~0 → "esmagados" no centro |
| **t-Student** | **Caudas longas** | Pontos distantes no original continuam distantes na visualização → clusters nítidos |

#### Perplexidade — o hiperparâmetro central do t-SNE

**Perplexidade** define o tamanho da vizinhança (número estimado de vizinhos próximos).

| Valor | Efeito |
|-------|--------|
| **Muito baixa (2-5)** | Foco extremo no micro-local → fragmenta dados em falsos agrupamentos |
| **Adequada (5-50)** | Clusters nítidos e representativos |
| **Muito alta (≈ total de amostras)** | Trata tudo como uma vizinhança → pontos colapsam em massa homogênea |

> **Regra:** perplexidade deve ser **estritamente menor** que o número total de amostras. Teste com múltiplos valores para validar se os clusters são reais.

#### Vantagens e Desvantagens do t-SNE

| | t-SNE |
|--|-------|
| **Vantagem** | Excelente para revelar clusters em dados complexos. Resolve o amontoamento visual |
| **Desvantagem** | **Probabilístico:** cada execução gera gráfico diferente (fixar `random_state`). Lento: **O(n²)**. Não preserva estrutura global (espaço vazio entre clusters não significa nada) |

---

### 3.6 UMAP — A Evolução do t-SNE

**UMAP** (Uniform Manifold Approximation and Projection) usa geometria e topologia algébrica (simplicial sets) para criar um grafo dos dados e projetá-lo em dimensão menor. É o **estado da arte** em redução de dimensionalidade para visualização.

> **Analogia:** O t-SNE desenha os bairros perfeitos da cidade, mas os joga aleatoriamente no mapa. O UMAP não apenas desenha os bairros, mas os **posiciona no mapa respeitando a distância correta entre cidades**.

#### Comparação t-SNE vs UMAP

| Critério | t-SNE | UMAP |
|----------|-------|------|
| **Complexidade** | O(n²) | O(n^1.14) — muito mais rápido |
| **Estrutura local** | Excelente | Excelente |
| **Estrutura global** | Não preserva | Preserva |
| **Reprojetabilidade** | Não (cada run diferente) | Sim (projeta novos dados) |
| **Base matemática** | Probabilística | Geometria Riemanniana |

#### Vantagens e Desvantagens do UMAP

| | UMAP |
|--|------|
| **Vantagem** | Muito mais rápido e escalável que t-SNE. Preserva vizinhança local E topologia global. Permite projetar dados novos em um mapa já treinado |
| **Desvantagem** | Base matemática densa (Geometria Riemanniana). Sensível ao parâmetro `n_neighbors` |

---

### 3.7 Autoencoders — Redução via Redes Neurais

**Autoencoder** é uma rede neural profunda treinada para **copiar a entrada para a saída**. O segredo está no **gargalo** (camada oculta com poucos neurônios) — a rede é forçada a comprimir o dado para passar por ele.

```
Entrada (784D) → [Encoder] → Gargalo (2D) → [Decoder] → Saída (784D reconstruída)
```

> **Analogia:** Pedimos à rede para resumir um livro de 300 páginas em apenas 2 páginas (o Gargalo). Depois, sem ver o original, o Decoder tenta reescrever o livro lendo só essas 2 páginas. Se a reconstrução for boa, as 2 coordenadas capturaram a essência dos dados.

#### Vantagens e Desvantagens dos Autoencoders

| | Autoencoders |
|--|-------------|
| **Vantagem** | Aprende padrões **não-lineares** (funções de ativação). Aceita imagens brutas (CNNs). Pode limpar ruídos |
| **Desvantagem** | Pode **memorizar** (overfitting) em vez de extrair essência. Exige tentativa/erro para projetar o gargalo. **Alto custo de GPU** |

---

### 3.8 Tabela Comparativa Geral

| Técnica | Supervisionado? | Linear? | Preserva global? | Uso principal | Complexidade |
|---------|----------------|---------|-----------------|--------------|-------------|
| **SVD** | Não | Sim | Sim | Compressão / base do PCA | — |
| **PCA** | Não | Sim | Sim | Pré-processamento, redução de ruído | O(d³) |
| **LDA** | **Sim** | Sim | Sim | Pré-processamento antes de classificação | Limitado a N_c-1 dims |
| **t-SNE** | Não | **Não** | **Não** | Visualização 2D/3D de clusters | O(n²) |
| **UMAP** | Não | **Não** | **Sim** | Visualização + análise | O(n^1.14) |
| **Autoencoder** | Não (auto-supervisionado) | **Não** | Depende | Compressão não-linear, denoising | Alto (GPU) |

---

### 3.9 Código Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Carregar dataset exemplo
data = load_iris()
X, y = data.data, data.target  # 150 amostras, 4 features, 3 classes

# Sempre normalizar antes de PCA/LDA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- PCA ----
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Variância explicada por PC:", pca.explained_variance_ratio_)
print("Variância acumulada:", pca.explained_variance_ratio_.cumsum())

# ---- Escolher n_components por variância acumulada ≥ 95% ----
pca_full = PCA()
pca_full.fit(X_scaled)
n_components_95 = np.argmax(pca_full.explained_variance_ratio_.cumsum() >= 0.95) + 1
print(f"Componentes para 95% de variância: {n_components_95}")

# ---- LDA ----
lda = LDA(n_components=2)  # máximo = N_classes - 1 = 2 para Iris (3 classes)
X_lda = lda.fit_transform(X_scaled, y)

# ---- t-SNE ----
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# ---- SVD manual ----
U, sigma, Vt = np.linalg.svd(X_scaled, full_matrices=False)
# Reconstruir com apenas os 2 maiores valores singulares (compressão)
k = 2
X_svd_reduced = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]

# ---- Visualização comparativa ----
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, X_red, title in zip(axes,
    [X_pca, X_lda, X_tsne],
    ['PCA (linear, sem rótulo)', 'LDA (linear, com rótulo)', 't-SNE (não-linear)']):
    scatter = ax.scatter(X_red[:, 0], X_red[:, 1], c=y, cmap='Set1')
    ax.set_title(title)
plt.tight_layout()
plt.show()
```

---

### 3.10 Questões no Estilo da Prova

**Questão 1 — Cenário de escolha de técnica**

Uma empresa farmacêutica tem um dataset de pacientes com **500 features genômicas** e **3 grupos de diagnóstico** rotulados (Saudável, Pré-diabético, Diabético). O objetivo é reduzir a dimensionalidade para melhorar um classificador. Qual técnica é a mais adequada?

a) t-SNE, pois é não-linear e lida bem com dados de alta dimensão  
b) PCA, pois é rápido e determinístico  
c) LDA, pois usa os rótulos para maximizar a separação entre os 3 grupos diagnósticos  
d) UMAP, pois preserva estrutura local e global  

> **Resposta: C**  
> **Por que C:** O LDA é supervisionado e usa os rótulos ativamente para projetar os dados de forma que as 3 classes fiquem maximamente separadas — ideal como pré-processamento antes de um classificador. Com 3 classes, terá no máximo 2 dimensões discriminantes.  
> **Por que não A:** t-SNE é para **visualização**, não para pré-processamento de classificadores. Não permite reprojetar dados novos.  
> **Por que não B:** PCA ignora os rótulos — pode projetar em direções que misturam as classes.  
> **Por que não D:** UMAP também é para visualização e não usa rótulos de forma supervisionada.

---

**Questão 2 — Fórmulas do PCA**

Uma cientista de dados aplica PCA em um dataset centralizado X e calcula a matriz de covariância C. O próximo passo é resolver a equação **Cv = λv**. O que representam **v** e **λ** nessa equação?

a) v é a taxa de aprendizado e λ é o erro de reconstrução  
b) v são os autovetores (novas direções/componentes principais) e λ são os autovalores (quantidade de variância capturada por cada eixo)  
c) v é a matriz de dados projetados e λ é o número de componentes selecionados  
d) v são os valores singulares e λ são os vetores de rotação  

> **Resposta: B**  
> **Por que B:** Na autodecomposição Cv = λv, os **autovetores v** definem as novas direções no espaço (os Componentes Principais), e os **autovalores λ** indicam quanta variância (informação) cada direção captura. O componente com maior λ é o PC1.  
> **Por que não D:** Valores singulares e autovetores são conceitos relacionados (SVD), mas na equação Cv = λv, λ são especificamente **autovalores**, não valores singulares diretamente.

---

**Questão 3 — t-SNE e perplexidade**

Um pesquisador aplica t-SNE em 10.000 amostras usando perplexidade = 9.999 (próxima ao total). Qual será o resultado provável?

a) Clusters extremamente nítidos e bem separados  
b) Os pontos colapsarão em uma massa homogênea, destruindo a estrutura real do dado  
c) O algoritmo falhará com erro de overflow  
d) Os clusters serão fragmentados em centenas de pequenos grupos  

> **Resposta: B**  
> **Por que B:** Uma perplexidade muito alta faz o algoritmo tratar o dataset inteiro como uma única vizinhança — cada ponto considera todos os outros como "vizinhos próximos". O resultado é que as probabilidades ficam uniformes e os pontos colapsam em uma massa central sem estrutura.  
> **Por que não A:** Alta perplexidade DESTRÓI a estrutura de clusters.  
> **Por que não D:** Fragmentação em pequenos grupos ocorre com perplexidade BAIXA demais.

---

**Questão 4 — SVD vs PCA: relação interna**

O scikit-learn implementa o PCA internamente usando SVD em vez de calcular a matriz de covariância X^T X. Qual é a principal razão para essa escolha?

a) O SVD permite usar rótulos de classe para melhorar a separação dos componentes  
b) A covariância X^T X é matematicamente incorreta para datasets não normalizados  
c) Calcular a covariância X^T X para datasets com milhares de dimensões é lento e propenso a erros numéricos de arredondamento; o SVD é mais rápido e estável  
d) O SVD produz componentes diferentes dos do PCA clássico, capturando mais informação  

> **Resposta: C**  
> **Por que C:** Para um dataset com d features, a matriz de covariância tem dimensões d×d. Para d=10.000 (ex: expressão gênica), calcular e decompor essa matriz é extremamente custoso. O SVD aplicado diretamente em X é numericamente mais estável e eficiente. O resultado final (os componentes principais) é matematicamente equivalente.  
> **Por que não A:** O SVD (e o PCA) são não-supervisionados — não usam rótulos.  
> **Por que não D:** As colunas de V no SVD são **exatamente** os mesmos componentes principais do PCA clássico.

---

*Fim da Seção 3 — próxima seção: Desafios de Aprendizado de Máquina*

---

## Seção 4 — Desafios de Aprendizado de Máquina

> **Resumo do PDF:** Os problemas em AM vêm de duas fontes: **algoritmos ruins** e/ou **dados ruins**. A maioria dos problemas práticos são de dados.

---

### 4.1 Visão Geral dos Desafios

| # | Desafio | Categoria |
|---|---------|-----------|
| 1 | Quantidade insuficiente de dados de treinamento | Dados |
| 2 | Dados de treinamento não representativos | Dados |
| 3 | Dados de baixa qualidade | Dados |
| 4 | Características irrelevantes | Dados/Algoritmo |
| 5 | **Overfitting** (sobreajuste) | Algoritmo |
| 6 | **Underfitting** (subajuste) | Algoritmo |

> **Garbage in, garbage out!** — Dados ruins + modelo excelente = resultado ruim.

---

### 4.2 Quantidade Insuficiente de Dados

**Regra prática:**
- **Problemas simples** (regressão linear, classificação binária): algumas centenas de amostras podem bastar.
- **Problemas complexos** (visão computacional, NLP): milhares a milhões de amostras são necessárias.

**Dados vs. Algoritmos — insight crucial:**

O paper de Banko & Brill (Microsoft Research) mostrou que, com volume suficiente de dados, algoritmos simples (Naïve Bayes) atingem performance semelhante a algoritmos sofisticados. Isso levou ao artigo "The Unreasonable Effectiveness of Data" (Halevy, Norvig, Pereira — Google):

> **Conclusão:** com dados suficientes, o algoritmo importa menos do que a quantidade e qualidade dos dados.

---

### 4.3 Dados Não Representativos

Os dados de treinamento **devem representar os casos que o modelo vai encontrar no mundo real**. Quando não representam, o modelo aprende padrões enviesados.

**Dois tipos de viés clássicos:**

| Tipo | Definição | Exemplo histórico |
|------|-----------|------------------|
| **Viés de amostragem** | A amostra não cobre todos os subgrupos da população | Literary Digest (1936): pesquisa com leitores de revista e donos de telefone previu errado as eleições Roosevelt x Landon — excluiu classes mais pobres que votaram em Roosevelt |
| **Viés de não-resposta** | Subgrupos que não respondem à pesquisa ficam sub-representados | Mesma pesquisa: quem não respondeu votou diferente de quem respondeu |

**No exemplo do PDF:** um modelo treinado só em países com GDP intermediário não generaliza bem para países muito ricos ou muito pobres.

---

### 4.4 Dados de Baixa Qualidade

Dados do mundo real são sujos. É necessário **mineração, higienização e padronização** antes de treinar qualquer modelo.

**Tipos de problemas comuns:**

| Problema | Exemplo do PDF |
|----------|---------------|
| Capitalização inconsistente | `COROLLA` vs `Corolla` |
| Typo/OCR | `V0lkswagem` (zero no lugar do "o") |
| Valor absurdo/outlier | Corolla 2019 com valor R$60.000.000 |
| Ano impossível | Duster 2218 |
| Separador decimal errado | `1.000` km vs `1000` km |

**Fluxo de limpeza:**
```python
import pandas as pd

df = pd.read_csv('carros.csv')

# Verificar tipos e valores nulos
print(df.info())
print(df.describe())

# Padronizar texto
df['Modelo'] = df['Modelo'].str.title()
df['Marca'] = df['Marca'].str.replace('0', 'o')  # typo simples

# Remover outliers de preço
Q1 = df['Valor'].quantile(0.25)
Q3 = df['Valor'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Valor'] >= Q1 - 1.5*IQR) & (df['Valor'] <= Q3 + 1.5*IQR)]

# Anos impossíveis
df = df[df['Ano'].between(1900, 2026)]
```

---

### 4.5 Características Irrelevantes — Feature Engineering

**Feature engineering** = processo de selecionar, extrair e criar features para maximizar a performance do modelo.

| Operação | Significado | Exemplo |
|----------|-------------|---------|
| **Seleção** | Escolher as features mais relevantes e descartar as irrelevantes | Descartar "Ano" se "Idade do carro" já está calculada |
| **Extração** | Combinar/transformar features existentes em novas | `Idade = 2026 - Ano` |
| **Criação** | Criar features novas a partir de domínio de conhecimento | `Km_por_ano = Quilometragem / Idade` |

> **Exemplo do PDF:** a coluna "Ano" foi riscada no dataset de carros — ela é irrelevante porque o que importa é a **idade** do carro, não o ano de fabricação em si.

---

### 4.6 Overfitting — Sobreajuste

**Definição:** o modelo tem desempenho excelente no conjunto de treino, mas **falha em dados novos** (generalização ruim). O modelo "decorou" os dados de treino em vez de aprender os padrões.

**Causa principal:** modelo muito complexo em relação ao tamanho/ruído do dataset.

**Diagnóstico:**
```
Erro treino: BAIXO  →  Erro teste: ALTO  →  OVERFITTING
```

**Solução: Regularização**

Regularização = imposição de restrições ao algoritmo, controladas por **hiperparâmetros**, para tornar o modelo mais simples.

```python
from sklearn.linear_model import Ridge, Lasso

# Ridge (L2): penaliza pesos grandes, mas não os zera
ridge = Ridge(alpha=1.0)  # alpha = força da regularização

# Lasso (L1): penaliza e pode zerar pesos (seleção automática de features)
lasso = Lasso(alpha=0.1)
```

**Como resolver overfitting — 5 estratégias:**

| Estratégia | Mecanismo |
|-----------|-----------|
| 1. Modelo mais simples | Menos parâmetros, menos capacidade de memorizar |
| 2. Reduzir número de atributos | Menos features = menos espaço para decorar ruído |
| 3. Regularização (L1/L2) | Penaliza pesos excessivos |
| 4. Adquirir mais dados | Mais exemplos = padrões mais sólidos |
| 5. Reduzir ruído nos dados | Limpeza de outliers e erros |

---

### 4.7 Underfitting — Subajuste

**Definição:** o modelo **não tem bom desempenho nem no conjunto de treino**. O modelo é simples demais para capturar os padrões dos dados.

**Causa principal:** modelo muito simples (ex: tentar ajustar uma linha reta em dados claramente não-lineares).

**Diagnóstico:**
```
Erro treino: ALTO  →  Erro teste: ALTO  →  UNDERFITTING
```

**Como resolver underfitting — 3 estratégias:**

| Estratégia | Mecanismo |
|-----------|-----------|
| 1. Modelo mais complexo | Mais parâmetros, mais capacidade de aprendizado |
| 2. Melhores atributos | Feature engineering: criar features mais informativas |
| 3. Reduzir restrições | Diminuir regularização se estiver muito intensa |

---

### 4.8 Diagrama: Trade-off Overfitting vs. Underfitting

```
Complexidade do modelo
        │
  ALTO  │  Underfitting ←──── ponto ideal ────→ Overfitting
        │      (simples)          ↑              (complexo)
  BAIXO │                    equilíbrio
        └──────────────────────────────────────→ Erro
                            treino  teste
```

| | Erro Treino | Erro Teste | Diagnóstico |
|-|------------|-----------|-------------|
| **Underfitting** | Alto | Alto | Modelo muito simples |
| **Bom ajuste** | Baixo | Baixo | Ideal |
| **Overfitting** | Muito baixo | Alto | Modelo decorou o treino |

---

### 4.9 Questões no Estilo da Prova

**Questão 1 — Identificação do problema**

Uma startup treinou um modelo de previsão de churn (cancelamento) de clientes. O modelo atingiu 98% de acurácia no conjunto de treinamento, mas apenas 61% no conjunto de teste. Qual é o diagnóstico mais provável?

a) Underfitting — o modelo é muito simples para os dados  
b) Overfitting — o modelo memorizou os dados de treinamento sem generalizar  
c) Dados insuficientes — o problema foi a quantidade de amostras  
d) Viés de amostragem — os dados de treino não eram representativos  

> **Resposta: B**  
> **Por que B:** A assinatura clássica do overfitting é exatamente essa: erro muito baixo no treino (98%) e erro alto no teste (61%). O modelo "decorou" os padrões de treino incluindo o ruído, mas falhou ao generalizar para dados novos.  
> **Por que não A:** Underfitting produziria erro alto TANTO no treino quanto no teste.  
> **Por que não C/D:** Embora possam co-ocorrer, o padrão treino-alto/teste-baixo é definitório do overfitting.

---

**Questão 2 — Soluções para overfitting**

Um engenheiro de ML percebe que seu modelo de classificação de e-mails está em overfitting. Ele decide adicionar um termo de penalização à função de custo que penaliza pesos elevados, controlado por um hiperparâmetro α. Que técnica ele está usando?

a) Feature engineering  
b) Data augmentation  
c) Regularização  
d) Normalização  

> **Resposta: C**  
> **Por que C:** Regularização é exatamente a imposição de restrições ao modelo através de um hiperparâmetro (α/lambda) que penaliza a complexidade — pesos grandes são "punidos" na função de custo. L2 (Ridge) penaliza o quadrado dos pesos; L1 (Lasso) penaliza o valor absoluto e pode zerá-los.  
> **Por que não D:** Normalização/padronização (StandardScaler) é pré-processamento de features, não controle de overfitting diretamente.

---

**Questão 3 — Viés de amostragem**

Uma empresa de delivery treina um modelo de previsão de demanda usando apenas dados de entregas realizadas nos últimos 6 meses — que incluem apenas o período pós-pandemia. Qual problema de dados esse cenário representa?

a) Dados de baixa qualidade — os dados contêm erros e outliers  
b) Quantidade insuficiente de dados — 6 meses é muito pouco  
c) Dados não representativos — o comportamento pós-pandemia pode não generalizar para o futuro pós-normalização  
d) Características irrelevantes — as features escolhidas são inadequadas  

> **Resposta: C**  
> **Por que C:** Os dados refletem um período atípico e podem não representar o comportamento "normal" da demanda futura. O modelo aprendeu padrões de um contexto específico (pós-pandemia) que pode não generalizar — é um problema de representatividade temporal da amostra.  
> **Por que não B:** A quantidade pode ser suficiente; o problema não é volume, mas representatividade do período.

---

**Questão 4 — Underfitting vs. Overfitting**

Um aluno treina uma regressão linear simples (f(x) = wx + b) em um conjunto de dados de preços de imóveis que claramente tem comportamento não-linear (polinomial). O modelo obtém erro de 35% no treino e 37% no teste. O diagnóstico correto é:

a) Overfitting — o modelo está memorizando o treino  
b) Underfitting — o modelo linear é simples demais para dados não-lineares  
c) Bom ajuste — os erros de treino e teste estão próximos  
d) Viés nos dados — a diferença indica dados não representativos  

> **Resposta: B**  
> **Por que B:** Quando o erro é alto tanto no treino quanto no teste, e a causa é um modelo muito simples para a complexidade dos dados, temos underfitting. Usar regressão linear para dados não-lineares é o exemplo clássico. A solução seria usar um modelo mais complexo (regressão polinomial, árvore de decisão, etc.).  
> **Por que não A:** No overfitting, o erro de treino é MUITO menor que o de teste — aqui os erros são próximos e ambos altos.  
> **Por que não C:** Erros próximos apenas indicam consistência, não que o modelo é bom (35% é alto).

---

*Fim da Seção 4 — próxima seção: Classificação*

---

## Seção 5 — Classificação

> **Fonte:** `5. Classificação.pptx.pdf`

---

### 5.1 Classificação vs Regressão

Ambas são tarefas de **Aprendizado Supervisionado** (dados rotulados), mas diferem no tipo de saída:

| | Regressão | Classificação |
|--|-----------|---------------|
| **Saída** | Valor **contínuo** | Valor **discreto** (categoria) |
| **Exemplo** | Prever o preço de um imóvel | Prever se o e-mail é spam ou não |
| **Modelo** | Regressão Linear | Regressão Logística, SVM, Árvore... |

**Exemplos do PDF de problemas de classificação:**

| Pergunta | y = 0 (negativo) | y = 1 (positivo) |
|----------|-----------------|-----------------|
| Este e-mail é spam? | Não spam | Spam |
| Esta transação é fraudulenta? | Legítima | Fraudulenta |
| Este tumor é maligno? | Benigno | Maligno |

> **ATENÇÃO:** Classe negativa ≠ "ruim", classe positiva ≠ "bom".  
> Negativa = **ausência** do evento. Positiva = **presença** do evento.  
> Um tumor benigno é a classe **negativa** (ausência de malignidade).

**Classificação binária:** y ∈ {0, 1} — apenas dois valores possíveis.  
**Classificação multiclasse:** y ∈ {0, 1, 2, ..., k} — mais de duas categorias.

---

### 5.2 Regressão Logística — A Ideia

**Por que não usar regressão linear para classificação?**

A regressão linear pode produzir valores fora do intervalo [0, 1] (ex: -0.3 ou 1.7), o que não faz sentido para probabilidades. Precisamos de uma função que **espreme qualquer valor real para o intervalo (0, 1)**.

**Solução: a função sigmóide (logística)**

```
g(z) = 1 / (1 + e^(-z))     com     0 < g(z) < 1
```

| z → -∞ | z = 0 | z → +∞ |
|--------|-------|--------|
| g(z) → 0 | g(z) = **0,5** | g(z) → 1 |

> **Analogia:** z é o "termômetro" da regressão linear. A sigmóide converte esse termômetro em uma probabilidade. Temperatura muito baixa (z << 0) → quase certeza de classe 0. Temperatura muito alta (z >> 0) → quase certeza de classe 1.

---

### 5.3 A Fórmula Completa da Regressão Logística

**Passo a passo — como a regressão logística é construída:**

**1. Calcular z (regressão linear):**
```
z = w · x + b
```

**2. Aplicar a sigmóide:**
```
f_{w,b}(x) = g(z) = 1 / (1 + e^(-(w·x + b)))
```

**Fórmula unificada:**
```
f_{w,b}(x) = g(w · x + b) = 1 / (1 + e^(-(w·x+b)))
```

#### Dissecando a fórmula

| Símbolo | Significado |
|---------|-------------|
| **f_{w,b}(x)** | Saída do modelo — a probabilidade predita |
| **w** | Vetor de pesos (parâmetros aprendidos) |
| **b** | Bias (intercepto) |
| **x** | Vetor de features da amostra |
| **e** | Número de Euler ≈ 2,718 |
| **g(z)** | Função sigmóide — garante saída entre 0 e 1 |

**Interpretação probabilística:**
```
f_{w,b}(x) = P(y = 1 | x; w, b)
```
→ "Probabilidade de y ser 1, dado x e os parâmetros w e b"

Como as probabilidades devem somar 1:
```
P(y = 0) + P(y = 1) = 1
```

**Exemplo concreto do PDF:**  
Se x = tamanho do tumor e f_{w,b}(x) = 0.7 → **70% de chance de ser maligno** (y = 1) e 30% de chance de ser benigno (y = 0).

---

### 5.4 Limite de Decisão (Decision Boundary)

A saída da regressão logística é uma probabilidade. Para classificar, precisamos decidir um **limiar (threshold)**:

```
Se f_{w,b}(x) ≥ 0.5 → ŷ = 1
Se f_{w,b}(x) < 0.5 → ŷ = 0
```

**Por que 0.5 é o padrão?**  
g(0) = 0.5, então o threshold equivale a dizer: se z ≥ 0, prediz 1.

O **limiar de decisão** é a região no espaço de features onde z = 0 (onde a probabilidade é exatamente 0.5).

#### Exemplo numérico do PDF

Dados: w₁ = 1, w₂ = 1, b = -3  
Modelo: f(x) = g(w₁x₁ + w₂x₂ + b) = g(x₁ + x₂ - 3)

A fronteira de decisão é onde z = 0:
```
x₁ + x₂ - 3 = 0
x₁ + x₂ = 3   ← reta que separa as classes
```

| Região | Condição | Predição |
|--------|----------|----------|
| Acima da reta | x₁ + x₂ > 3 → z > 0 → g(z) > 0.5 | ŷ = 1 |
| Abaixo da reta | x₁ + x₂ < 3 → z < 0 → g(z) < 0.5 | ŷ = 0 |

> **O modelo aprende os valores de w e b que posicionam essa fronteira de decisão da melhor forma possível nos dados de treino.**

---

### 5.5 Função de Custo — Log-Loss / Entropia Cruzada

**Por que não usar MSE na Regressão Logística?**

Se usarmos MSE com a sigmóide, a função de custo fica **não-convexa** (cheia de mínimos locais), dificultando o gradiente descendente. A solução é a **Log-Loss** (Entropia Cruzada), que garante uma curva convexa (formato de tigela).

**Fórmula da Log-Loss:**
```
J(θ) = (1/m) Σᵢ₌₁ᵐ [-y^(i) · log(h_θ(x^(i))) - (1 - y^(i)) · log(1 - h_θ(x^(i)))]
```

Onde h_θ(x) é a saída da sigmóide (= f_{w,b}(x)).

#### Dissecando a fórmula

| Símbolo | Significado |
|---------|-------------|
| **J(θ)** | Custo total — queremos minimizá-lo |
| **m** | Número de amostras de treinamento |
| **y^(i)** | Rótulo verdadeiro da amostra i (0 ou 1) |
| **h_θ(x^(i))** | Probabilidade predita para a amostra i |
| **log** | Logaritmo natural — penaliza previsões erradas com alta confiança |

#### Como funciona o truque do y^(i)?

Quando y^(i) = 1 (positivo):
```
Custo = -log(h_θ(x))
```
→ Se h_θ → 1 (correto): log(1) = 0 → custo → 0  
→ Se h_θ → 0 (errado): log(0) → -∞ → custo → ∞

Quando y^(i) = 0 (negativo):
```
Custo = -log(1 - h_θ(x))
```
→ Se h_θ → 0 (correto): log(1) = 0 → custo → 0  
→ Se h_θ → 1 (errado): log(0) → -∞ → custo → ∞

| Situação | Custo |
|----------|-------|
| Prediz certo com alta confiança | Próximo de **zero** |
| Prediz errado com alta confiança | Tende a **infinito** — punição severa |

> **Propriedade-chave:** Ao contrário do MSE com sigmóide, a log-loss produz uma superfície de custo **convexa** — o gradiente descendente sempre encontra o mínimo global.

#### Comparação MSE vs Log-Loss

| | MSE | Log-Loss |
|--|-----|---------|
| **Convexidade** | Não com sigmóide | Sim — garante mínimo global |
| **Usado em** | Regressão | Classificação |
| **Penalização** | Quadrática | Logarítmica (mais severa para erros confiantes) |

---

### 5.6 Código Python — Regressão Logística

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Cenário: classificar tumores como benignos (0) ou malignos (1)
# Features: tamanho do tumor, idade do paciente
np.random.seed(42)
n = 200

tamanho = np.random.normal(5, 2, n)
idade = np.random.normal(50, 10, n)

# Maligno se tamanho > 6 ou idade > 60 (com ruído)
y = ((tamanho > 6) | (idade > 60)).astype(int)
y = np.where(np.random.rand(n) < 0.1, 1 - y, y)  # 10% ruído

X = np.column_stack([tamanho, idade])

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar (importante para gradiente descendente)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar modelo
modelo = LogisticRegression(random_state=42)
modelo.fit(X_train_scaled, y_train)

# Avaliar
y_pred = modelo.predict(X_test_scaled)
y_prob = modelo.predict_proba(X_test_scaled)[:, 1]  # probabilidades para classe 1

print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Benigno', 'Maligno']))

# Parâmetros aprendidos
print(f"\nPesos (w): {modelo.coef_[0]}")
print(f"Bias (b): {modelo.intercept_[0]:.4f}")

# Calcular z e probabilidade manualmente para 1 amostra
amostra = np.array([[7.0, 65.0]])  # tumor grande, paciente idoso
amostra_scaled = scaler.transform(amostra)
z = np.dot(amostra_scaled, modelo.coef_.T) + modelo.intercept_
prob = 1 / (1 + np.exp(-z))
print(f"\nAmostra [tamanho=7, idade=65]:")
print(f"  z = {z[0][0]:.4f}")
print(f"  P(maligno) = {prob[0][0]:.4f}  →  {'Maligno' if prob > 0.5 else 'Benigno'}")

# Ajustar threshold (útil em casos com classes desbalanceadas)
threshold = 0.3  # mais sensível a detectar positivos
y_pred_custom = (y_prob >= threshold).astype(int)
print(f"\nAcurácia com threshold=0.3: {accuracy_score(y_test, y_pred_custom):.2f}")
```

---

### 5.7 Questões no Estilo da Prova

---

**Questão 1**

Uma startup de saúde está desenvolvendo um sistema para detectar transações fraudulentas no plano de saúde. O modelo de Regressão Logística produz, para uma determinada transação, o valor f_{w,b}(x) = 0.72. O que isso significa e qual a decisão do modelo com threshold padrão?

a) O modelo classifica a transação como legítima com 72% de confiança  
b) O modelo classifica a transação como fraudulenta, pois 0.72 ≥ 0.5, representando 72% de probabilidade de ser fraude  
c) O modelo erra na predição porque o valor deveria ser 0 ou 1  
d) O valor 0.72 indica o custo da predição, não a classe

> **Resposta: B**  
> **Por que B:** A saída da Regressão Logística representa P(y=1|x; w,b) — a probabilidade da classe positiva (fraude = 1). Com 0.72 ≥ 0.5 (threshold padrão), o modelo prediz ŷ = 1 (fraude). O valor 72% significa que o modelo estima 72% de chance de a transação ser fraudulenta.  
> **Por que não A:** f=0.72 é a probabilidade de y=1 (fraude), não de y=0 (legítima). A probabilidade de ser legítima seria 1 - 0.72 = 0.28 = 28%.  
> **Por que não C:** A saída da sigmóide é sempre um número entre 0 e 1. O valor final y ∈ {0,1} só aparece após aplicar o threshold.  
> **Por que não D:** A função de custo J é definida sobre o conjunto de treino, não sobre uma única predição.

---

**Questão 2**

Um pesquisador treina uma Regressão Logística para classificar e-mails em spam (1) ou não-spam (0). Os parâmetros aprendidos são w₁ = 2, w₂ = -1, b = -4, onde x₁ = frequência da palavra "grátis" e x₂ = número de links no e-mail. Qual é a fronteira de decisão do modelo?

a) x₁ = 2 e x₂ = -1 (cada feature tem seu próprio limite)  
b) 2x₁ - x₂ - 4 = 0, ou seja, x₂ = 2x₁ - 4  
c) f(x) = 0.5 para todos os e-mails  
d) A fronteira só existe se z > 0

> **Resposta: B**  
> **Por que B:** A fronteira de decisão é onde z = 0, ou seja, onde f(x) = 0.5. Calculando: z = w₁x₁ + w₂x₂ + b = 2x₁ + (-1)x₂ + (-4) = 0 → 2x₁ - x₂ - 4 = 0 → x₂ = 2x₁ - 4. Acima dessa reta (onde z > 0), o modelo prediz spam.  
> **Por que não A:** w₁ e w₂ são os pesos da equação linear z — eles não são fronteiras isoladas por feature.  
> **Por que não C:** f(x) = 0.5 é o VALOR da sigmoid na fronteira, não a fronteira em si. A fronteira é a superfície no espaço de features onde f(x) = 0.5.  
> **Por que não D:** z > 0 é a condição para predizer classe 1, mas a fronteira é exatamente onde z = 0.

---

**Questão 3**

Durante o treinamento de um classificador de crédito bancário (inadimplente = 1 / adimplente = 0), um engenheiro compara usar MSE vs Log-Loss como função de custo para a Regressão Logística. Qual a principal vantagem da Log-Loss nesse contexto?

a) A Log-Loss é mais rápida de calcular do que o MSE  
b) A Log-Loss garante uma função de custo **convexa** ao combinar a sigmóide, evitando mínimos locais no gradiente descendente  
c) O MSE não pode ser usado com dados binários  
d) A Log-Loss produz probabilidades calibradas, enquanto o MSE produz apenas 0 ou 1

> **Resposta: B**  
> **Por que B:** Quando se combina MSE com a função sigmóide, a função de custo resultante é **não-convexa** — tem múltiplos mínimos locais onde o gradiente descendente pode ficar preso. A Log-Loss (Entropia Cruzada) com a sigmóide produz uma curva convexa (formato de tigela), garantindo que o gradiente descendente converge para o mínimo global.  
> **Por que não A:** A complexidade computacional das duas funções é comparável. Não é questão de velocidade.  
> **Por que não C:** Tecnicamente, o MSE pode ser calculado com y ∈ {0,1}, mas o resultado é uma superfície não-convexa — o problema é geométrico, não de compatibilidade.  
> **Por que não D:** Tanto MSE quanto Log-Loss produzem probabilidades quando a saída é a sigmóide. A diferença é a forma da superfície de custo.

---

**Questão 4**

Um modelo de Regressão Logística para detectar pacientes com risco cardíaco produz os seguintes resultados com threshold padrão (0.5):
- Acurácia de treino: 92%
- Paciente A (risco real: alto): f(x) = 0.48 → predito como **saudável**
- Paciente B (risco real: baixo): f(x) = 0.51 → predito como **risco**

O cardiologista responsável sugere reduzir o threshold para 0.3. Qual o efeito esperado?

a) A acurácia geral vai aumentar  
b) O modelo vai detectar mais casos positivos (mais sensível), mas vai ter mais falsos positivos também  
c) O modelo vai ignorar todos os pacientes com probabilidade abaixo de 0.3  
d) Reduzir o threshold equivale a treinar novamente o modelo com outros parâmetros

> **Resposta: B**  
> **Por que B:** Reduzir o threshold de 0.5 para 0.3 significa que o modelo vai classificar como "risco" qualquer paciente com P(risco) ≥ 0.3. O Paciente A (f=0.48) passaria a ser corretamente identificado como risco. Porém, pacientes com probabilidade entre 0.30 e 0.49 — que antes eram considerados saudáveis — também seriam marcados como risco, aumentando os falsos positivos. Há um **trade-off entre recall e precisão**.  
> **Por que não A:** A acurácia geral pode até cair, pois mais falsos positivos surgem. O objetivo da mudança não é acurácia, mas **recall** (sensibilidade para não perder casos reais).  
> **Por que não C:** O threshold define onde começa a predição positiva, não onde os casos são ignorados.  
> **Por que não D:** Alterar o threshold é uma decisão pós-treinamento, não retreinamento. Os parâmetros w e b permanecem os mesmos.

---

*Fim da Seção 5 — próxima seção: Validação Cruzada, Viés e Variância, Métricas*

---

## Seção 6 — Validação Cruzada, Viés e Variância, Métricas

> **Fonte:** `6. Validação Cruzada, Viês e Variância, Métricas.pptx.pdf`

---

### 6.1 Conjuntos de Treinamento e Teste

Antes de treinar qualquer modelo, os dados são divididos em dois grupos:

| Conjunto | Proporção típica | Finalidade |
|----------|-----------------|-----------|
| **Treinamento** | 80% | Otimizar a função de custo e ajustar os parâmetros do modelo |
| **Teste** | 20% | Avaliação final — **nunca** usado durante o treino |

**Exemplo do PDF:** Dataset de 10 imóveis (Área × Preço) → 8 para treino, 2 para teste.

> **REGRA DE OURO:** O conjunto de teste é "sagrado" — ele só é usado UMA vez, no final. Usá-lo para tunar hiperparâmetros contamina a avaliação (data leakage).

---

### 6.2 Validação Cruzada (K-Fold Cross-Validation)

**Problema com divisão simples treino/teste:** A divisão pode ser "sortuda" ou "azarada" — o desempenho depende de quais amostras caíram em cada grupo.

**Solução:** K-Fold Cross-Validation — treinar e validar **k vezes** com subconjuntos diferentes.

#### Como funciona o K-Fold (cv = 5):

```
Dados totais: [  Treino (80%)  |  Teste (20%)  ]

O conjunto de TREINO é dividido em k=5 folds:
Split 1: [Val|Tr |Tr |Tr |Tr ] → treina em 4, valida em 1
Split 2: [Tr |Val|Tr |Tr |Tr ] → treina em 4, valida em 1
Split 3: [Tr |Tr |Val|Tr |Tr ] → treina em 4, valida em 1
Split 4: [Tr |Tr |Tr |Val|Tr ] → treina em 4, valida em 1
Split 5: [Tr |Tr |Tr |Tr |Val] → treina em 4, valida em 1

Score final = média dos 5 scores de validação

Avaliação final: modelo retreinado em TODO o treino → avaliado no Teste (20%)
```

#### Fluxo completo do CV no scikit-learn:

```
Dataset → split → [Treino 80% | Teste 20%]
                        ↓
              Cross-Validation (K-fold)
              com candidatos de hiperparâmetros
                        ↓
              Best parameters encontrados
                        ↓
              Retreinar com TODO o treino + best params
                        ↓
              Final evaluation no Teste
```

**Propósito:** Encontrar os **melhores hiperparâmetros** com uso eficiente de dados — cada amostra serve como validação pelo menos uma vez.

#### Variações do K-Fold

| Variação | Quando usar |
|----------|-------------|
| **KFold** | Padrão, folds contíguos. Pode gerar folds desbalanceados |
| **StratifiedKFold** | Classes desbalanceadas — garante mesma proporção de classes em cada fold |
| **ShuffleSplit** | Embaralha antes de dividir — folds com sobreposição possível |
| **StratifiedShuffleSplit** | Combina estratificação + embaralhamento |

> **Regra prática:** Para classificação, sempre prefira **StratifiedKFold** para manter a distribuição das classes em cada fold.

---

### 6.3 Viés e Variância

Viés e variância são os dois tipos fundamentais de erro em modelos de AM.

**Viés (Bias):**
> Erro causado por **suposições errôneas** no algoritmo. Alto viés → o modelo assume uma forma muito simples → perde relações importantes entre features e targets → **underfitting**.

**Variância:**
> Erro causado por **sensibilidade excessiva** às flutuações do conjunto de treinamento. Alta variância → o modelo aprende até o ruído dos dados → não generaliza → **overfitting**.

#### Visualizando viés e variância

| Situação | Regressão | Classificação | Causa |
|----------|-----------|---------------|-------|
| **Alto viés** | Linha reta horizontal em dados curvos | Fronteira linear simples que não separa as classes | Modelo muito simples (underfitting) |
| **Alta variância** | Linha que oscila passando por cada ponto | Fronteira ondulada/irregular que "decora" os dados | Modelo muito complexo (overfitting) |
| **Ideal** | Curva suave que captura a tendência | Fronteira curva que separa as classes corretamente | Complexidade adequada |

#### O Trade-off Viés-Variância

```
Complexidade do modelo aumenta →
    Viés diminui (menos suposições)
    Variância aumenta (mais sensível ao treino)

Objetivo: encontrar o "ponto doce" entre os dois extremos
```

| | Alto Viés | Alta Variância |
|--|-----------|---------------|
| **Erro de treino** | Alto | **Baixo** |
| **Erro de teste** | Alto | **Alto** |
| **Diferença treino-teste** | Pequena | **Grande** |
| **Equivale a** | Underfitting | Overfitting |
| **Solução** | Modelo mais complexo | Regularização, mais dados |

---

### 6.4 Métricas — Regressão

Métricas medem uma característica específica do desempenho do modelo. No scikit-learn, métricas de **distância** (erro) seguem a convenção `neg_*` (negado) para que valores maiores sejam sempre melhores.

#### Tabela completa de métricas de regressão

| Sigla | Nome | Fórmula | Quando usar |
|-------|------|---------|------------|
| **MAE** | Erro Absoluto Médio | (1/n) Σ\|yᵢ - ŷᵢ\| | Erros tratados igualmente; robusto a outliers moderados |
| **MSE** | Erro Quadrático Médio | (1/n) Σ(yᵢ - ŷᵢ)² | Penaliza erros grandes mais severo; sensível a outliers |
| **RMSE** | Raiz do MSE | √MSE | Mesma unidade do alvo; interpretação mais fácil |
| **MSLE** | Erro Log Quadrático Médio | (1/n) Σ(log(1+yᵢ) - log(1+ŷᵢ))² | Dados com crescimento exponencial (vendas, população) |
| **MedAE** | Erro Absoluto Mediano | median(\|y₁-ŷ₁\|, ..., \|yₙ-ŷₙ\|) | Muito robusto a outliers — usa mediana, não média |
| **R²** | Coeficiente de Determinação | 1 - Σ(yᵢ-ŷᵢ)² / Σ(yᵢ-ȳ)² | Proporção da variância explicada pelo modelo |
| **EV** | Variância Explicada | 1 - Var(y-ŷ) / Var(y) | Similar ao R², mas não conta viés sistemático |

#### Dissecando o R²

```
R²(y, ŷ) = 1 - [Σ(yᵢ - ŷᵢ)²] / [Σ(yᵢ - ȳ)²]
```

| Símbolo | Significado |
|---------|-------------|
| **yᵢ** | Valor real da amostra i |
| **ŷᵢ** | Valor predito pelo modelo |
| **ȳ** | Média de todos os valores reais |
| **Numerador** | Variância dos erros do modelo (resíduos) |
| **Denominador** | Variância total dos dados |

| Valor de R² | Interpretação |
|-------------|---------------|
| **R² = 1.0** | Modelo perfeito — zero erro |
| **R² = 0.85** | Modelo explica 85% da variância dos dados |
| **R² = 0.0** | Modelo equivale a prever a média sempre |
| **R² < 0** | Modelo pior do que simplesmente prever a média |

#### R² vs Variância Explicada

| | R² | Variância Explicada |
|--|----|--------------------|
| **Considera viés sistemático** | Sim | Não |
| **Medida mais completa** | Sim | Não |
| **Recomendação** | Preferir R² | Usar EV como complemento |

---

### 6.5 Métricas — Classificação

#### A Matriz de Confusão — Base de Tudo

```
                  PREDITO
                Negativo  Positivo
REAL  Negativo |   TN   |   FP   |
      Positivo |   FN   |   TP   |
```

| Termo | Significado | Impacto |
|-------|-------------|---------|
| **TP** (True Positive) | Predito positivo, era positivo | Acerto na classe relevante |
| **TN** (True Negative) | Predito negativo, era negativo | Acerto na classe não-relevante |
| **FP** (False Positive) | Predito positivo, era negativo | **Erro Tipo I** — alarme falso |
| **FN** (False Negative) | Predito negativo, era positivo | **Erro Tipo II** — miss crítico |

#### As 4 métricas principais de classificação

**Acurácia (Accuracy):**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
> "Qual proporção de todas as predições está correta?"  
> **Limitação:** Enganosa em dados desbalanceados. Ex: 95% das transações são legítimas → modelo que prevê tudo como legítimo tem acurácia 95% mas detecta 0% de fraudes.

**Precisão (Precision):**
```
Precision = TP / (TP + FP)
```
> "Dos que predi como positivos, quantos realmente são?"  
> Alta precisão = poucas falsas alarmes. Usar quando **FP é caro** (ex: spam — não quero perder e-mail legítimo).

**Revocação/Recall/Sensibilidade:**
```
Recall = TP / (TP + FN)
```
> "Dos que realmente são positivos, quantos identifiquei?"  
> Alto recall = poucas detecções perdidas. Usar quando **FN é caro** (ex: câncer — não posso perder caso real).

**F1-Score (Média Harmônica):**
```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```
> Equilíbrio entre precisão e recall. **A média harmônica penaliza valores extremos** — se precisão = 1.0 e recall = 0.0, F1 = 0 (não 0.5).  
> Usar quando há **desequilíbrio entre classes** e tanto FP quanto FN importam.

#### Trade-off Precisão × Recall

```
Threshold baixo (ex: 0.3):
  → Mais positivos preditos → Recall ↑ → Precision ↓
  → Use em: diagnóstico médico (não perder casos)

Threshold alto (ex: 0.7):
  → Menos positivos preditos → Precision ↑ → Recall ↓
  → Use em: filtro de spam (não perder e-mails legítimos)
```

#### Curva ROC e AUC-ROC

A **Curva ROC** (Receiver Operating Characteristic) plota:
- Eixo X: **FPR** (Taxa de Falsos Positivos) = FP / (FP + TN)
- Eixo Y: **TPR** (= Recall) = TP / (TP + FN)
- Cada ponto = um threshold diferente

| Modelo | AUC-ROC | Interpretação |
|--------|---------|---------------|
| Classificador perfeito | **1.0** | TPR=1, FPR=0 para algum threshold |
| Classificador aleatório | **0.5** | Linha diagonal — sem poder discriminativo |
| Modelo ruim | < 0.5 | Pior do que aleatório |

> **AUC-ROC** (Área Sob a Curva): quanto maior, melhor o classificador distingue entre classes. Independe do threshold escolhido — avalia o modelo globalmente.

---

### 6.6 Código Python — Validação Cruzada e Métricas

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      KFold, StratifiedKFold, GridSearchCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score,
                              mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler

# Cenário: detecção de crédito inadimplente (1 = inadimplente, 0 = adimplente)
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2,
                            weights=[0.85, 0.15],  # 85% adimplente — dados desbalanceados
                            random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

modelo = LogisticRegression(random_state=42)

# --- Validação Cruzada simples ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kfold = cross_val_score(modelo, X_train_sc, y_train, cv=kf, scoring='f1')
print(f"KFold F1 (5-fold): {scores_kfold}")
print(f"Média: {scores_kfold.mean():.4f} ± {scores_kfold.std():.4f}")

# --- Stratified KFold (recomendado para classes desbalanceadas) ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skf = cross_val_score(modelo, X_train_sc, y_train, cv=skf, scoring='f1')
print(f"\nStratifiedKFold F1: {scores_skf.mean():.4f} ± {scores_skf.std():.4f}")

# --- GridSearchCV: buscar melhores hiperparâmetros com CV ---
param_grid = {'C': [0.01, 0.1, 1.0, 10.0], 'max_iter': [100, 500]}
grid_search = GridSearchCV(LogisticRegression(random_state=42),
                            param_grid, cv=skf, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_sc, y_train)
print(f"\nMelhores hiperparâmetros: {grid_search.best_params_}")
print(f"Melhor F1 na CV: {grid_search.best_score_:.4f}")

# --- Avaliar modelo final no conjunto de TESTE ---
melhor_modelo = grid_search.best_estimator_
y_pred = melhor_modelo.predict(X_test_sc)
y_prob = melhor_modelo.predict_proba(X_test_sc)[:, 1]

print("\n=== MÉTRICAS DE CLASSIFICAÇÃO ===")
print(f"Acurácia:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precisão:  {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")

print("\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")

# --- Métricas de regressão (exemplo com valores numéricos) ---
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X_r, y_r = make_regression(n_samples=200, n_features=3, noise=15, random_state=42)
X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_r, y_r, test_size=0.2)
reg = LinearRegression().fit(X_r_train, y_r_train)
y_r_pred = reg.predict(X_r_test)

print("\n=== MÉTRICAS DE REGRESSÃO ===")
print(f"MAE:  {mean_absolute_error(y_r_test, y_r_pred):.4f}")
print(f"MSE:  {mean_squared_error(y_r_test, y_r_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_r_test, y_r_pred)):.4f}")
print(f"R²:   {r2_score(y_r_test, y_r_pred):.4f}")
```

---

### 6.7 Questões no Estilo da Prova

---

**Questão 1**

Uma equipe de ML treina um classificador de risco de crédito usando validação cruzada com K-Fold (cv=5) em 1000 amostras. O conjunto de teste (20%) é separado antes. Como o K-Fold é aplicado no processo?

a) Os 1000 dados são divididos em 5 partes iguais, e o modelo é avaliado 5 vezes direto no conjunto de teste final  
b) Os 800 dados de treinamento são divididos em 5 folds: em cada iteração, 640 treinam o modelo e 160 validam, gerando 5 scores que são combinados para encontrar os melhores hiperparâmetros  
c) O K-Fold divide o conjunto de teste em 5 partes para uma avaliação mais robusta  
d) Cada fold treina um modelo diferente que depois vota na predição final

> **Resposta: B**  
> **Por que B:** O K-Fold é aplicado APENAS sobre o conjunto de treino (800 amostras). A cada fold, 4/5 treinam (640) e 1/5 valida (160). Isso é feito 5 vezes com folds diferentes. O resultado médio dos 5 scores guia a seleção de hiperparâmetros. O conjunto de teste (200 amostras) fica intocado até a avaliação final.  
> **Por que não A:** Usar o conjunto de teste durante o K-Fold seria data leakage — a avaliação final ficaria contaminada.  
> **Por que não C:** O K-Fold é aplicado no treino, nunca no teste.  
> **Por que não D:** K-Fold não é ensemble (Bagging/Boosting fazem isso). Aqui, cada fold avalia o mesmo tipo de modelo com diferentes splits.

---

**Questão 2**

Um modelo de detecção de fraude bancária apresenta os seguintes resultados no conjunto de teste (1000 transações, 50 fraudes reais):
- Acurácia: 96%
- Precisão: 60%
- Recall: 30%
- F1-score: 40%

O gerente afirma que "96% de acurácia é excelente". O cientista de dados discorda. Quem está certo e por quê?

a) O gerente — 96% de acurácia é de fato excelente em qualquer cenário  
b) O cientista — com 50/1000 fraudes (5% de positivos), um modelo que prevê TUDO como legítimo já teria 95% de acurácia, mas recall=0%. O modelo está detectando apenas 30% das fraudes reais  
c) O gerente — precisão de 60% confirma que o modelo funciona bem  
d) O cientista — a acurácia deveria ser calculada com K-Fold para ser válida

> **Resposta: B**  
> **Por que B:** Com dados desbalanceados (5% de fraudes), a acurácia é enganosa. Um modelo ingênuo que prediz "legítimo" para tudo teria acurácia 95%. O recall de 30% significa que o modelo perde 70% das fraudes reais — um desastre em detecção de fraudes. A métrica correta aqui é o **F1-score** (40%) ou o **recall** (maximizar detecção de fraudes).  
> **Por que não A:** Acurácia sozinha é enganosa em dados desbalanceados — é a limitação clássica descrita no PDF.  
> **Por que não C:** Precisão de 60% diz que, dos alertas de fraude, 60% são reais. Mas recall 30% diz que o modelo alerta em apenas 30% das fraudes existentes — os outros 70% passam despercebidos.  
> **Por que não D:** O problema não é metodológico (K-Fold), é de escolha de métrica.

---

**Questão 3**

Um modelo de regressão para prever consumo de energia elétrica de prédios obteve R² = 0.92 e MAE = 15 kWh. O engenheiro suspeita de outliers (alguns prédios com consumo anormalmente alto). Qual par de métricas seria mais informativo para investigar o impacto dos outliers?

a) R² e Variância Explicada — ambas medem a mesma coisa  
b) MAE e MedAE — MAE é influenciado por outliers, MedAE usa mediana e é robusto; a diferença entre eles revela o impacto dos outliers  
c) RMSE e MSLE — ambos são insensíveis a outliers  
d) MAE e MSE — MSE penaliza menos os outliers do que o MAE

> **Resposta: B**  
> **Por que B:** O MAE calcula a média das diferenças absolutas — é influenciado por valores extremos. O MedAE usa a mediana — é robusto a outliers. Se MAE >> MedAE, os outliers estão inflacionando a média. Essa diferença quantifica o impacto dos prédios com consumo anormal.  
> **Por que não A:** R² e Variância Explicada são diferentes: R² considera viés sistemático, a Variância Explicada não. Nenhuma das duas investigam outliers diretamente.  
> **Por que não C:** RMSE eleva os erros ao quadrado — é MAIS sensível a outliers, não menos. MSLE aplica log e é melhor para dados exponenciais.  
> **Por que não D:** É o contrário — o MSE (quadrático) penaliza outliers MAIS severamente do que o MAE (linear). Um erro de 10 kWh vira 100 no MSE mas só 10 no MAE.

---

**Questão 4**

Um sistema de diagnóstico de câncer (1 = maligno, 0 = benigno) tem a seguinte matriz de confusão nos dados de teste:

```
              Predito 0   Predito 1
Real 0  (TN=85)  (FP=5)
Real 1  (FN=8)   (TP=2)
```

Calcule Precisão, Recall e F1. Qual é o problema crítico do modelo e que ajuste seria indicado?

a) Precision=29%, Recall=20%, F1=23%. O modelo tem baixo F1 — aumentar a complexidade resolve  
b) Precision=29%, Recall=20%, F1=23%. O problema é o recall extremamente baixo (só 20% dos cânceres são detectados). Reduzir o threshold aumentaria o recall às custas da precisão  
c) Precision=86%, Recall=91%, F1=89%. O modelo é excelente — a acurácia de 87% confirma  
d) Precision=29%, Recall=20%, F1=23%. A solução é remover outliers do conjunto de treinamento

> **Resposta: B**  
> **Calculando:**  
> - Precision = TP/(TP+FP) = 2/(2+5) = **29%**  
> - Recall = TP/(TP+FN) = 2/(2+8) = **20%**  
> - F1 = 2·(0.29·0.20)/(0.29+0.20) = **23%**  
> - Acurácia = (85+2)/(85+5+8+2) = 87% ← enganosa!  
> **Por que B:** O recall de 20% é catastrófico para diagnóstico de câncer: 8 de cada 10 pacientes com câncer são mandados para casa como "saudáveis" (FN). Em cenários de saúde, **FN é o erro mais perigoso**. Reduzir o threshold (ex: de 0.5 para 0.2) faz o modelo ser mais agressivo em predizer positivo, aumentando o recall (mais cânceres detectados) mas aumentando também os FP (mais falsos alarmes — aceitável vs. perder câncer).  
> **Por que não A:** A causa não é complexidade do modelo — é o threshold. Além disso, "aumentar complexidade" é a solução para underfitting, não para threshold incorreto.  
> **Por que não C:** Os valores calculados são 29%/20%/23%, não 86%/91%/89%.  
> **Por que não D:** Outliers no treino não causam baixo recall no teste — o problema aqui é o threshold de decisão.

---

*Fim da Seção 6 — próxima seção: SVM*

---

## Seção 7 — Máquinas Vetores de Suporte (SVM)

---

### 7.1 Visão Geral

**SVM (Support Vector Machines)** é um modelo de AM versátil, fundamentado na **Teoria de Vapnik-Chervonenkis**, que busca a fronteira de decisão com **maior margem possível** entre as classes.

**Aplicações principais:**
- Classificação (linear e não-linear)
- Regressão
- Detecção de anomalias (outliers)

**Diferencial frente a redes neurais:** o SVM resolve um problema de **otimização convexa** — sem mínimos locais, sempre converge para o **mínimo global**.

**Cenário ideal:** dados de alta complexidade em conjuntos de **pequeno a médio porte** (escala ruim em Big Data).

---

### 7.2 Essência Geométrica: Hiperplano e Margem

| Conceito | Definição |
|----------|-----------|
| **Hiperplano** | Fronteira de decisão: linha em 2D, plano em 3D, hiperplano em nD |
| **Margem** | "Corredor" entre as duas classes — SVM busca o corredor **mais largo possível** |
| **Vetores de Suporte** | Pontos **nas bordas** da margem que a definem — os únicos que importam |

> "Qualquer outro classificador linear separa os dados. O SVM separa com a maior folga possível."

**Propriedade dos Vetores de Suporte:** se você apagar todos os outros dados e retreinar apenas com os vetores de suporte, o hiperplano será **exatamente o mesmo**. O modelo é altamente esparso — eficiente em memória pós-treinamento.

---

### 7.3 Formulação Matemática

**Equação do hiperplano:**

```
w^T · x + b = 0
```

| Símbolo | Significado |
|---------|-------------|
| **w** | Vetor de pesos (perpendicular ao hiperplano) |
| **b** | Viés (bias) |
| **‖w‖** | Norma do vetor de pesos |
| **2 / ‖w‖** | Largura total da margem |

**Objetivo de otimização:** maximizar a margem = **minimizar ‖w‖²**, sujeito à restrição de que nenhum ponto caia no lado errado da margem.

> Quanto menor ‖w‖, maior a margem — o SVM resolve isso via programação quadrática convexa.

---

### 7.4 Margem Rígida vs. Margem Suave

#### Margem Rígida (Hard Margin)
- Exige que **todos** os pontos fiquem fora da margem e no lado correto
- Só funciona com dados **linearmente separáveis**
- **Sensível a outliers** — um único ponto fora do lugar pode tornar a separação impossível ou distorcer toda a margem

#### Margem Suave (Soft Margin) — prática real

O hiperparâmetro **C** controla o equilíbrio entre margem larga e erros tolerados:

| C | Comportamento | Risco |
|---|---------------|-------|
| **C alto** (ex: 100) | Margem estreita, penaliza cada erro severamente, tenta acertar tudo no treino | **Overfitting** |
| **C baixo** (ex: 1) | Margem larga, tolera violações, foca em generalização | **Underfitting** |

> Analogia: C alto = professor rígido que não aceita nenhum erro; C baixo = professor tolerante que prefere uma regra geral simples.

---

### 7.5 Classificação Não-Linear: Kernel Trick

Quando as classes são **não-linearmente separáveis** (ex: círculos concêntricos, duas luas), um hiperplano reto nunca separa. A solução é **projetar os dados para dimensões superiores** onde eles se tornem separáveis.

**O problema:** calcular essa projeção explícita para dimensões infinitas é computacionalmente inviável.

**A solução — Kernel Trick (Teorema de Mercer):** o SVM não precisa calcular as coordenadas na dimensão superior. Ele só precisa do **produto escalar** (similaridade) entre os pontos nesse espaço — e o kernel calcula isso diretamente.

#### Kernels disponíveis no Scikit-Learn

| Kernel | Fórmula | Quando usar | Parâmetros |
|--------|---------|-------------|------------|
| **Linear** | K(x,z) = x^T·z | Dados linearmente separáveis, textos, alta dimensionalidade | C |
| **Polinomial** | K(x,z) = (γ·x^T·z + coef0)^d | Curvas polinomiais | C, degree, coef0 |
| **RBF (Gaussiano)** | K(x,z) = exp(−γ·‖x−z‖²) | Caso geral não-linear, mais versátil | C, γ (gamma) |

#### Parâmetro Gamma (γ) no Kernel RBF

γ define o **raio de influência** de cada vetor de suporte:

| γ | Comportamento |
|---|---------------|
| **γ alto** | Raio pequeno → "ilhas" de decisão ao redor de cada ponto → **overfitting** |
| **γ baixo** | Raio grande → fronteira suave e ampla → **underfitting** |

**Fórmula RBF aplicada:**
```
K(x1, x2) = exp(−γ · ‖x1 − x2‖²)
```
Exemplo: x = -1, referência em -2, γ = 0.30  
→ distância = |-1 - (-2)| = 1  
→ K = exp(-0.3 × 1²) = **0.74** (alta similaridade)

---

### 7.6 SVM para Regressão (SVR)

A lógica se **inverte** em relação à classificação:

| | Classificação (SVC) | Regressão (SVR) |
|--|---------------------|-----------------|
| **Objetivo** | Maximizar margem entre classes | Encaixar o **máximo de pontos dentro** da margem |
| **Violação** | Pontos no lado errado | Pontos **fora** da margem |
| **Hiperparâmetro chave** | C | **ε (epsilon)** + C |

**Epsilon (ε):** define a largura da "faixa de tolerância" ao redor da linha de regressão. Pontos dentro de ε não geram erro — o modelo é **ε-insensitive**.

- ε pequeno → faixa estreita → mais vetores de suporte → mais regularização
- ε grande → faixa larga → aceita mais variação → previsão mais suave

---

### 7.7 Implementação no Scikit-Learn

| Tarefa | Classe | Parâmetros principais |
|--------|--------|-----------------------|
| Classificação Linear | `LinearSVC` | C |
| Classificação Não-Linear | `SVC` | C, kernel, gamma, degree |
| Regressão Linear | `LinearSVR` | C, epsilon |
| Regressão Não-Linear | `SVR` | C, kernel, gamma, epsilon |

> **Obrigatório:** normalizar os dados antes de treinar SVM (`StandardScaler`). SVM é sensível à escala das features.

---

### 7.8 Vantagens e Desvantagens

| | Vantagens | Desvantagens |
|-|-----------|--------------|
| **Dimensionalidade** | Funciona bem com features > amostras | Lento para Big Data: O(n²) a O(n³) |
| **Memória** | Esparso — só armazena vetores de suporte | Sem probabilidades diretas (precisa Platt Scaling) |
| **Fronteira** | Kernel Trick permite separações complexas | Escolha errada de kernel → overfitting |
| **Otimização** | Mínimo global garantido (convexa) | Difícil interpretar: "caixa-preta" |

---

### 7.9 Código Python — SVM Completo

```python
import numpy as np
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# ── Cenário: classificar tumores como benignos/malignos ──
np.random.seed(42)
n = 300
tamanho = np.random.normal(5, 2, n)
textura = np.random.normal(10, 3, n)
y = ((tamanho > 6) & (textura > 10)).astype(int)
y = np.where(np.random.rand(n) < 0.08, 1 - y, y)  # 8% de ruído
X = np.column_stack([tamanho, textura])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 1. SVM Linear via Pipeline (boas práticas) ──
pipe_linear = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(C=1.0, random_state=42, max_iter=5000))
])
pipe_linear.fit(X_train, y_train)
print("LinearSVC:", accuracy_score(y_test, pipe_linear.predict(X_test)))

# ── 2. SVM com Kernel RBF ──
pipe_rbf = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])
pipe_rbf.fit(X_train, y_train)
print("SVC RBF:", accuracy_score(y_test, pipe_rbf.predict(X_test)))
print(classification_report(y_test, pipe_rbf.predict(X_test),
                             target_names=['Benigno', 'Maligno']))

# ── 3. GridSearch para otimizar C e gamma ──
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [0.001, 0.01, 0.1, 1]
}
grid = GridSearchCV(pipe_rbf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print("Melhores params:", grid.best_params_)
print("Melhor F1:", grid.best_score_.round(3))

# ── 4. Calcular manualmente a decisão (distância ao hiperplano) ──
# (SVC com probability=True para obter probabilidades — usa Platt Scaling)
pipe_prob = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale',
                probability=True, random_state=42))
])
pipe_prob.fit(X_train, y_train)
amostra = np.array([[7.5, 12.0]])
print("Probabilidade maligno:", pipe_prob.predict_proba(amostra)[0, 1].round(3))
print("Distância ao hiperplano:", pipe_prob.decision_function(amostra)[0].round(3))
```

---

### 7.10 Questões no Estilo da Prova

---

**Questão 1 — Margem e Vetores de Suporte**

Uma empresa de crédito treinou um SVM linear para aprovar ou recusar empréstimos. O engenheiro percebeu que, dos 10.000 clientes no banco de dados, o modelo treinado utiliza apenas 47 pontos para definir sua fronteira de decisão. Esses 47 pontos são chamados de:

a) Hiperplanos de separação  
b) Pontos de regularização  
c) Vetores de suporte  
d) Parâmetros do kernel  

**Resposta: c)**

- **(a) Incorreta.** Hiperplano é a fronteira em si (linha/plano/hiperplano), não os pontos. Há apenas um hiperplano.
- **(b) Incorreta.** Não existe essa terminologia no SVM. Regularização é controlada pelo parâmetro C, não por pontos específicos.
- **(c) CORRETA.** Os vetores de suporte são exatamente os pontos nas bordas da margem que definem o hiperplano. Remover todos os outros dados e retreinar com apenas esses 47 pontos produz o mesmo hiperplano.
- **(d) Incorreta.** Parâmetros do kernel (C, γ) são hiperparâmetros numéricos do modelo, não pontos do dataset.

---

**Questão 2 — Hiperparâmetro C**

Um SVM com kernel RBF foi treinado com C = 1000 para classificar imagens de tumores. O modelo atingiu 99% de acurácia no treino, mas apenas 61% no teste. O engenheiro quer reduzir o overfitting. Qual ajuste é mais indicado?

a) Aumentar ainda mais C para penalizar mais os erros  
b) Reduzir C para permitir uma margem mais larga e tolerante  
c) Trocar o kernel RBF pelo kernel Linear, pois linear nunca faz overfitting  
d) Aumentar o gamma para ampliar o raio de influência dos vetores de suporte  

**Resposta: b)**

- **(a) Incorreta.** Aumentar C estreita ainda mais a margem e força o modelo a acertar todo o treino — piora o overfitting.
- **(b) CORRETA.** C baixo = margem suave e larga = modelo tolera erros no treino em favor da generalização. Reduzir C é o caminho para combater overfitting no SVM.
- **(c) Incorreta.** O kernel Linear também pode fazer overfitting, especialmente com C alto. Além disso, mudar o kernel não resolve o problema raiz (C excessivo).
- **(d) Incorreta.** Aumentar gamma torna a fronteira ainda mais complexa (raio de influência menor → mais "ilhas" → mais overfitting). O efeito é oposto ao desejado.

---

**Questão 3 — Kernel Trick**

O dataset de fraude de cartão de crédito de uma fintech tem padrões em formato de "espiral" — as transações fraudulentas estão entrelaçadas com as legítimas de forma não-linear. O SVM linear falhou (55% de acurácia). Qual recurso do SVM resolve esse problema e como funciona?

a) Aumentar C fará o SVM linear aprender a separação espiral  
b) O Kernel Trick projeta os dados para um espaço de maior dimensão onde eles se tornam linearmente separáveis, sem calcular a projeção explicitamente  
c) O Kernel Trick reduz o número de features, tornando o problema mais simples  
d) Usar regressão SVM (SVR) em vez de classificação resolve padrões não-lineares  

**Resposta: b)**

- **(a) Incorreta.** C não muda a geometria do hiperplano — apenas ajusta a tolerância à violações. Um hiperplano linear nunca separa uma espiral, independente de C.
- **(b) CORRETA.** O Kernel Trick (Teorema de Mercer) permite ao SVM operar como se projetasse os dados em dimensão superior, calculando apenas o produto escalar entre os pontos nesse espaço (sem a projeção explícita). O kernel RBF é ideal para padrões complexos como espirais.
- **(c) Incorreta.** O Kernel Trick **aumenta** a dimensionalidade implicitamente — ele não reduz features. Redução de dimensionalidade é trabalho do PCA/LDA.
- **(d) Incorreta.** SVR é para **regressão** (prever valores contínuos), não para classificação. O problema de fraude é binário (fraude / não-fraude).

---

**Questão 4 — SVM Regressão vs. Classificação**

Um analista precisa usar SVM para **prever o valor de imóveis** (regressão). Ele aumentou o parâmetro ε (epsilon) de 0.1 para 2.0. Qual é o efeito esperado?

a) A fronteira de decisão ficará mais larga, separando melhor as classes  
b) O modelo passará a ter mais vetores de suporte, aumentando a regularização  
c) A faixa de tolerância ao redor da linha de previsão será mais larga — menos pontos gerarão erro, resultando em previsão mais suave com menos vetores de suporte  
d) O ε não afeta o SVR, apenas o SVC  

**Resposta: c)**

- **(a) Incorreta.** "Fronteira de decisão" e "classes" são conceitos de classificação. Em regressão, ε controla a largura da faixa ao redor da linha de previsão, não uma fronteira entre classes.
- **(b) Incorreta.** É o contrário: **reduzir** ε aumenta o número de vetores de suporte (mais pontos ficam fora da faixa estreita). Aumentar ε reduz os vetores de suporte.
- **(c) CORRETA.** ε define a zona de tolerância: pontos dentro de ε não contribuem com erro (ε-insensitive). Aumentar ε → faixa mais larga → menos pontos ficam "fora" → menos vetores de suporte → previsão mais suave (menos complexa).
- **(d) Incorreta.** ε é exclusivo do SVR e é fundamental — controla a largura da margem de tolerância. Não existe no SVC.

---

*Fim da Seção 7 — próxima seção: Árvore de Decisão*

---

## Seção 8 — Árvore de Decisão

### 8.1 Visão Geral

A **Árvore de Decisão** é um algoritmo de aprendizado supervisionado **não-paramétrico** utilizado para **classificação e regressão**. Ela constrói uma estrutura hierárquica (grafo direcionado acíclico) que divide o espaço de características em regiões progressivamente menores e exclusivas por meio de **regras de decisão lógicas simples** (ex: `idade ≤ 27.5`).

**Por que "não-paramétrico"?**  
Ao contrário de modelos paramétricos (ex: Regressão Linear, que sempre aprende apenas β e b), a Árvore de Decisão não assume uma equação prévia — seu número de nós cresce e se molda à complexidade dos dados.

| Tipo de Modelo | Exemplo | Parâmetros |
|---|---|---|
| **Paramétrico (rígido)** | Regressão Linear y = ax + b | Fixos (a e b), independe do tamanho dos dados |
| **Não-Paramétrico (flexível)** | Árvore de Decisão | Varia livremente conforme a dificuldade do dataset |

---

### 8.2 Estrutura e Nomenclatura

```
         [Nó Raiz]          <- todo o dataset; aplica a primeira regra
         /       \
   [Nó Interno]  [Nó Folha] <- Nó Folha: sem mais divisões
    /      \
[Folha] [Folha]
```

| Componente | Descrição | O que contém |
|---|---|---|
| **Nó Raiz** | Ponto de partida do grafo | Todo o conjunto de dados |
| **Nó Interno** | Aplica regra de decisão (`x₁ ≤ 2.5`) | Subconjunto de dados |
| **Nó Folha (Terminal)** | Região final, sem mais divisões | **Classe majoritária** (classificação) ou **média dos valores** (regressão) |

**Leitura de um nó no Scikit-Learn:**

```
petal length (cm) <= 2.45   <- regra de decisão
gini = 0.667                <- impureza do nó
samples = 150               <- quantas instâncias chegam aqui
value = [50, 50, 50]        <- [qtd classe 0, qtd classe 1, qtd classe 2]
class = setosa              <- classe predita (majoritária)
```

---

### 8.3 Algoritmo CART — Como a Árvore é Construída

O algoritmo **CART** (*Classification and Regression Trees*) constrói a árvore de forma **top-down e gananciosa (greedy)**:

> A cada nó, avalia **todas as features** e **todos os pontos de corte possíveis**, escolhendo o par (feature k, limiar tₖ) que produz os subconjuntos mais **homogêneos (puros)**.

#### Função de Custo do CART

```
J(k, tk) = (m_esq / m) * G_esq  +  (m_dir / m) * G_dir
```

**Dissecando a fórmula:**

| Símbolo | Nome | O que significa |
|---|---|---|
| `k` | Feature escolhida | Ex: "idade", "petal length" |
| `tₖ` | Limiar de corte | Ex: `<= 27.5` |
| `m` | Total de instâncias no nó | Tamanho do conjunto atual |
| `m_esq / m` | Proporção no nó esquerdo | Peso ponderado do nó esquerdo |
| `m_dir / m` | Proporção no nó direito | Peso ponderado do nó direito |
| `G_esq`, `G_dir` | Impureza dos filhos | Calculada pelo Gini ou Entropia |

**Regra prática:** O corte com **menor J(k, tₖ)** é escolhido.

---

### 8.4 Métricas de Impureza

#### 8.4.1 Índice Gini (padrão no Scikit-Learn)

```
G_i = 1 - SUM(p_{i,k}²)   para k = 1..n
```

- **Gini = 0** → nó puro (todas as instâncias pertencem à mesma classe)
- **Gini = 0.5** → máxima impureza (classes perfeitamente misturadas, 2 classes 50/50)

**Exemplo do dataset Iris (nó versicolor com 54 amostras, classes: 0 setosa, 49 versicolor, 5 virginica):**
```
G₄ = 1 - (0/54)² - (49/54)² - (5/54)²
   = 1 - 0 - 0.824 - 0.0086
   = 0.168
```

#### 8.4.2 Entropia (Ganho de Informação)

```
H_i = - SUM( p_{i,k} * ln(p_{i,k}) )   onde p_{i,k} != 0
```

Onde `p_{i,k}` = nº de instâncias da classe k / nº de instâncias no i-ésimo nó.

**Mesmo nó, usando entropia:**
```
H = - (49/54)*ln(49/54) - (5/54)*ln(5/54) = 0.308
```

#### Gini vs. Entropia — Quando usar cada um?

| Critério | Gini | Entropia |
|---|---|---|
| **Velocidade** | Mais rápido (sem logaritmo) | Mais lento |
| **Padrão no sklearn** | Sim (`criterion='gini'`) | Não (`criterion='entropy'`) |
| **Formato da árvore** | Tende a isolar classe maior | Tende a produzir árvores mais **balanceadas** |
| **Diferença prática** | Geralmente resultados similares | Útil quando se quer simetria |

#### 8.4.3 Para Regressão: MSE

Na regressão, o CART minimiza o **MSE ponderado** em vez do Gini:

```
J(k, tk) = (m_L / m) * MSE_L  +  (m_R / m) * MSE_R

MSE_No  = SUM( (y_hat_No - y_i)² )   para i no nó
y_hat_No = (1 / m_No) * SUM(y_i)     média dos valores no nó
```

---

### 8.5 Exemplo Numérico Passo a Passo (CART com Gini)

**Cenário:** Prever se cliente compra produto (Sim/Não) com base na Idade.

Dados: `20 (Não), 25 (Não), 30 (Sim), 35 (Não), 40 (Sim), 50 (Sim)` — 3 Não, 3 Sim.

**Passo 1:** CART ordena e testa pontos médios: 22.5, 27.5, 32.5, 37.5, 45.0

**Passo 2:** Calcula Gini ponderado para cada corte:

| Corte | Nó Esq. | Gini Esq. | Nó Dir. | Gini Dir. | **Gini Ponderado** |
|---|---|---|---|---|---|
| Idade <= 32.5 | [20,25,30] → 2 Não, 1 Sim | 0.44 | [35,40,50] → 1 Não, 2 Sim | 0.44 | **(3/6)×0.44 + (3/6)×0.44 = 0.44** |
| **Idade <= 27.5** | [20,25] → 2 Não, 0 Sim (**puro!**) | **0.00** | [30,35,40,50] → 1 Não, 3 Sim | 0.375 | **(2/6)×0 + (4/6)×0.375 = 0.25** ✓ |

**Passo 3:** Corte em 27.5 vence (Gini 0.25 < 0.44). O processo repete recursivamente nos nós filhos.

> **Regra de ouro:** O CART sempre escolhe o corte com **menor Gini ponderado** — maior pureza nos filhos.

---

### 8.6 Regularização e Poda (Pruning)

Sem restrições, o CART divide os dados até que cada folha tenha **1 amostra** (Gini = 0) → **overfitting severo**.

#### Pré-poda (restrições de crescimento)

| Parâmetro | O que faz | Efeito quando aumentado |
|---|---|---|
| `max_depth` | Limita profundidade máxima da árvore | Menos divisões → menos overfitting |
| `min_samples_split` | Mín. de instâncias para permitir divisão de um nó | Nós com poucos dados não se dividem |
| `min_samples_leaf` | Mín. de instâncias que um nó folha deve ter | Evita folhas com 1 amostra |
| `max_features` | Nº máximo de features avaliadas por corte | Reduz variância (usado no Random Forest) |

#### Pós-poda (Poda de Complexidade de Custo)

```
R_alpha(T) = R(T)  +  alpha * |T|
```

| Símbolo | Significado |
|---|---|
| `R(T)` | Erro total da árvore no treinamento |
| `|T|` | Número de folhas (tamanho da árvore) |
| `alpha` | Penalidade de complexidade (hiperparâmetro) |

**Como aplicar:** Aumentar alpha força a eliminação de ramos que contribuem pouco para a redução do erro → árvore menor, mais generalizável. No sklearn: `ccp_alpha`.

---

### 8.7 Árvore para Classificação vs. Regressão

| Aspecto | Classificação | Regressão |
|---|---|---|
| **Classe sklearn** | `DecisionTreeClassifier` | `DecisionTreeRegressor` |
| **Métrica de impureza** | Gini ou Entropia | MSE (ou MAE) |
| **Predição da folha** | Classe majoritária | Média dos valores alvo |
| **Fronteira de decisão** | Regiões de classe | Patamares (degraus) de valor |
| **Critério padrão sklearn** | `criterion='gini'` | `criterion='squared_error'` |

---

### 8.8 Vantagens e Desvantagens

| Vantagens | Desvantagens |
|---|---|
| **Interpretabilidade** (caixa-branca: fácil explicar regras para negócio) | **Alto risco de overfitting** sem poda |
| **Preparo mínimo**: não exige normalização (sem StandardScaler) | **Alta instabilidade**: pequena mudança nos dados → estrutura completamente diferente |
| **Versatilidade**: lida com relações não-lineares e dados mistos | **Fronteiras ortogonais**: só cria cortes perpendiculares aos eixos (dificuldade com padrões diagonais) |
| Suporta classificação multiclasse e regressão | Solução ótima global não garantida (greedy) |

> **Instabilidade na prática:** Remover apenas 1 amostra outlier, ou rotacionar 45° o dataset, pode gerar uma árvore completamente diferente. Por isso usamos **Ensembles (Random Forest, XGBoost)** para mitigar esse problema.

---

### 8.9 Código Python Completo

**Cenário:** Sistema de aprovação de crédito bancário com base em renda e score de crédito.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

np.random.seed(42)
n = 500

# Dados sintéticos: renda (k R$) e score de crédito (0-1000)
renda = np.random.normal(5, 2, n).clip(1, 15)
score = np.random.normal(600, 150, n).clip(100, 1000)
# Aprovado = renda > 4 E score > 550, com 10% de ruído
aprovado = ((renda > 4) & (score > 550)).astype(int)
aprovado = np.where(np.random.rand(n) < 0.10, 1 - aprovado, aprovado)

X = np.column_stack([renda, score])
X_train, X_test, y_train, y_test = train_test_split(
    X, aprovado, test_size=0.2, random_state=42, stratify=aprovado
)

# 1. Árvore sem restrições (overfitting)
tree_deep = DecisionTreeClassifier(random_state=42)
tree_deep.fit(X_train, y_train)
print(f"Sem restrição  — treino: {tree_deep.score(X_train, y_train):.3f} | "
      f"teste: {tree_deep.score(X_test, y_test):.3f}")
print(f"Profundidade: {tree_deep.get_depth()} | Folhas: {tree_deep.get_n_leaves()}")

# 2. Árvore com pré-poda (max_depth)
tree_pruned = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=42)
tree_pruned.fit(X_train, y_train)
print(f"\nCom pré-poda   — treino: {tree_pruned.score(X_train, y_train):.3f} | "
      f"teste: {tree_pruned.score(X_test, y_test):.3f}")
print(classification_report(y_test, tree_pruned.predict(X_test),
                             target_names=['Reprovado', 'Aprovado']))

# 3. Exportar regras legíveis
feature_names = ['renda_k', 'score_credito']
print(export_text(tree_pruned, feature_names=feature_names, max_depth=3))

# 4. GridSearch: encontrar melhor profundidade
param_grid = {
    'max_depth': [2, 3, 4, 5, 6, None],
    'min_samples_leaf': [1, 5, 10, 20],
    'criterion': ['gini', 'entropy']
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"\nMelhores hiperparâmetros: {grid.best_params_}")

# 5. Importância das features
best_tree = grid.best_estimator_
for name, imp in zip(feature_names, best_tree.feature_importances_):
    print(f"  {name}: {imp:.3f}")

# 6. Pós-poda com ccp_alpha
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[::5]  # amostra de alphas
scores_teste = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    clf.fit(X_train, y_train)
    scores_teste.append(clf.score(X_test, y_test))
melhor_alpha = ccp_alphas[np.argmax(scores_teste)]
print(f"\nMelhor ccp_alpha: {melhor_alpha:.5f}")

# 7. Predict_proba
cliente_novo = np.array([[6.5, 720]])  # renda 6.5k, score 720
proba = tree_pruned.predict_proba(cliente_novo)[0]
print(f"\nCliente (renda=6.5k, score=720):")
print(f"  P(Reprovado) = {proba[0]:.3f} | P(Aprovado) = {proba[1]:.3f}")
print(f"  Decisão: {'Aprovado' if proba[1] > 0.5 else 'Reprovado'}")

# 8. Árvore de Regressão
reg = DecisionTreeRegressor(max_depth=3, random_state=42)
reg.fit(score.reshape(-1, 1), renda)
print(f"\nRegressão R²: {reg.score(score.reshape(-1, 1), renda):.3f}")
print(f"Score=700 → renda estimada: R$ {reg.predict([[700]])[0]:.2f}k")
```

---

### 8.10 Questões no Estilo da Prova

---

**Questão 1**

Um analista treinou uma Árvore de Decisão para classificar clientes como "churn" ou "retido" e observou: acurácia de treino = 99%, acurácia de teste = 68%. Ao inspecionar o modelo, verificou que a árvore tem profundidade 25 e 312 folhas para um dataset de 1.000 amostras. Qual é o diagnóstico correto e a ação mais adequada?

a) O modelo está em underfitting; aumentar `max_depth` resolverá o problema.  
b) O modelo está em overfitting; aplicar pré-poda com `max_depth` ou `min_samples_leaf` ou pós-poda com `ccp_alpha` resolverá o problema.  
c) O modelo está em overfitting; normalizar os dados com `StandardScaler` resolverá o problema.  
d) O modelo está corretamente treinado; a diferença entre treino e teste é esperada em datasets pequenos.

**Resposta: b)**

- **(a) Incorreta.** Acurácia de treino = 99% descarta underfitting (modelo subajustado teria erro alto no treino também). Aumentar `max_depth` pioraria ainda mais o overfitting.
- **(b) Correta.** Treino 99%, teste 68% com árvore de profundidade 25 é overfitting textbook — a árvore "decorou" os dados de treino. Pré-poda (`max_depth`, `min_samples_leaf`) ou pós-poda (`ccp_alpha`) são as soluções corretas.
- **(c) Incorreta.** Árvore de Decisão **não** exige normalização — ela é invariante a escala porque usa apenas comparações `<= threshold`. Normalizar não alteraria o resultado.
- **(d) Incorreta.** Uma diferença de 31 pontos percentuais entre treino e teste não é esperada ou aceitável; é sinal claro de overfitting severo.

---

**Questão 2**

Ao treinar uma Árvore de Decisão com `criterion='gini'` em um nó com 100 instâncias (60 da classe A e 40 da classe B), qual é o valor do índice Gini desse nó?

a) 0.48  
b) 0.24  
c) 0.60  
d) 0.50

**Resposta: a)**

**Cálculo:**
```
p_A = 60/100 = 0.6
p_B = 40/100 = 0.4
G = 1 - (0.6)² - (0.4)² = 1 - 0.36 - 0.16 = 0.48
```

- **(a) Correta.** G = 1 − 0.6² − 0.4² = 0.48. O nó tem impureza moderada.
- **(b) Incorreta.** 0.24 seria o resultado de G = 1 − (0.8)² − (0.2)² (distribuição 80/20), não 60/40.
- **(c) Incorreta.** 0.60 não tem relação com o cálculo; confunde a proporção da classe com a impureza.
- **(d) Incorreta.** Gini = 0.50 é a **máxima impureza** possível para 2 classes (proporção 50/50), não 60/40.

---

**Questão 3**

Um cientista de dados treinou uma Árvore de Decisão para regressão (prever preço de imóvel) e outra para classificação (aprovação de crédito). Qual afirmação descreve corretamente a diferença fundamental entre os dois modos?

a) A árvore de regressão usa o índice Gini como critério de divisão; a de classificação usa o MSE.  
b) A árvore de regressão prediz a **média dos valores alvo** das instâncias na folha, minimizando o MSE; a de classificação prediz a **classe majoritária**, minimizando o Gini ou a Entropia.  
c) A árvore de regressão não sofre overfitting pois prediz valores contínuos; apenas a de classificação necessita de poda.  
d) A árvore de regressão divide os dados minimizando a Entropia; a de classificação divide minimizando o MSE.

**Resposta: b)**

- **(a) Incorreta.** Invertido: Gini/Entropia são para **classificação**; MSE é para **regressão**.
- **(b) Correta.** Na regressão, a folha retorna `ŷ_nó = média(yᵢ)` e o critério de divisão é o MSE ponderado. Na classificação, retorna a classe com mais instâncias e divide por Gini ou Entropia.
- **(c) Incorreta.** Árvores de regressão sofrem overfitting tanto quanto as de classificação — sem restrições, cada folha terá 1 amostra com MSE = 0 no treino, mas péssima generalização.
- **(d) Incorreta.** Totalmente invertido em relação à alternativa (a).

---

**Questão 4**

Durante uma apresentação, um engenheiro demonstrou que ao **rotacionar 45°** os dados de treinamento de uma Árvore de Decisão, o modelo gerou fronteiras de decisão completamente diferentes e com desempenho muito inferior. Qual limitação da Árvore de Decisão esse fenômeno ilustra, e qual abordagem mitiga esse problema?

a) O algoritmo CART é quadrático e não escala bem; a solução é usar Regressão Logística.  
b) A Árvore de Decisão cria apenas **fronteiras ortogonais** (perpendiculares aos eixos), gerando alta instabilidade a rotações; Ensembles como **Random Forest** mitigam esse problema.  
c) O overfitting causou a instabilidade; aumentar `min_samples_leaf` tornaria o modelo invariante a rotações.  
d) A normalização com `StandardScaler` é obrigatória na Árvore de Decisão; sem ela o modelo é instável a transformações geométricas.

**Resposta: b)**

- **(a) Incorreta.** O problema não é custo computacional, mas a natureza dos cortes (ortogonais). Regressão Logística tem fronteiras lineares, não resolve padrões complexos rotacionados.
- **(b) Correta.** A Árvore de Decisão só cria divisões `feature_i <= threshold`, que são **perpendiculares aos eixos**. Ao rotacionar os dados, a fronteira diagonal "ótima" exige múltiplos degraus, gerando modelos piores. O **Random Forest** combina muitas árvores com subamostras aleatórias, reduzindo a variância e a instabilidade geométrica.
- **(c) Incorreta.** Poda reduz overfitting, mas **não** torna a árvore invariante a rotações. As fronteiras continuarão ortogonais independentemente de `min_samples_leaf`.
- **(d) Incorreta.** Árvore de Decisão **não exige** normalização — é uma das suas principais vantagens. A instabilidade a rotações é estrutural (fronteiras ortogonais), não relacionada à escala dos dados.

---

---

## Seção 9 — Aprendizado Ensemble

### 9.1 Visão Geral — A Sabedoria das Multidões

O **Aprendizado Ensemble** parte de um princípio filosófico antigo: perguntar a milhares de pessoas aleatórias e agregar as respostas frequentemente supera a opinião de um único especialista. Esse conceito — chamado de *sabedoria das multidões* — foi formalizado por **Aristóteles → Francis Galton → James Surowiecki** e é o fundamento teórico dos métodos ensemble em ML.

**Definições fundamentais:**

| Termo | Significado |
|---|---|
| **Ensemble** | Grupo de preditores (classificadores ou regressores) |
| **Aprendizado Ensemble** | Técnica que agrega previsões de múltiplos modelos |
| **Método Ensemble** | Algoritmo específico de aprendizado ensemble |
| **Weak Learner** | Preditor individualmente fraco (pouco melhor que aleatório) |
| **Strong Learner** | Modelo final resultante da combinação de weak learners |

**Princípio central:** ao agregar as previsões de um grupo de preditores, você obtém previsões **melhores** do que com o melhor preditor individual.

---

### 9.2 Por que Funciona?

Modelos individuais sofrem de um de dois problemas:

- **Viés alto (Underfitting):** modelo simples demais, não captura a complexidade dos dados.
- **Variância alta (Overfitting):** modelo complexo demais, decora o treino, falha em dados novos.

O ensemble equilibra esse trade-off:

> Mesmo que cada weak learner cometa erros, a **probabilidade de que todos cometam o mesmo erro** é baixa. A agregação cancela erros descorrelacionados.

**Resultado prático do Bagging:** cada preditor individual tem viés ligeiramente mais alto do que treinado no conjunto completo, mas a agregação **reduz tanto o viés quanto a variância** — especialmente a variância.

---

### 9.3 Classificadores de Votação

A forma mais simples de ensemble combina múltiplos classificadores treinados sobre os **mesmos dados** (modelos diferentes) e agrega por votação.

#### Hard Voting vs Soft Voting

| Aspecto | Hard Voting | Soft Voting |
|---|---|---|
| **Mecanismo** | Conta votos de classe (moda) | Soma probabilidades por classe |
| **Entrada usada** | Classe predita por cada modelo | `predict_proba` de cada modelo |
| **Exemplo** | Clf1=A, Clf2=A, Clf3=B → **A** | P(A)=0,6+0,7+0,4=1,7 > P(B)=1,3 → **A** |
| **Vantagem** | Simples, funciona com qualquer classificador | Mais preciso — usa confiança dos modelos |
| **Desvantagem** | Ignora a confiança de cada preditor | Exige que todos os modelos tenham `predict_proba` |
| **Sklearn** | `voting='hard'` | `voting='soft'` |

**Exemplo numérico do Soft Voting:**
```
Clf1: P(A) = 0,60 | P(B) = 0,40
Clf2: P(A) = 0,70 | P(B) = 0,30
Clf3: P(A) = 0,40 | P(B) = 0,60
────────────────────────────────
Soma A = 1,70  |  Soma B = 1,30
→ Previsão final: Classe A
```

**Lei dos Grandes Números:** os classificadores de votação baseiam-se na LGN — com mais preditores independentes, a média das previsões converge para o valor esperado real. Quanto mais preditores, mais estável e preciso o ensemble.

---

### 9.4 Bagging e Pasting

Outra abordagem usa o **mesmo algoritmo base** para cada preditor, mas treina cada um em **subconjuntos aleatórios diferentes** dos dados.

#### Diferença fundamental

| Característica | **Bagging** | **Pasting** |
|---|---|---|
| Nome completo | Bootstrap Aggregating | Pasting |
| Amostragem | **Com reposição** (bootstrap) | **Sem reposição** |
| Instâncias repetidas | Sim, no mesmo preditor | Não (cada instância aparece ≤1x por preditor) |
| Objetivo | Reduzir variância, evitar overfitting | Reduzir variância (menos diversidade) |
| Uso prático | Mais comum (Random Forest usa Bagging) | Menos comum |

**Analogia do Bagging:** imagine um baralho de 10 cartas. Você retira uma, anota e **devolve antes da próxima retirada** — a mesma carta pode ser sorteada várias vezes.

#### Função de Agregação

| Tarefa | Função de Agregação |
|---|---|
| Classificação | **Moda** (hard voting) |
| Regressão | **Média** dos valores preditos |

**Por que Bagging/Pasting escalam bem?** Os preditores são treinados em **paralelo** (diferentes CPUs ou servidores), e as previsões também podem ser feitas em paralelo — escalabilidade excelente.

**Efeito na curva viés-variância:**
- Cada preditor individual: viés ligeiramente mais alto (subconjunto menor)
- Conjunto agregado: viés semelhante ao modelo único, mas **variância significativamente menor**

---

### 9.5 Boosting

O Boosting converte vários **weak learners** em um único **strong learner**, com objetivo central de **reduzir o viés** (ao contrário do Bagging, que foca na variância).

**Diferença crítica frente ao Bagging:** o Boosting opera de forma **sequencial** — cada novo preditor foca nos erros do anterior.

#### 9.5.1 Weak Learners e Decision Stumps

- **Weak learner:** preditor com desempenho apenas ligeiramente melhor que aleatoriedade.
- **Decision Stump:** árvore de decisão com `max_depth=1` — apenas **uma única divisão**.
- **Por que usar stumps?** Raramente sofrem overfitting. A força do modelo final vem da **combinação inteligente** de centenas de modelos simples, não da complexidade de um único.

#### 9.5.2 AdaBoost (Adaptive Boosting)

A correção sequencial no AdaBoost ocorre via **ajuste de pesos das instâncias**:

```
PASSO 1: Treinar Classificador Base → avaliar previsões
PASSO 2: Aumentar peso das instâncias classificadas INCORRETAMENTE
PASSO 3: Treinar próximo classificador com pesos atualizados
PASSO 4: Repetir N vezes
RESULTADO: Soma ponderada (modelos mais precisos têm maior peso)
```

**Fluxo visual:**
```
Dataset original
    ↓
Weak Learner #1 → classifica → errou alguns pontos
    ↓ (pontos errados ganham peso maior)
Weighted Dataset
    ↓
Weak Learner #2 → foca nos casos difíceis
    ↓
... repete ...
    ↓
Strong Learner (combinação ponderada de todos)
```

#### 9.5.3 Gradient Boosting (GBM)

Em vez de ajustar pesos de instâncias, o GBM usa **gradiente descendente sobre os erros residuais**:

```
PASSO 1: Modelo 1 faz previsão y_hat₁
PASSO 2: Calcular resíduo r₁ = y - y_hat₁
PASSO 3: Modelo 2 aprende a prever r₁
PASSO 4: Previsão atualizada: y_hat₂ = y_hat₁ + η·r₂
PASSO 5: Calcular novo resíduo r₂ = y - y_hat₂
PASSO N: Previsão final = soma ponderada de todos os modelos
```

Onde **η (learning rate)** controla a contribuição de cada modelo (taxa de aprendizado).

**Derivações modernas do GBM** (vencedores de Kaggle):

| Algoritmo | Características |
|---|---|
| **XGBoost** | Regularização L1/L2, poda de árvores, paralelismo em colunas |
| **LightGBM** | Cresce a árvore por folhas (leaf-wise), muito rápido em dados grandes |
| **CatBoost** | Trata variáveis categóricas nativamente, ordered boosting |

#### Comparação AdaBoost vs Gradient Boosting

| Aspecto | AdaBoost | Gradient Boosting |
|---|---|---|
| **Mecanismo de correção** | Ajusta **pesos das instâncias** | Aprende os **resíduos** diretamente |
| **Base teórica** | Votação adaptativa | Gradiente descendente no espaço funcional |
| **Sensibilidade a outliers** | Alta (outliers ganham peso crescente) | Moderada (depende da função de perda) |
| **Hiperparâmetro chave** | `n_estimators`, `learning_rate` | `n_estimators`, `learning_rate`, `max_depth` |
| **Sklearn** | `AdaBoostClassifier` | `GradientBoostingClassifier` |
| **Previsão final** | Soma ponderada (peso = precisão do modelo) | Soma ponderada (cada modelo prevê o resíduo) |

---

### 9.6 Stacking (Stacked Generalization)

O Stacking combina modelos de **naturezas completamente diferentes** (heterogêneo), introduzindo um modelo de agregação inteligente chamado **Blender** (ou Meta-modelo).

#### Fluxo do Stacking

```
Conjunto de Treino
    ↓ (dividido em 2 subconjuntos)
Subset 1 ──────────→ Treina modelos base (Layer 1)
                      ex: SVM, Regressão Logística, KNN

Subset 2 ──────────→ Modelos base fazem previsões
                      (dados nunca vistos → previsões "limpas")
                      ↓
                   Novo dataset: features = previsões da Layer 1
                      ↓
                   Treina o BLENDER (Meta-modelo)

Nova instância:
    → Layer 1 prevê → previsões viram features → Blender faz previsão final
```

#### Por que usar Hold-out Set para treinar o Blender?

Se o Blender fosse treinado com as mesmas previsões usadas para treinar a Layer 1, os modelos base estariam "decorando" seus próprios erros — causando **data leakage** e overfitting do meta-modelo.

#### Stacking Multicamadas

| Camada | Função |
|---|---|
| **Layer 1** | Modelos base (SVM, KNN, Reg. Logística...) — treinados no Subset 1 |
| **Layer 2** | Blenders intermediários — treinados nas previsões do Layer 1 (Subset 2) |
| **Layer 3** | Meta-modelo final — treinado nas previsões do Layer 2 (Subset 3) |

**Regra:** para N camadas, o conjunto de treino deve ser dividido em N+1 subconjuntos.

---

### 9.7 Tabela Comparativa — As Famílias do Ensemble

| Característica | Bagging & Pasting | Boosting | Stacking |
|---|---|---|---|
| **Arquitetura** | Homogênea (mesmo modelo base) | Homogênea (mesmo modelo base) | Heterogênea (modelos diferentes) |
| **Treinamento** | Paralelo (independentes) | Sequencial (um corrige o outro) | Camadas (base paralela, blender depois) |
| **Objetivo principal** | Reduzir Variância (evitar overfitting) | Reduzir Viés (fracos → fortes) | Combinação ótima de diferentes modelos |
| **Agregação final** | Votação simples ou Média | Soma Ponderada (melhor modelo → maior peso) | Meta-modelo (Blender aprende a combinação) |
| **Exemplos clássicos** | Random Forest | AdaBoost, XGBoost, LightGBM | Arquitetura customizada |
| **Paralelizável?** | Sim | Não (dependência sequencial) | Parcialmente (Layer 1 sim, Blender não) |
| **Risco principal** | Viés residual | Overfitting (muitos estimadores) | Data leakage se hold-out não for usado |
| **Sklearn** | `BaggingClassifier` | `AdaBoostClassifier`, `GradientBoostingClassifier` | `StackingClassifier` |

---

### 9.8 Código Python — Ensemble Completo (Cenário: Detecção de Fraude)

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score

# ── Dados sintéticos: detecção de fraude (binário, desbalanceado) ──
np.random.seed(42)
X, y = make_classification(
    n_samples=2000, n_features=10, n_informative=6,
    n_redundant=2, weights=[0.85, 0.15], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ══════════════════════════════════════════════════════
# 1. VOTING CLASSIFIER — Hard e Soft Voting
# ══════════════════════════════════════════════════════
lr  = LogisticRegression(max_iter=1000, random_state=42)
svm = SVC(probability=True, random_state=42)     # probability=True para soft voting
dt  = DecisionTreeClassifier(max_depth=4, random_state=42)

hard_voting = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('dt', dt)],
    voting='hard'
)
soft_voting = VotingClassifier(
    estimators=[('lr', lr), ('svm', svm), ('dt', dt)],
    voting='soft'
)

hard_voting.fit(X_train, y_train)
soft_voting.fit(X_train, y_train)

print("=== Voting Classifiers ===")
print(f"Hard Voting AUC: {roc_auc_score(y_test, hard_voting.predict(X_test)):.4f}")
print(f"Soft Voting AUC: {roc_auc_score(y_test, soft_voting.predict_proba(X_test)[:,1]):.4f}")

# ══════════════════════════════════════════════════════
# 2. BAGGING — com Decision Tree base
# ══════════════════════════════════════════════════════
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    max_samples=0.8,        # 80% das instâncias por bootstrap
    bootstrap=True,         # com reposição = Bagging
    n_jobs=-1,
    random_state=42
)
bagging.fit(X_train, y_train)
bagging_auc = roc_auc_score(y_test, bagging.predict_proba(X_test)[:,1])
print(f"\nBagging (100 DTs) AUC: {bagging_auc:.4f}")

# Pasting: bootstrap=False
pasting = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    bootstrap=False,        # sem reposição = Pasting
    n_jobs=-1,
    random_state=42
)
pasting.fit(X_train, y_train)
pasting_auc = roc_auc_score(y_test, pasting.predict_proba(X_test)[:,1])
print(f"Pasting (100 DTs) AUC: {pasting_auc:.4f}")

# ══════════════════════════════════════════════════════
# 3. ADABOOST — stumps sequenciais com ajuste de pesos
# ══════════════════════════════════════════════════════
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Decision Stump
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)
ada.fit(X_train, y_train)
ada_auc = roc_auc_score(y_test, ada.predict_proba(X_test)[:,1])
print(f"\nAdaBoost (200 stumps, lr=0.5) AUC: {ada_auc:.4f}")

# ══════════════════════════════════════════════════════
# 4. GRADIENT BOOSTING — aprende resíduos iterativamente
# ══════════════════════════════════════════════════════
gbm = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,     # taxa de aprendizado (η)
    max_depth=3,            # árvores rasas para reduzir overfitting
    subsample=0.8,          # Stochastic GB: 80% dos dados por iteração
    random_state=42
)
gbm.fit(X_train, y_train)
gbm_auc = roc_auc_score(y_test, gbm.predict_proba(X_test)[:,1])
print(f"GBM (200 est., lr=0.05, depth=3) AUC: {gbm_auc:.4f}")

# Importância das features no GBM
feature_importance = gbm.feature_importances_
top3_idx = np.argsort(feature_importance)[::-1][:3]
print("\nTop 3 features (GBM):")
for i in top3_idx:
    print(f"  Feature {i}: {feature_importance[i]:.4f}")

# ══════════════════════════════════════════════════════
# 5. STACKING — blender aprende a combinar modelos base
# ══════════════════════════════════════════════════════
estimadores_base = [
    ('lr',  LogisticRegression(max_iter=1000, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('dt',  DecisionTreeClassifier(max_depth=4, random_state=42)),
]
blender = LogisticRegression(max_iter=1000, random_state=42)

stacking = StackingClassifier(
    estimators=estimadores_base,
    final_estimator=blender,
    cv=5,              # cross-val para gerar previsões "limpas" (evita data leakage)
    passthrough=False  # blender só recebe previsões da Layer 1
)
stacking.fit(X_train, y_train)
stack_auc = roc_auc_score(y_test, stacking.predict_proba(X_test)[:,1])
print(f"\nStacking (LR+KNN+DT → LR blender) AUC: {stack_auc:.4f}")

# ══════════════════════════════════════════════════════
# 6. COMPARAÇÃO FINAL
# ══════════════════════════════════════════════════════
print("\n" + "="*50)
print("COMPARAÇÃO DE AUC — Detecção de Fraude")
print("="*50)
modelos = {
    'Hard Voting':     roc_auc_score(y_test, hard_voting.predict(X_test)),
    'Soft Voting':     roc_auc_score(y_test, soft_voting.predict_proba(X_test)[:,1]),
    'Bagging':         bagging_auc,
    'Pasting':         pasting_auc,
    'AdaBoost':        ada_auc,
    'Gradient Boost':  gbm_auc,
    'Stacking':        stack_auc,
}
for nome, auc in sorted(modelos.items(), key=lambda x: -x[1]):
    print(f"  {nome:<20} AUC = {auc:.4f}")
```

---

### 9.9 Questões no Estilo da Prova

**Questão 1.** Um cientista de dados treina 5 classificadores independentes (Regressão Logística, SVM, Árvore de Decisão, KNN e Naive Bayes), cada um com acurácia de 75%. Ele combina as previsões escolhendo a classe mais votada. Qual método ele está usando, e por que o resultado tende a ser melhor que 75%?

**(a)** Boosting — porque cada modelo corrige os erros do anterior sequencialmente.
**(b)** Hard Voting — porque a Lei dos Grandes Números garante que a moda de previsões independentes converge para a classe correta.
**(c)** Soft Voting — porque as probabilidades são somadas, dando mais peso a modelos confiantes.
**(d)** Stacking — porque um meta-modelo aprende a combinar as previsões de forma ótima.

**Resposta: (b)**
- **(a) Incorreta.** Boosting é sequencial e usa um único tipo de modelo base com ajuste de pesos/resíduos. O cenário descrito usa modelos independentes de tipos diferentes.
- **(b) Correta.** Escolher a classe **mais votada (moda)** é exatamente o **Hard Voting**. Com classificadores independentes que cada um erra em casos diferentes, a probabilidade de a maioria errar simultaneamente é baixa — isso é a Lei dos Grandes Números aplicada ao ensemble.
- **(c) Incorreta.** Soft Voting soma **probabilidades** (`predict_proba`), não conta votos de classe. O enunciado diz "classe mais votada", que é Hard Voting.
- **(d) Incorreta.** Stacking usa um meta-modelo (Blender) treinado nas previsões dos modelos base em um hold-out set. O cenário não menciona nada disso — é uma simples votação por maioria.

---

**Questão 2.** No contexto de Bagging, um conjunto de dados com 1.000 amostras gera subconjuntos bootstrap de 1.000 amostras cada (com reposição). Qual afirmação está correta sobre esses subconjuntos?

**(a)** Cada subconjunto contém exatamente as 1.000 amostras originais, apenas em ordem diferente.
**(b)** Em média, cerca de 63,2% das amostras originais aparecem em cada subconjunto bootstrap, e as demais (~36,8%) podem ser usadas como conjunto de validação Out-of-Bag.
**(c)** O subconjunto bootstrap garante que nenhuma amostra se repita dentro do mesmo preditor, distinguindo-se assim do Pasting.
**(d)** O Bagging usa amostragem sem reposição, enquanto o Pasting usa com reposição.

**Resposta: (b)**
- **(a) Incorreta.** Com reposição, algumas amostras aparecem múltiplas vezes e outras não aparecem. Não é apenas uma reordenação.
- **(b) Correta.** A probabilidade de uma amostra **não** ser selecionada em uma tentativa é `(1 - 1/n)`. Para n amostras com n sorteios, isso converge para `e⁻¹ ≈ 36,8%`. Portanto, ~63,2% aparecem e ~36,8% ficam de fora — estas últimas formam o **Out-of-Bag (OOB) set**, que pode ser usado como validação gratuita sem necessidade de um conjunto separado.
- **(c) Incorreta.** É exatamente o contrário: no **Bagging** (com reposição), a mesma amostra *pode* aparecer múltiplas vezes no mesmo preditor. É o **Pasting** (sem reposição) que garante que cada amostra apareça no máximo uma vez por preditor.
- **(d) Incorreta.** Invertido. **Bagging = com reposição (bootstrap)**. **Pasting = sem reposição**.

---

**Questão 3.** Sobre AdaBoost e Gradient Boosting, qual é a diferença fundamental na forma como cada um corrige os erros do modelo anterior?

**(a)** AdaBoost treina modelos em paralelo enquanto Gradient Boosting os treina sequencialmente.
**(b)** AdaBoost aumenta os **pesos das instâncias** mal classificadas; Gradient Boosting treina cada novo modelo para prever os **resíduos** do modelo anterior.
**(c)** AdaBoost usa árvores profundas (max_depth=10) enquanto Gradient Boosting usa stumps (max_depth=1).
**(d)** AdaBoost reduz o viés do modelo; Gradient Boosting reduz a variância.

**Resposta: (b)**
- **(a) Incorreta.** Ambos são **sequenciais** — essa é a característica fundamental do Boosting. O treinamento paralelo é característica do Bagging.
- **(b) Correta.** Essa é a distinção central cobrada em prova: **AdaBoost** corrige erros ajustando os **pesos das instâncias de treinamento** (exemplos errados ficam "mais pesados" para o próximo modelo). **Gradient Boosting** corrige erros treinando o próximo modelo para prever diretamente os **resíduos** (y − ŷ), usando gradiente descendente no espaço funcional. XGBoost, LightGBM e CatBoost são variantes modernas do GBM.
- **(c) Incorreta.** É o contrário: **AdaBoost** tipicamente usa **Decision Stumps** (max_depth=1) como weak learners. O Gradient Boosting usa árvores rasas, mas não necessariamente stumps — max_depth=3 é comum.
- **(d) Incorreta.** **Ambos** têm como objetivo principal a **redução do viés** (transformar weak learners em strong learner). A redução de variância é objetivo do **Bagging**.

---

**Questão 4.** Um engenheiro de ML implementa um Stacking com três modelos base (SVM, KNN e Regressão Logística) e um Blender (outra Regressão Logística). Ele treina todos os modelos no conjunto de treinamento completo e usa as previsões desses mesmos dados para treinar o Blender. Qual é o problema dessa abordagem?

**(a)** Stacking exige que todos os modelos base sejam do mesmo tipo; usar SVM, KNN e Regressão Logística juntos é inválido.
**(b)** O Blender sofrerá **data leakage** — ele é treinado em previsões otimistas (os modelos base "memorizaram" esses dados), resultando em overfitting e baixo desempenho em dados novos.
**(c)** O número de modelos base é insuficiente; Stacking exige pelo menos 10 estimadores na Layer 1.
**(d)** Regressão Logística não pode ser usada como Blender — o meta-modelo deve ser sempre uma Árvore de Decisão.

**Resposta: (b)**
- **(a) Incorreta.** O Stacking é **heterogêneo por design** — sua vantagem é exatamente combinar modelos de naturezas completamente diferentes (SVM, KNN, Regressão Logística etc.). Isso é correto e desejável.
- **(b) Correta.** Se os modelos base são treinados e avaliados **nos mesmos dados**, suas previsões serão artificialmente boas (overfitting nos dados de treino). O Blender aprende a combinar previsões irrealisticamente precisas, e falha em dados novos. A solução correta é usar um **hold-out set** ou **cross-validation** (como faz o `StackingClassifier` do sklearn com `cv=5`) para garantir que o Blender aprenda com previsões "limpas" — feitas em dados que os modelos base nunca viram.
- **(c) Incorreta.** Não existe requisito mínimo de 10 estimadores para Stacking. Até 2 modelos base já formam um Stacking válido.
- **(d) Incorreta.** O meta-modelo (Blender) pode ser qualquer algoritmo de ML — Regressão Logística, Árvore de Decisão, SVM, etc. Regressão Logística é inclusive uma escolha clássica e eficiente para o Blender.

---

---

## Seção 10 — Florestas Aleatórias (Random Forest)

### 10.1 Conceito e Definição

**O que é?**
A Floresta Aleatória (Random Forest) é um algoritmo de aprendizado supervisionado que cria uma **floresta** de várias Árvores de Decisão **independentes** para resolver problemas de classificação e regressão.

**Por que funciona?**
Baseia-se no princípio de Ensemble: ao agregar previsões de muitos modelos individuais (árvores), obtém-se um modelo global **muito mais robusto e com menor variância** do que qualquer árvore isolada.

**Analogia:** Imagine consultar 500 médicos especialistas (cada um com experiência em subconjuntos diferentes de pacientes) e tomar a decisão pela maioria — muito mais confiável do que consultar apenas um.

| Conceito | Descrição |
|---|---|
| Ensemble | Grupo de preditores treinados em subconjuntos diferentes |
| Árvore base | DecisionTree com profundidade irrestrita (pode crescer completa) |
| Aleatoriedade | Duas fontes: nas amostras (Bootstrap) e nos atributos (Feature Sampling) |
| Agregação | Moda (classificação) ou Média (regressão) |
| Paralelismo | Todas as árvores treinadas independentemente em paralelo |

---

### 10.2 As Duas Fontes de Aleatoriedade

O grande diferencial da Floresta Aleatória em relação ao Bagging simples é introduzir **duas camadas de aleatoriedade**:

#### Fonte 1 — Bootstrap (Aleatoriedade nas Amostras)

Para cada árvore, o algoritmo amostra o dataset original **com reposição**:

- Cada subconjunto tem o mesmo tamanho do dataset original (n amostras)
- ~**63,2%** das amostras aparecem ao menos uma vez por subconjunto
- ~**36,8%** ficam de fora → chamados **Out-of-Bag (OOB)**
- Os dados OOB servem como validação gratuita sem necessidade de holdout separado

```
Dataset original: [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]

Bootstrap 1:  [x8, x6, x2, x9, x5, x8, x1, x4, x8, x2] → OOB: {x3, x7, x10}
Bootstrap 2:  [x10, x1, x3, x5, x1, x7, x4, x2, x1, x8] → OOB: {x6, x9}
Bootstrap 3:  [x6, x5, x4, x1, x2, x4, x2, x6, x9, x2] → OOB: {x3, x7, x8, x10}
```

**Estimativa OOB:** cada amostra é classificada pelas árvores que **não** a usaram no treino. A média dessas predições é o **OOB Score** — equivalente a uma validação cruzada "grátis".

#### Fonte 2 — Feature Sampling (Aleatoriedade nos Atributos)

Em **cada nó** de cada árvore, o algoritmo **sorteia um subconjunto aleatório de atributos** e só considera esses para escolher a melhor divisão:

| Problema | Padrão do parâmetro `max_features` |
|---|---|
| Classificação | `sqrt(p)` — raiz quadrada do total de features |
| Regressão | `1.0` (todas) ou `sqrt(p)` |
| Extra-Trees | Limiar aleatório em vez de busca do melhor corte |

**Por que isso importa?** Se há uma feature muito dominante (ex: "renda" para prever crédito), em árvores normais ela apareceria na raiz de todas as árvores → todas ficariam correlacionadas → não haveria diversidade. Com `max_features=sqrt(p)`, às vezes essa feature não está disponível no nó, forçando a árvore a explorar outros padrões.

| Fonte de Aleatoriedade | O que varia entre árvores | Benefício |
|---|---|---|
| Bootstrap | Quais instâncias são usadas | Reduz covariância entre árvores |
| Feature Sampling | Quais atributos disponíveis em cada nó | Descorrelaciona estrutura das árvores |
| Combinação das duas | Amostras + atributos | Máxima diversidade → menor variância |

---

### 10.3 Arquitetura de Treinamento (4 Etapas)

```
Dataset Original
       │
       ▼
┌──────────────────────────────────────────────┐
│  1. SELEÇÃO — Bootstrap                       │
│  Para cada árvore k:                          │
│  • Amostrar n instâncias com reposição        │
│  • ~37% ficam OOB (validação gratuita)        │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  2. CRESCIMENTO — Feature Sampling em cada nó │
│  Para cada nó de cada árvore:                 │
│  • Sortear sqrt(p) features                   │
│  • Escolher melhor split dentre essas         │
│  • Crescer a árvore sem poda (profunda)       │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  3. PREVISÃO — Paralela e Independente        │
│  Cada árvore gera sua predição individual     │
│  sem comunicação com as outras                │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  4. AGREGAÇÃO — Decisão Final                 │
│  Classificação: Moda (votação majoritária)    │
│  Regressão:     Média aritmética das saídas   │
└──────────────────────────────────────────────┘
```

| Etapa | Ação | Parâmetro Scikit-Learn |
|---|---|---|
| Seleção | Bootstrap com reposição | `bootstrap=True` (padrão) |
| N.º de árvores | Quantas árvores na floresta | `n_estimators=100` |
| Crescimento | Profundidade máxima da árvore | `max_depth=None` (irrestrito) |
| Feature Sampling | Atributos sorteados por nó | `max_features='sqrt'` |
| Avaliação OOB | Score sem holdout | `oob_score=True` |
| Paralelismo | Usar todos os núcleos | `n_jobs=-1` |

---

### 10.4 Feature Importance — Importância dos Atributos

Um dos maiores diferenciais da Floresta Aleatória é fornecer **Feature Importance**: uma medida de qual atributo mais contribui para as predições.

**Como é calculada:**

$$\text{Importance}(j) = \frac{1}{K} \sum_{k=1}^{K} \sum_{\text{nó } t \in T_k \text{ que usa } j} \frac{n_t}{n} \cdot \Delta\text{impureza}(t)$$

| Símbolo | Significado |
|---|---|
| $K$ | Número total de árvores |
| $k$ | Índice de uma árvore específica |
| $t$ | Nó que usa o atributo $j$ para dividir |
| $n_t$ | Número de amostras que chegam ao nó $t$ |
| $n$ | Total de amostras no dataset |
| $\Delta\text{impureza}(t)$ | Redução de Gini (ou Entropia) no nó $t$ |

**Em palavras:** "quanto impureza (em média, ponderada pelo número de amostras) o atributo $j$ reduz ao longo de todas as árvores?"

**Exemplo — Dataset Iris (500 árvores):**

| Atributo | Importância |
|---|---|
| petal length (cm) | **0.441** ← mais importante |
| petal width (cm) | **0.423** ← segundo mais importante |
| sepal length (cm) | 0.112 |
| sepal width (cm) | 0.023 ← menos importante |

**Interpretação:** Pétalas (comprimento e largura) dominam a classificação das espécies de Iris; sépalas contribuem pouco.

**Uso prático no estilo do Prof. Túlio:**
- Selecionar as `k` features mais importantes → reduzir dimensionalidade
- Diagnosticar se uma variável está dominando excessivamente → correlação espúria
- Justificar quais variáveis incluir em um modelo mais simples (logística, por ex.)

```python
# Exemplo de leitura da importância
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rnd_clf.fit(iris["data"], iris["target"])

for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(f"{name}: {score:.4f}")
# sepal length (cm): 0.1125
# sepal width (cm):  0.0231
# petal length (cm): 0.4410   ← dominante
# petal width (cm):  0.4234
```

---

### 10.5 Extra-Trees (Árvores Extremamente Aleatórias)

As **Extra-Trees** (Extremely Randomized Trees) são uma variação da Floresta Aleatória que eleva ainda mais a aleatoriedade ao escolher **limiares aleatórios** para as divisões em vez de buscar o corte ótimo.

**Diferença fundamental:**

| Aspecto | Random Forest | Extra-Trees |
|---|---|---|
| Seleção de amostras | Bootstrap (com reposição) | Bootstrap ou todo o dataset |
| Seleção de features por nó | `sqrt(p)` features | `sqrt(p)` features (igual) |
| Escolha do limiar de corte | **Busca o MELHOR limiar** entre os sorteados | **Limiar ALEATÓRIO** para cada feature sorteada |
| Velocidade de treino | Moderada | **Muito mais rápida** |
| Variância | Baixa | **Ainda mais baixa** |
| Viés | Baixo | Ligeiramente maior |
| Risco de overfitting | Baixo | Muito baixo |

**Por que o limiar aleatório ajuda?**
- Random Forest busca o melhor `threshold` dentre os valores presentes → computacionalmente caro
- Extra-Trees sorteia o threshold aleatoriamente → elimina busca exaustiva → **treinamento muito mais rápido**
- A aleatoriedade adicional força as árvores a serem ainda mais diferentes entre si → **menor correlação → menor variância do ensemble**

**Trade-off:** Ligeiramente mais viés em troca de muito menos variância e muito mais velocidade. Em datasets grandes com muitas features, Extra-Trees frequentemente supera Random Forest na prática.

**Scikit-Learn:**
```python
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
```

| Classe | Uso |
|---|---|
| `ExtraTreesClassifier` | Classificação com limiares aleatórios |
| `ExtraTreesRegressor` | Regressão com limiares aleatórios |
| `RandomForestClassifier` | Classificação com limiares ótimos |
| `RandomForestRegressor` | Regressão com limiares ótimos |

---

### 10.6 Floresta Aleatória vs. Árvore de Decisão Única

| Característica | Árvore de Decisão | Floresta Aleatória |
|---|---|---|
| Número de modelos | 1 | 100–1000 árvores |
| Variância | **Alta** (instável) | **Baixa** (estável) |
| Viés | Baixo (profunda) | Baixo (idem) |
| Overfitting | **Sim** (sem poda) | Resistente |
| Interpretabilidade | Alta (visualizável) | **Baixa** (caixa preta) |
| Feature Importance | Sim (por árvore) | **Sim (média geral)** |
| Velocidade de treino | Rápida | Mais lenta (mas paralelizável) |
| Velocidade de predição | Rápida | Mais lenta (K árvores) |
| Dados ausentes | Sensível | Mais robusto |
| Escalabilidade | Limitada | Alta (n_jobs=-1) |

**Regra prática:** Se uma única árvore overfita → use Random Forest. Se Random Forest é lenta demais → use Extra-Trees.

---

### 10.7 Hiperparâmetros Principais e Seus Efeitos

| Hiperparâmetro | Padrão | Efeito ao Aumentar | Quando Ajustar |
|---|---|---|---|
| `n_estimators` | 100 | Mais estável, mais lento | Sempre aumentar até plateaus |
| `max_features` | `'sqrt'` | Mais features → menos aleatoriedade | Diminuir se overfitting |
| `max_depth` | `None` | Mais profundidade → mais variância | Limitar se overfitting |
| `min_samples_leaf` | 1 | Mais restrição → menos variância | Aumentar para datasets pequenos |
| `bootstrap` | `True` | `False` → pasting sem OOB | Raramente mudar |
| `oob_score` | `False` | `True` → validação grátis | Ligar para estimar generalização |
| `n_jobs` | 1 | `-1` usa todos os núcleos | Sempre `-1` em produção |
| `class_weight` | `None` | `'balanced'` corrige desbalanceamento | Dados desbalanceados |

---

### 10.8 Código Python Completo

**Cenário:** Diagnóstico de câncer de mama (Wisconsin Breast Cancer Dataset). Comparação entre Árvore única, Random Forest e Extra-Trees com análise de Feature Importance.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance

# ── 1. CARREGAR E DIVIDIR DADOS ──────────────────────────────────────────────
dados = load_breast_cancer()
X, y = dados.data, dados.target
nomes_features = dados.feature_names

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Treino: {X_treino.shape[0]} amostras | Teste: {X_teste.shape[0]} amostras")
print(f"Features: {X.shape[1]} | Classes: {dados.target_names}")

# ── 2. TREINAR OS 3 MODELOS ───────────────────────────────────────────────────
arvore = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(
    n_estimators=200,
    max_features='sqrt',
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
extra = ExtraTreesClassifier(
    n_estimators=200,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

for nome, modelo in [("Árvore", arvore), ("Random Forest", rf), ("Extra-Trees", extra)]:
    modelo.fit(X_treino, y_treino)
    y_pred = modelo.predict(X_teste)
    y_proba = modelo.predict_proba(X_teste)[:, 1]
    acc = accuracy_score(y_teste, y_pred)
    auc = roc_auc_score(y_teste, y_proba)
    print(f"\n{'='*45}")
    print(f"Modelo: {nome}")
    print(f"  Acurácia:  {acc:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    if hasattr(modelo, 'oob_score_'):
        print(f"  OOB Score: {modelo.oob_score_:.4f}")

# ── 3. FEATURE IMPORTANCE — RANDOM FOREST ────────────────────────────────────
importancias = rf.feature_importances_
indices_ordenados = np.argsort(importancias)[::-1]

print("\nTop 10 Features mais importantes (Random Forest):")
print(f"{'Rank':<5} {'Feature':<35} {'Importância':>12}")
print("-" * 55)
for i in range(10):
    idx = indices_ordenados[i]
    print(f"{i+1:<5} {nomes_features[idx]:<35} {importancias[idx]:>12.4f}")

# ── 4. VALIDAÇÃO CRUZADA ──────────────────────────────────────────────────────
print("\nValidação Cruzada (5-fold, ROC-AUC):")
for nome, modelo in [("Árvore", arvore), ("RF", rf), ("Extra-Trees", extra)]:
    scores = cross_val_score(modelo, X, y, cv=5, scoring='roc_auc', n_jobs=-1)
    print(f"  {nome:<15} AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# ── 5. EFEITO DO N_ESTIMATORS ─────────────────────────────────────────────────
# Monitorar como o OOB score evolui com o número de árvores
oob_scores = []
n_range = range(10, 201, 10)
for n in n_range:
    modelo_temp = RandomForestClassifier(
        n_estimators=n, oob_score=True, n_jobs=-1, random_state=42
    )
    modelo_temp.fit(X_treino, y_treino)
    oob_scores.append(modelo_temp.oob_score_)

# Encontrar platô
platô_idx = next(i for i in range(1, len(oob_scores))
                 if abs(oob_scores[i] - oob_scores[i-1]) < 0.001)
print(f"\nOOB Score estabiliza em ~{list(n_range)[platô_idx]} árvores "
      f"(score={oob_scores[platô_idx]:.4f})")

# ── 6. DIAGNÓSTICO: DETECTAR FEATURE DOMINANTE ───────────────────────────────
top_feature = nomes_features[indices_ordenados[0]]
top_imp = importancias[indices_ordenados[0]]
if top_imp > 0.3:
    print(f"\nAVISO: '{top_feature}' domina com {top_imp:.1%} de importância.")
    print("Considere investigar correlação espúria ou data leakage.")
else:
    print(f"\nDistribuição de importância equilibrada. Top feature: "
          f"'{top_feature}' ({top_imp:.1%})")
```

**Saída esperada (aproximada):**
```
Treino: 455 amostras | Teste: 114 amostras
Features: 30 | Classes: ['malignant' 'benign']

=============================================
Modelo: Árvore
  Acurácia:  0.9123
  ROC-AUC:   0.9104

=============================================
Modelo: Random Forest
  Acurácia:  0.9649
  ROC-AUC:   0.9941
  OOB Score: 0.9626

=============================================
Modelo: Extra-Trees
  Acurácia:  0.9561
  ROC-AUC:   0.9918

Top 10 Features mais importantes (Random Forest):
Rank  Feature                             Importância
-------------------------------------------------------
1     worst concave points                     0.1423
2     worst perimeter                          0.1187
3     mean concave points                      0.1054
...

Validação Cruzada (5-fold, ROC-AUC):
  Árvore          AUC = 0.9231 ± 0.0187
  RF              AUC = 0.9923 ± 0.0043
  Extra-Trees     AUC = 0.9912 ± 0.0051
```

---

### 10.9 Questões no Estilo da Prova (Prof. Túlio Ribeiro)

---

**Questão 1**

Um cientista de dados está construindo uma Floresta Aleatória com 200 árvores e `max_features='sqrt'` para um dataset com 100 features. Após o treino, ele verifica que uma única feature aparece com 60% da importância total. Qual é a interpretação mais correta?

a) Isso confirma que o modelo está bem calibrado, pois identificou a variável mais relevante.
b) A feature dominante indica que as outras 99 features são irrelevantes e devem ser removidas.
c) Pode haver data leakage ou correlação espúria; a feature pode conter informação futura ou derivada do alvo.
d) O parâmetro `max_features` deveria ser aumentado para `'log2'` para corrigir esse comportamento.

**Resposta correta: C**

- **A — ERRADO:** Importância muito concentrada não é sinal de bom calibramento; pode indicar vazamento de dados ou proxy do alvo no treino.
- **B — ERRADO:** As outras features podem ser importantes entre si. Remover 99 features com base nisso seria precipitado sem investigação.
- **C — CORRETO:** Uma dominância de 60% é suspeita em datasets com 100 features. Causas comuns: a feature é um proxy do alvo (ex: "total de pagamentos" prediz "inadimplência"), está computada com dados futuros, ou há alta correlação não-causal. Deve-se investigar antes de confiar no modelo.
- **D — ERRADO:** `'log2'` sortearia ainda menos features por nó (log₂(100) ≈ 6,6 vs. √100 = 10), aumentando a aleatoriedade mas não resolvendo o problema de dominância de uma feature específica.

---

**Questão 2**

Numa Floresta Aleatória treinada com `oob_score=True`, o OOB Score foi 0.91 e a acurácia no conjunto de validação (holdout 20%) foi 0.93. Qual afirmação é verdadeira?

a) O OOB Score é sempre igual à acurácia de validação; os valores diferentes indicam bug no código.
b) O OOB Score usa amostras que cada árvore não viu, sendo uma estimativa válida de generalização — a pequena diferença é normal.
c) O OOB Score de 0.91 indica underfitting grave; é necessário aumentar `max_depth`.
d) Como o OOB Score é menor que a acurácia de validação, o modelo está overfittando no conjunto de validação.

**Resposta correta: B**

- **A — ERRADO:** OOB Score e acurácia de validação usam amostras diferentes e metodologias diferentes; diferenças pequenas são esperadas e normais.
- **B — CORRETO:** O OOB Score funciona como uma validação cruzada embutida, usando os ~36,8% de amostras que cada árvore não usou no treino. É uma estimativa pessimista ligeiramente conservadora, por isso 0.91 vs 0.93 é uma diferença completamente normal.
- **C — ERRADO:** 0.91 de OOB Score é um resultado excelente. Underfitting seria indicado por baixa acurácia tanto no treino quanto na validação, não por uma ligeira diferença entre OOB e holdout.
- **D — ERRADO:** O modelo overfitar no conjunto de validação seria impossível por definição (validação não é usada no treino). A diferença reflete variabilidade amostral entre os dois conjuntos.

---

**Questão 3**

Um analista compara Random Forest e Extra-Trees no mesmo dataset com 1 milhão de amostras e 500 features. Extra-Trees treinou 10x mais rápido. Por qual razão técnica isso ocorre?

a) Extra-Trees usa menos árvores por padrão, reduzindo o tempo de treino proporcionalmente.
b) Extra-Trees aplica poda nas árvores durante o treinamento, o que reduz a complexidade.
c) Extra-Trees não realiza bootstrap, treinando cada árvore em todo o dataset sem reamostragem.
d) Extra-Trees escolhe limiares de corte aleatoriamente em vez de buscar o melhor limiar, eliminando a busca exaustiva em cada nó.

**Resposta correta: D**

- **A — ERRADO:** O padrão de `n_estimators` é 100 para ambos. A diferença de velocidade não está no número de árvores, mas na computação por nó.
- **B — ERRADO:** Extra-Trees não poda as árvores; na verdade, as árvores costumam crescer tão profundas quanto Random Forest. A velocidade vem da escolha do limiar, não da profundidade.
- **C — ERRADO:** Extra-Trees por padrão ainda usa bootstrap (`bootstrap=True`). Mesmo quando usa o dataset completo, a principal diferença de velocidade é o limiar aleatório, não a amostragem.
- **D — CORRETO:** O gargalo computacional no treino de uma árvore de decisão é a busca do melhor threshold. Com 500 features e n valores únicos por feature, cada nó exige ordenação e avaliação de n×500 splits. Extra-Trees sorteia um threshold aleatório por feature → apenas `sqrt(500) ≈ 22` avaliações por nó sem ordenação → ganho de velocidade massivo.

---

**Questão 4**

Em um problema de classificação com dados desbalanceados (90% classe 0, 10% classe 1), um modelo Random Forest atingiu 90% de acurácia mas identificou apenas 20% dos casos positivos (baixo Recall). Qual das intervenções abaixo é mais apropriada?

a) Aumentar `n_estimators` de 100 para 1000, pois mais árvores sempre melhoram o Recall.
b) Usar `class_weight='balanced'` para que o algoritmo penalize mais os erros na classe minoritária.
c) Reduzir `max_features` para `'log2'` pois menos features por nó forçam o modelo a aprender a minoria.
d) Desativar o bootstrap (`bootstrap=False`) para que todas as árvores vejam todos os casos positivos.

**Resposta correta: B**

- **A — ERRADO:** Aumentar `n_estimators` melhora a estabilidade e reduz variância, mas não corrige o desequilíbrio de classes. Com 90% de dados negativos, mais árvores apenas confirmam o viés existente.
- **B — CORRETO:** `class_weight='balanced'` faz o scikit-learn calcular pesos inversamente proporcionais à frequência: classe 0 peso ~0.56, classe 1 peso ~5.0. Isso penaliza muito mais erros na classe minoritária, forçando as árvores a aprender padrões dos casos positivos → Recall aumenta.
- **C — ERRADO:** `max_features='log2'` aumenta aleatoriedade nas features, mas não endereça o problema de desequilíbrio de classes. As instâncias negativas ainda dominam nas amostras bootstrap.
- **D — ERRADO:** Desativar bootstrap faz cada árvore treinar no dataset completo (pasting), mas com 90% de negativos, isso ainda não resolve o desequilíbrio. Pior: perde-se a estimativa OOB e reduz-se a diversidade entre árvores.

---

---

## Seção 11 — KNN (K-Nearest Neighbors)

### 11.1 Métodos Baseados em Distância — A Premissa Fundamental

O KNN pertence à família dos **Métodos Baseados em Distância**, onde os dados são interpretados de forma **estritamente espacial**: cada instância é um ponto em um espaço n-dimensional, e a premissa central é:

> **"Similaridade equivale a proximidade espacial"** — se dois pontos estão geometricamente próximos, tendem a pertencer à mesma classe ou ter o mesmo comportamento.

O KNN é o representante mais elementar dessa família, baseando **100% de suas decisões** no cálculo matemático de distância.

**Frase do professor:** *"Diga-me com quem andas e te direi quem és."*

| Conceito | Definição |
|---|---|
| Espaço de features | Cada instância vira um ponto em R^n (n = número de features) |
| Proximidade | Pontos próximos → classe provável similar |
| Premissa | Vizinhança local é mais informativa que padrões globais |
| Paradigma | Não cria modelo explícito; decide na hora da predição |

---

### 11.2 Conceito do KNN — Lazy Learner e Não-Paramétrico

O KNN tem duas propriedades características que o diferenciam de todos os outros algoritmos do curso:

#### Lazy Learner (Aprendizado Preguiçoso)

O KNN praticamente **não tem fase de treinamento**. Durante o treino, ele simplesmente **memoriza** (armazena) todo o conjunto de dados na memória. Todo o trabalho computacional pesado acontece apenas na hora da **predição** — quando um novo dado chega.

```
TREINO (KNN):    Apenas armazena os dados → custo ≈ O(1)
PREDIÇÃO (KNN):  Calcula distância para TODOS os pontos → custo = O(n × d)
                 onde n = amostras no treino, d = dimensionalidade
```

**Contraste com outros algoritmos:**

| Algoritmo | Fase de Treino | Fase de Predição |
|---|---|---|
| Regressão Linear | Pesada (ajusta parâmetros) | Instantânea (y = wx + b) |
| Random Forest | Pesada (treina 100+ árvores) | Moderada (percorre K árvores) |
| SVM | Muito pesada (otimização quadrática) | Rápida (apenas vetores de suporte) |
| **KNN** | **Instantânea (só memoriza)** | **Lenta (distância para todos)** |

#### Não-Paramétrico

O KNN **não faz nenhuma suposição matemática prévia** sobre a distribuição dos dados. Ele não assume que os dados são linearmente separáveis (como Regressão Logística), nem que seguem distribuição normal, nem que existe uma fronteira de decisão de forma específica.

Isso o torna muito **flexível**: consegue capturar fronteiras de decisão arbitrariamente complexas e não-lineares, desde que haja dados suficientes na vizinhança.

---

### 11.3 Mecânica do Algoritmo — 4 Passos

Para classificar um novo dado (não rotulado), o KNN executa 4 passos lógicos:

```
Novo dado (x_q) chega
        │
        ▼
┌────────────────────────────────────────────────────┐
│ 1. CÁLCULO                                          │
│ Mede a distância de x_q para TODOS os n pontos     │
│ do dataset de treino                               │
│ d(x_q, x_i) para i = 1, 2, ..., n                 │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│ 2. ORDENAÇÃO                                        │
│ Classifica as n distâncias em ordem crescente       │
│ (do mais próximo ao mais distante)                  │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│ 3. SELEÇÃO                                          │
│ Recorta os K primeiros pontos da lista ordenada     │
│ → esses são os K-vizinhos mais próximos             │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│ 4. DECISÃO                                          │
│ Classificação: Votação majoritária (moda das classes│
│                dos K vizinhos)                      │
│ Regressão:     Média aritmética (ou mediana) do     │
│                valor alvo dos K vizinhos            │
└────────────────────────────────────────────────────┘
```

**Exemplo numérico (do slide do professor):**

Novo ponto chega numa região com K = 11 vizinhos:
- 7 vizinhos são da **classe Vermelha**
- 3 vizinhos são da **classe Laranja**
- 1 vizinho é da **classe Verde**

→ Votação: Vermelho=7, Laranja=3, Verde=1 → **Predição: VERMELHO**

| K | Efeito na Fronteira de Decisão | Risco |
|---|---|---|
| K = 1 | Ultra-complexa, recortada | Overfitting (alta variância) |
| K = 3 | Moderadamente complexa | Baixo |
| K = 11 | Mais suave | Equilíbrio razoável |
| K = 100 | Muito suave, quase uma linha reta | Underfitting (alto viés) |

---

### 11.4 Métricas de Distância

A escolha da métrica de distância é crítica para o KNN, pois **define quem são os "vizinhos"**.

#### Fórmulas Dissecadas

**1. Distância Euclidiana** (padrão, `p=2` em Scikit-Learn)

$$d_E(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_{j=1}^{n} (x_j - z_j)^2}$$

| Símbolo | Significado |
|---|---|
| $\mathbf{x}, \mathbf{z}$ | Dois pontos no espaço de features |
| $j$ | Índice da feature (dimensão) |
| $n$ | Número de features (dimensões) |
| $x_j - z_j$ | Diferença na feature $j$ |
| $\sqrt{(\cdot)}$ | Raiz quadrada → distância em linha reta |

**Quando usar:** Dados numéricos contínuos sem outliers extremos. É o Teorema de Pitágoras generalizado para n dimensões.

**2. Distância Manhattan** (City Block, `p=1`)

$$d_M(\mathbf{x}, \mathbf{z}) = \sum_{j=1}^{n} |x_j - z_j|$$

**Quando usar:** Dados com muitos outliers (menos sensível a grandes diferenças em uma única dimensão), dados categóricos codificados, ou quando o movimento diagonal não faz sentido no domínio.

**3. Distância de Minkowski** (generalização, `metric='minkowski', p=p`)

$$d_p(\mathbf{x}, \mathbf{z}) = \left(\sum_{j=1}^{n} |x_j - z_j|^p\right)^{1/p}$$

**Casos especiais:**
- $p = 1$ → Manhattan
- $p = 2$ → Euclidiana
- $p \to \infty$ → Chebyshev (máxima diferença entre dimensões)

#### Tabela Comparativa das Métricas

| Métrica | Fórmula | Geometria | Ideal Para |
|---|---|---|---|
| Euclidiana | $\sqrt{\sum (x_j-z_j)^2}$ | Linha reta | Numérico contínuo, sem outliers |
| Manhattan | $\sum |x_j - z_j|$ | Quarteirões de cidade | Outliers, dados esparsos |
| Minkowski | $(\sum|x_j-z_j|^p)^{1/p}$ | Generalização | Ajuste fino via parâmetro p |
| Chebyshev | $\max_j |x_j - z_j|$ | Movimento do rei no xadrez | Jogos, otimização de estoque |
| Cosseno | $1 - \frac{\mathbf{x} \cdot \mathbf{z}}{||\mathbf{x}||||\mathbf{z}||}$ | Ângulo entre vetores | NLP, texto, dados de alta dimensão |
| Hamming | Posições diferentes / total | Comparação bit a bit | Strings, dados binários/categóricos |
| Haversine | Fórmula esférica | Arco na superfície da Terra | Dados geográficos (lat/lon) |

---

### 11.5 O Parâmetro K — Viés-Variância no KNN

O valor de K é o **hiperparâmetro mais crítico** do KNN e controla diretamente o trade-off viés-variância.

```
K PEQUENO                              K GRANDE
(ex: K=1)                              (ex: K=100)

Fronteira de decisão:                  Fronteira de decisão:
Ultra-recortada, complexa              Muito suave, quase linear
↓                                      ↓
Alta VARIÂNCIA                         Alto VIÉS
→ Overfitting                          → Underfitting
→ Sensível a ruído                     → Padrões locais ignorados
→ Perfeito no treino                   → Ruim no treino e no teste
→ Ruim no teste                        → Prevê sempre a classe majoritária
```

**Analogia:** K=1 é como perguntar a opinião de apenas 1 pessoa (pode ser um outlier); K=n é como perguntar a todos e sempre escolher a opinião da maioria geral.

**Como encontrar o K ideal:**

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param_grid = {'n_neighbors': range(1, 31, 2)}  # K ímpar evita empates
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_treino, y_treino)
print(f"Melhor K: {grid.best_params_}")
```

**Regras práticas:**
- Sempre testar K ímpar (em classificação binária evita empates)
- Ponto de partida comum: $K = \sqrt{n}$ onde n = número de amostras
- Usar validação cruzada para escolher K definitivo

---

### 11.6 KNN Ponderado (Weighted KNN)

No KNN clássico, todos os K vizinhos têm **igual peso** na votação, independentemente de quão perto ou longe estejam. O KNN Ponderado corrige isso:

$$\hat{y} = \underset{c}{\arg\max} \sum_{i \in \mathcal{N}_K} w_i \cdot \mathbb{1}[y_i = c]$$

Onde o peso de cada vizinho é inversamente proporcional à distância:

$$w_i = \frac{1}{d(x_q, x_i)}$$

| Aspecto | KNN Clássico (`weights='uniform'`) | KNN Ponderado (`weights='distance'`) |
|---|---|---|
| Peso de cada vizinho | Igual (1/K) para todos | Inversamente proporcional à distância |
| Vizinho distante | Mesmo peso que o próximo | Menor influência |
| Empates | Comum | Raramente ocorre |
| Dados esparsos | Problemático | Mais robusto |
| Scikit-Learn | `weights='uniform'` | `weights='distance'` |

**Exemplo:** Se K=3 e as distâncias são [0.1, 0.5, 2.0] com classes [A, A, B]:
- Clássico: 2 votos A, 1 voto B → **A**
- Ponderado: w=[10, 2, 0.5] → A=12, B=0.5 → **A** (com muito mais confiança)

---

### 11.7 Radius Neighbors — KNN por Densidade

O KNN tradicional tem um problema: em regiões esparsas do espaço de features, o algoritmo é forçado a buscar K vizinhos mesmo que todos sejam excessivamente distantes e não representem o padrão local.

**Solução — Radius Neighbors Classifier:** substitui o parâmetro K por um **raio de distância limite** r:

```
KNN Tradicional:                    Radius Neighbors:
"Me dê sempre K vizinhos"           "Me dê todos os vizinhos dentro do raio r"
→ Forçado a pegar vizinhos          → Cluster denso → muitos vizinhos
  distantes em áreas esparsas         Área esparsa → poucos (ou 0) vizinhos
```

| Região do Espaço | KNN (K fixo) | Radius Neighbors (r fixo) |
|---|---|---|
| Cluster denso | Apenas K dos muitos vizinhos próximos | Todos os vizinhos legítimos |
| Área esparsa | Forçado a pegar vizinhos distantes | Poucos ou nenhum vizinho |
| Comportamento | Uniforme (sempre K) | Adaptativo à densidade local |

```python
from sklearn.neighbors import RadiusNeighborsClassifier
modelo_raio = RadiusNeighborsClassifier(radius=1.5, weights='distance')
```

---

### 11.8 Cuidados Obrigatórios na Modelagem com KNN

Por basear **100% das decisões em cálculos de distância**, o KNN exige dois pré-processamentos inegociáveis:

#### Cuidado 1 — Escala dos Dados (OBRIGATÓRIO)

**Problema:** Features com magnitudes diferentes dominam o cálculo.

**Exemplo real:** Para prever crédito com as features `renda` (R$ 5.000–50.000) e `filhos` (0–5):

$$d = \sqrt{(50000 - 5000)^2 + (3 - 2)^2} \approx 45000$$

A feature `renda` domina completamente. O número de filhos é irrelevante para o KNN, mesmo que seja crucial para o problema.

**Solução:** Normalização **antes** de aplicar o KNN.

| Scaler | Fórmula | Quando usar |
|---|---|---|
| `StandardScaler` | $(x - \mu) / \sigma$ | Distribuição aproximadamente normal; outliers moderados |
| `MinMaxScaler` | $(x - x_{min}) / (x_{max} - x_{min})$ | Limites conhecidos; sem outliers extremos |
| `RobustScaler` | $(x - Q_2) / (Q_3 - Q_1)$ | Muitos outliers; mediana como centro |

**Regra:** Sempre ajuste o scaler **apenas no treino** (`fit` no treino, `transform` no treino e teste).

#### Cuidado 2 — Maldição da Dimensionalidade

**Problema:** Com muitas features (alta dimensão), o espaço se torna tão vasto que a distância entre quase todos os pontos se torna similar — a noção de "vizinhança" perde significado.

**Intuitivamente:** Em 2D, "próximo" é claro. Em 1000D, dois pontos aleatórios têm distâncias quase idênticas entre si → não há vizinhos "mais próximos" de verdade.

**Consequência:** KNN degrada severamente com muitas features.

**Soluções:**
1. **PCA** (Seção 3) → reduzir para k componentes principais antes do KNN
2. **Seleção de features** → Feature Importance da Random Forest + KNN
3. **Limitar features** → manter apenas as mais relevantes ao problema

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # manter 95% da variância
    ('knn', KNeighborsClassifier(n_neighbors=5))
])
```

---

### 11.9 KNN para Detecção de Anomalias (Não-Supervisionado)

Embora o KNN seja tradicionalmente supervisionado, sua arquitetura baseada em distâncias é usada para **detecção de anomalias** sem rótulos:

**Lógica de isolamento:**
1. Para cada ponto, calcular a distância até seus K vizinhos mais próximos
2. Calcular a distância média dos K vizinhos: $\bar{d}_K(x_i)$
3. Se $\bar{d}_K(x_i) \gg$ média geral do dataset → ponto **anômalo** (isolado)

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

knn = NearestNeighbors(n_neighbors=5)
knn.fit(X)
distancias, _ = knn.kneighbors(X)
score_anomalia = distancias.mean(axis=1)  # maior score = mais isolado

# Threshold: pontos com score > média + 2*desvio são anomalias
threshold = score_anomalia.mean() + 2 * score_anomalia.std()
anomalias = np.where(score_anomalia > threshold)[0]
```

**Comparação:** O KNN para anomalias é mais interpretável que Isolation Forest, mas mais lento em datasets grandes.

---

### 11.10 Prós e Contras Completos

| Dimensão | Vantagem | Desvantagem |
|---|---|---|
| **Treinamento** | Instantâneo — apenas memoriza | — |
| **Predição** | — | Lento — O(n × d) por predição |
| **Memória** | — | Exige dataset completo na RAM |
| **Interpretabilidade** | Alta — lógica transparente | — |
| **Flexibilidade** | Não-paramétrico, fronteiras complexas | — |
| **Escalabilidade** | — | Não escala bem para n > 100k |
| **Dimensionalidade** | — | Maldição da dimensionalidade |
| **Novas amostras** | Incorporadas imediatamente sem retreino | — |
| **Normalização** | — | Obrigatória (dados brutos inviáveis) |
| **Hiperparâmetros** | Poucos (K, métrica, pesos) | K sensível — errar K → overfitting/underfitting |

**Quando usar KNN:**
- Dataset pequeno a médio (< 50k amostras)
- Poucas features (< 50, idealmente < 20 após PCA)
- Fronteiras de decisão muito irregulares
- Quando precisar de um baseline rápido de implementar
- Prototipagem / comparação com modelos mais complexos

**Quando NÃO usar KNN:**
- Dataset muito grande (predição em tempo real impossível)
- Alta dimensionalidade sem redução prévia
- Recursos de memória limitados (precisa do dataset inteiro)
- Features sem normalização disponível

---

### 11.11 Classes Scikit-Learn e Parâmetros

| Classe | Uso |
|---|---|
| `KNeighborsClassifier` | Classificação com K fixo |
| `KNeighborsRegressor` | Regressão com K fixo |
| `RadiusNeighborsClassifier` | Classificação com raio fixo |
| `RadiusNeighborsRegressor` | Regressão com raio fixo |
| `NearestNeighbors` | Busca de vizinhos (sem predição — usado em anomalias, recomendação) |

| Parâmetro | Padrão | Descrição |
|---|---|---|
| `n_neighbors` | 5 | Valor de K |
| `weights` | `'uniform'` | `'uniform'` (igual) ou `'distance'` (ponderado) |
| `metric` | `'minkowski'` | Métrica de distância |
| `p` | 2 | Parâmetro de Minkowski (2 = Euclidiana) |
| `algorithm` | `'auto'` | Estrutura de busca: `'ball_tree'`, `'kd_tree'`, `'brute'` |
| `n_jobs` | 1 | `-1` usa todos os núcleos |
| `leaf_size` | 30 | Tamanho da folha para BallTree/KDTree |

---

### 11.12 Código Python Completo

**Cenário:** Sistema de recomendação de tipo de tratamento médico com base em histórico de pacientes. Comparação de K=1 vs K=5 vs K=optimal, uso de scaler, GridSearch e pipeline completo.

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ── 1. CARREGAR DADOS ────────────────────────────────────────────────────────
dados = load_wine()
X, y = dados.data, dados.target
print(f"Dataset Wine: {X.shape[0]} amostras, {X.shape[1]} features, {len(np.unique(y))} classes")

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. SEM NORMALIZAÇÃO (demonstrar o problema) ───────────────────────────────
knn_sem_norm = KNeighborsClassifier(n_neighbors=5)
knn_sem_norm.fit(X_treino, y_treino)
acc_sem = accuracy_score(y_teste, knn_sem_norm.predict(X_teste))
print(f"\nSem normalização: Acurácia = {acc_sem:.4f}")

# ── 3. COM NORMALIZAÇÃO ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_treino_norm = scaler.fit_transform(X_treino)  # fit+transform no treino
X_teste_norm  = scaler.transform(X_teste)       # apenas transform no teste

knn_com_norm = KNeighborsClassifier(n_neighbors=5)
knn_com_norm.fit(X_treino_norm, y_treino)
acc_com = accuracy_score(y_teste, knn_com_norm.predict(X_teste_norm))
print(f"Com normalização: Acurácia = {acc_com:.4f}")
print(f"Ganho com normalização: {(acc_com - acc_sem)*100:+.1f}%")

# ── 4. EFEITO DO K — OVERFITTING vs UNDERFITTING ─────────────────────────────
print("\nEfeito do K (com normalização, 5-fold CV):")
print(f"{'K':<6} {'Treino':>10} {'Validação':>12} {'Diagnóstico':>15}")
print("-" * 50)
for k in [1, 3, 5, 9, 15, 25, 51]:
    modelo = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    # Acurácia no treino
    modelo.fit(X_treino_norm, y_treino)
    acc_tr = accuracy_score(y_treino, modelo.predict(X_treino_norm))
    # Acurácia em CV
    scores_cv = cross_val_score(modelo, X_treino_norm, y_treino, cv=5)
    acc_cv = scores_cv.mean()
    # Diagnóstico
    gap = acc_tr - acc_cv
    diag = "Overfitting" if gap > 0.1 else ("Underfitting" if acc_cv < 0.7 else "OK")
    print(f"K={k:<4} {acc_tr:>10.4f} {acc_cv:>12.4f} {diag:>15}")

# ── 5. GRIDSEARCH — ENCONTRAR K E PESOS ÓTIMOS ───────────────────────────────
param_grid = {
    'n_neighbors': list(range(1, 21, 2)),  # K ímpares de 1 a 19
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_treino_norm, y_treino)
print(f"\nGridSearch — Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor score CV: {grid_search.best_score_:.4f}")

# Avaliar no teste
melhor_knn = grid_search.best_estimator_
y_pred = melhor_knn.predict(X_teste_norm)
print(f"Acurácia no teste (melhor KNN): {accuracy_score(y_teste, y_pred):.4f}")
print("\nRelatório completo:")
print(classification_report(y_teste, y_pred, target_names=dados.target_names))

# ── 6. PIPELINE COMPLETO (prática profissional) ───────────────────────────────
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),       # manter 95% da variância
    ('knn', KNeighborsClassifier(
        n_neighbors=grid_search.best_params_['n_neighbors'],
        weights=grid_search.best_params_['weights'],
        metric=grid_search.best_params_['metric']
    ))
])
pipeline.fit(X_treino, y_treino)
acc_pipeline = accuracy_score(y_teste, pipeline.predict(X_teste))
n_componentes = pipeline.named_steps['pca'].n_components_
print(f"\nPipeline (Scaler + PCA + KNN):")
print(f"  Features originais: {X.shape[1]} → PCA reteve: {n_componentes}")
print(f"  Acurácia pipeline: {acc_pipeline:.4f}")

# ── 7. DETECÇÃO DE ANOMALIAS ─────────────────────────────────────────────────
from sklearn.neighbors import NearestNeighbors

knn_anomalia = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn_anomalia.fit(X_treino_norm)
distancias, _ = knn_anomalia.kneighbors(X_treino_norm)
score_anomalia = distancias.mean(axis=1)

threshold = score_anomalia.mean() + 2 * score_anomalia.std()
idx_anomalias = np.where(score_anomalia > threshold)[0]
print(f"\nDetecção de anomalias (KNN não-supervisionado):")
print(f"  Threshold: {threshold:.4f}")
print(f"  Anomalias detectadas: {len(idx_anomalias)} de {len(X_treino_norm)} pontos")
```

**Saída esperada (aproximada):**
```
Dataset Wine: 178 amostras, 13 features, 3 classes

Sem normalização: Acurácia = 0.7222
Com normalização: Acurácia = 0.9444
Ganho com normalização: +22.2%

Efeito do K (com normalização, 5-fold CV):
K      Treino    Validação     Diagnóstico
--------------------------------------------------
K=1      1.0000       0.9014     Overfitting
K=3      0.9577       0.9296              OK
K=5      0.9296       0.9296              OK
K=9      0.9085       0.9155              OK
K=15     0.8944       0.9014              OK
K=25     0.8662       0.8803              OK
K=51     0.7817       0.7746     Underfitting

GridSearch — Melhores parâmetros: {'metric': 'euclidean', 'n_neighbors': 7, 'weights': 'distance'}
Melhor score CV: 0.9437
Acurácia no teste (melhor KNN): 0.9722
```

---

### 11.13 Questões no Estilo da Prova (Prof. Túlio Ribeiro)

---

**Questão 1**

Um cientista de dados aplica KNN com K=5 em um dataset de preços de imóveis com as features: `área` (50–500 m²), `quartos` (1–5) e `distância_centro` (0–50 km). Sem normalização, o modelo tem acurácia de 62%. Após normalização com StandardScaler, a acurácia sobe para 87%. Qual é a explicação técnica correta?

a) A normalização aumenta o K efetivo, consultando mais vizinhos e reduzindo o viés.
b) Sem normalização, a feature `área` domina o cálculo de distância por ter magnitude muito maior, tornando `quartos` e `distância_centro` virtualmente invisíveis ao modelo.
c) O StandardScaler remove outliers do dataset, melhorando a qualidade das amostras usadas no treino.
d) A normalização transforma o KNN em um modelo paramétrico, permitindo que ele aprenda os parâmetros ótimos.

**Resposta correta: B**

- **A — ERRADO:** A normalização não altera o parâmetro K. O número de vizinhos consultados permanece 5.
- **B — CORRETO:** Sem normalização, calcular $\sqrt{(400-50)^2 + (3-2)^2 + (10-5)^2} \approx 350$. A contribuição de `quartos` (diferença máxima ~4) e `distância_centro` (diferença máxima ~50) é negligenciável frente à `área` (diferença até 450). O KNN efetivamente usa apenas a `área` para decidir a vizinhança. Com StandardScaler, todas as features têm média 0 e desvio 1 → contribuições equilibradas.
- **C — ERRADO:** StandardScaler não remove outliers. Ele transforma os dados para média zero e desvio padrão 1, mas pontos extremos continuam presentes (apenas reescalados).
- **D — ERRADO:** A normalização não muda a natureza do algoritmo. KNN permanece não-paramétrico após a normalização — a decisão ainda é baseada em distância, não em parâmetros aprendidos.

---

**Questão 2**

Um KNN com K=1 atinge 99% de acurácia no treino mas apenas 61% no teste. Um KNN com K=25 atinge 76% no treino e 74% no teste. Qual afirmação descreve corretamente essa situação?

a) K=1 deve ser preferido por ter maior acurácia absoluta no treino; o teste ruim indica que o conjunto de teste é de baixa qualidade.
b) K=1 sofre de overfitting (alta variância): memoriza os dados de treino mas não generaliza. K=25 pode estar no limite do underfitting mas generaliza razoavelmente.
c) K=25 é sempre melhor que K=1 porque algoritmos com menor variância são matematicamente superiores.
d) Ambos os modelos estão inadequados; a solução é reduzir K para próximo de zero para minimizar o viés.

**Resposta correta: B**

- **A — ERRADO:** 99% no treino e 61% no teste é o sinal clássico de overfitting. O modelo memorizou os dados de treino (K=1 → cada ponto é seu próprio vizinho → acurácia 100% no treino com 0 erros). A culpa não é do conjunto de teste.
- **B — CORRETO:** Com K=1, o KNN cria uma fronteira de decisão ultra-complexa que se adapta perfeitamente a cada ponto de treino, incluindo ruído. No teste, qualquer pequena variação gera erros. K=25 tem fronteiras mais suaves: gap treino-teste de apenas 2%, indicando boa generalização (74% pode ser aceitável dependendo da tarefa).
- **C — ERRADO:** Nem sempre menor variância é melhor — o trade-off viés-variância exige equilíbrio. K=25 pode ter viés alto demais para alguns problemas. "Sempre melhor" é incorreto.
- **D — ERRADO:** K próximo de zero não existe (K mínimo é 1). Além disso, K=1 já demonstrou overfitting severo — diminuir ainda mais K é a direção oposta da solução.

---

**Questão 3**

Em um dataset com 50.000 amostras e 200 features, um cientista avalia KNN e obtém predições extremamente lentas em produção e baixa acurácia. Qual intervenção resolve AMBOS os problemas simultaneamente?

a) Aumentar K para 1000 para consultar mais vizinhos e melhorar a acurácia.
b) Trocar `weights='uniform'` por `weights='distance'` para acelerar o cálculo de distâncias.
c) Aplicar PCA antes do KNN para reduzir a dimensionalidade, aliviando a Maldição da Dimensionalidade (melhora acurácia) e reduzindo o custo computacional (melhora velocidade).
d) Usar `algorithm='brute'` em vez de `algorithm='auto'` para garantir cálculo exato da distância.

**Resposta correta: C**

- **A — ERRADO:** Aumentar K de 5 para 1000 aumenta ainda mais o trabalho de ordenação e agregação, tornando a predição ainda mais lenta. E K muito alto tende a underfitting, não melhora a acurácia em datasets complexos.
- **B — ERRADO:** `weights='distance'` muda apenas como os votos são ponderados, não o número de distâncias calculadas. O custo computacional de calcular distâncias contra 50.000 pontos em 200 dimensões permanece o mesmo.
- **C — CORRETO:** PCA resolve ambos os problemas: (1) **Velocidade** — reduzir de 200 para ~20 componentes reduz o custo de cada distância em 10×; (2) **Acurácia** — com 200 features, a Maldição da Dimensionalidade faz com que todas as distâncias sejam semelhantes, invalidando a noção de vizinhança. PCA retém as dimensões de maior variância e elimina o ruído dimensional.
- **D — ERRADO:** `algorithm='brute'` é o método mais lento (força bruta pura). `'kd_tree'` e `'ball_tree'` são estruturas de dados que aceleram a busca de vizinhos. Usar `'brute'` pioraria o problema de velocidade.

---

**Questão 4**

Qual afirmação descreve corretamente a diferença entre KNN Clássico (`weights='uniform'`) e KNN Ponderado (`weights='distance'`)?

a) O KNN Ponderado usa mais vizinhos que o clássico, consultando K × distância pontos ao total.
b) No KNN Clássico, cada vizinho vota com peso 1/K; no Ponderado, o peso é inversamente proporcional à distância — vizinhos mais próximos têm mais influência.
c) O KNN Ponderado resolve a Maldição da Dimensionalidade ao ponderar features pela sua importância.
d) O KNN Clássico é sempre mais acurado que o ponderado por não sofrer instabilidade numérica quando a distância é próxima de zero.

**Resposta correta: B**

- **A — ERRADO:** Ambos consultam o mesmo número K de vizinhos. A diferença está apenas em como os votos são contados, não na quantidade de vizinhos.
- **B — CORRETO:** No KNN Clássico com K=5, cada vizinho contribui com peso 1/5 = 0.2. No Ponderado, se as distâncias são [0.1, 0.2, 0.3, 0.4, 0.5], os pesos são [10, 5, 3.3, 2.5, 2] — o vizinho mais próximo tem 5× mais influência que o mais distante. Isso é especialmente útil em regiões de fronteira onde um vizinho muito próximo é um sinal mais confiável da classe correta.
- **C — ERRADO:** O KNN Ponderado pondera os **vizinhos pela distância entre instâncias**, não as features pela importância. Feature importance é um conceito de seleção de features (como na Random Forest), não do mecanismo de votação do KNN.
- **D — ERRADO:** Quando um vizinho está a distância ≈ 0, seu peso 1/0 → infinito, causando instabilidade numérica no KNN Ponderado. Scikit-Learn resolve isso adicionando um epsilon, mas a afirmação de que o clássico é "sempre mais acurado" é falsa — depende do dataset.

---

*Fim da Seção 11 — Guia de Estudos Completo (Seções 1–11)*

---

---
