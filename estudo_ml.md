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
