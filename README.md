# Otimização de Mola Helicoidal via Redes Neurais e Algoritmos Bio-inspirados

## 🔍 Resumo

Este projeto aplica uma abordagem híbrida para otimizar o projeto de molas helicoidais, substituindo a função analítica tradicional por uma **rede neural treinada**. Um **algoritmo evolutivo** realiza a otimização com base na previsão da rede.

Testamos quatro modificadores de gradiente diferentes durante o treinamento da rede neural, incluindo uma nova proposta baseada em `tanh + seno`, que obteve os melhores resultados.

- ✅ **Erro médio da RN**: 1,57% (comparado ao algoritmo HHO)
- ✅ **Todas as soluções são viáveis** (respeitam as restrições de projeto)

> Baseado em Villarrubia et al. (2018) e Yi et al. (2020)

---

## 📁 Estrutura do Projeto

```

├── main.py                      # Executa todo o processo
├── requirements.txt             # Dependências do projeto
├── spring\_optimization\_results/      # Resultados gerados
└── my\_spring\_optimization\_results/   # Resultados prontos para consulta

```

---

## 🚀 Como Executar (Passo a Passo para Iniciantes)

### 1. Baixe o projeto

- Acesse: https://github.com/SEU_USUARIO/SEU_REPOSITORIO
- Clique em **Code** > **Download ZIP**
- Extraia o conteúdo para uma pasta no seu computador

### 2. Crie um ambiente virtual

Abra o terminal ou prompt de comando na pasta do projeto e digite:

```

python -m venv venv

```

### 3. Ative o ambiente virtual

- **Windows**:
```

venv\Scripts\activate

```

- **Linux/Mac**:
```

source venv/bin/activate

```

### 4. Instale as dependências

```

pip install -r requirements.txt

```

### 5. Execute o projeto

```

python main.py

```

---

## 📊 Resultados

- Resultados gerados: `spring_optimization_results/`
- Exemplos prontos: `my_spring_optimization_results/`

---

## 📚 Referências

- Villarrubia, G. et al. (2018). *Artificial Neural Networks Used in Optimization Problems.*
- Yi, S. et al. (2020). *An Effective Optimization Method for Machine Learning Based on ADAM.*

---

🔧 Projeto acadêmico desenvolvido para experimentação com otimização híbrida em engenharia mecânica.
```
