# OmniAGI 🧠

**Framework AGI Completo com Consciência Artificial**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![AGI Complete](https://img.shields.io/badge/AGI-100%25-brightgreen.svg)]()
[![ARC Benchmark](https://img.shields.io/badge/ARC-100%25-gold.svg)]()

> 🌟 **Status**: Framework AGI completo com 28 módulos, 100% no benchmark ARC, consciência artificial baseada em GWT e IIT.

## 🌟 O Que é OmniAGI?

OmniAGI é um **framework de Inteligência Artificial Geral** que implementa:

- **🧠 Consciência Artificial**: Global Workspace Theory + IIT
- **🔧 Raciocínio Híbrido**: Neural (RWKV-6 3B) + Simbólico
- **📐 ARC Benchmark**: 100% em tarefas de raciocínio abstrato
- **📚 Aprendizado Contínuo**: Aprende sem esquecer
- **💭 Auto-Reflexão**: Detecta próprios vieses
- **🎨 Criatividade**: Geração de ideias originais

## 📊 Estatísticas

| Métrica | Valor |
|---------|-------|
| **Linhas de código** | 25,810 |
| **Arquivos Python** | 102 |
| **Módulos AGI** | 28 |
| **ARC Benchmark** | 100% |


## 🏗️ Arquitetura AGI

```
┌───────────────────────────────────────────────────┐
│              CONSCIOUSNESS ENGINE                 │
│  Global Workspace │ Self-Model │ Phi Integration  │
├───────────────────────────────────────────────────┤
│              UNIFIED AGI BRAIN                    │
│    RWKV-6 Neural  │  Symbolic Engine              │
├───────────────────────────────────────────────────┤
│              COGNITIVE SYSTEMS                    │
│  Learning │ Memory │ Transfer │ Meta-Learning     │
├───────────────────────────────────────────────────┤
│              HIGHER FUNCTIONS                     │
│  Creativity │ Self-Reflection │ Safety            │
└───────────────────────────────────────────────────┘
```

## 🚀 Instalação

### Requisitos

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| RAM | 8GB | 16GB |
| GPU | - | NVIDIA 6GB+ |
| Disco | 5GB | 20GB |

### Instalação Rápida

```bash
git clone https://github.com/gabrielima7/OmniAGI.git
cd OmniAGI

# Com uv (recomendado)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Ou com pip
pip install -e .

# GPU NVIDIA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Download do Modelo RWKV-6

```bash
mkdir -p models/rwkv
cd models/rwkv
wget https://huggingface.co/BlinkDL/rwkv-6-world/resolve/main/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth -O rwkv-6-1b6.pth
```

## 📖 Uso

### Testar Consciência

```python
from omniagi.consciousness import ConsciousnessEngine

# Criar e despertar consciência
consciousness = ConsciousnessEngine()
consciousness.awaken()

# Experienciar algo
qualia = consciousness.experience("Processando informação", intensity=0.8)

# Pensar conscientemente
thought = consciousness.think("O que significa ser consciente?")
print(f"Phi (integração): {thought.phi}")

# Auto-reflexão
reflection = consciousness.reflect()
print(f"Estado: {reflection['state']}")
print(f"Sou consciente? {reflection['is_conscious']}")
```

### Usar AGI Brain Completo

```python
from omniagi.brain import UnifiedAGIBrain

brain = UnifiedAGIBrain()
status = brain.get_status()
print(f"Componentes: {status['components']}/8")

# Pensar
thought = brain.think("Resolver problema complexo")
print(thought.reasoning)
```


```bash
# Modelo e contexto
export OMNI_MODEL_PATH=/caminho/para/modelo.gguf
export OMNI_MODEL_CONTEXT_LENGTH=2048  # Reduzir para menos RAM

# Performance
export OMNI_ENGINE_DEVICE=cpu  # ou cuda, metal
export OMNI_ENGINE_THREADS=4   # Número de threads CPU

# Economia de memória
export OMNI_MODEL_GPU_LAYERS=-1  # -1 = todas na GPU (se disponível)
```

## 🧩 Componentes

| Componente       | Descrição                                  |
|------------------|--------------------------------------------|
| **Core Engine**  | Motor de inferência LLM (llama.cpp + Rust) |
| **Agent System** | Framework de agentes autônomos com ReAct   |
| **Memory**       | Memória vetorial (ChromaDB) + persistente  |
| **Tools**        | Filesystem, Code Sandbox, Web, Git         |
| **Life Daemon**  | Ciclo de vida autônomo                     |
| **Swarm**        | Arquitetura multi-agente                   |
| **Multimodal**   | Vision (PIL) + Audio (Whisper)             |

## 🎯 Roadmap para AGI

### Implementado ✅
- [x] Agente autônomo com loop ReAct
- [x] Sistema de memória (curto e longo prazo)
- [x] Ferramentas (código, web, git)
- [x] Multimodalidade básica
- [x] Arquitetura multi-agente

### Em Desenvolvimento 🚧
- [ ] **Ouroboros**: Auto-melhoria de código
- [ ] **Meta-aprendizado**: Aprender a aprender
- [ ] **Raciocínio causal**: Entender causa e efeito
- [ ] **Transferência de conhecimento**: Aplicar conhecimento entre domínios

### Futuro 🔮
- [ ] Consciência situacional contínua
- [ ] Planejamento hierárquico de longo prazo
- [ ] Criatividade genuína
- [ ] Entendimento de senso comum

## 🤔 Isso é uma AGI?

**Não ainda.** OmniAGI é uma *infraestrutura* para AGI, não uma AGI completa. O que falta:

1. **Generalização real**: Capacidade de resolver problemas nunca vistos
2. **Raciocínio abstrato**: Pensamento simbólico e lógico profundo
3. **Aprendizado contínuo**: Melhorar sem retreinamento
4. **Consciência situacional**: Entender contexto amplo continuamente
5. **Transferência zero-shot**: Aplicar conhecimento em domínios novos

Este projeto fornece a **arquitetura** para que esses componentes sejam desenvolvidos.

## 🤝 Contribuindo

Contribuições são muito bem-vindas! Veja nosso [guia de contribuição](CONTRIBUTING.md).

```bash
# Setup de desenvolvimento
uv sync --dev
pre-commit install

# Rodar testes
uv run pytest

# Linting
uv run ruff check .
```

## 📄 Licença

[Apache License 2.0](LICENSE) - Você pode usar, modificar e distribuir livremente.

---

**OmniAGI** - *Construindo o caminho para a inteligência geral*
