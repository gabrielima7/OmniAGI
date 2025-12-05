# OmniAGI 🧠

**Sistema Operacional Cognitivo Soberano, Descentralizado e Autônomo**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/gabrielima7/OmniAGI/actions/workflows/ci.yml/badge.svg)](https://github.com/gabrielima7/OmniAGI/actions)

> ⚠️ **Status**: Em desenvolvimento ativo. Este projeto visa criar uma infraestrutura AGI, mas ainda não é uma AGI completa.

## 🌟 O Que é OmniAGI?

OmniAGI é uma infraestrutura de **Inteligência Artificial Geral** projetada para ser:

- **🏠 Soberana**: Roda 100% local, sem dependências de APIs externas
- **🔧 Descentralizada**: Arquitetura modular e extensível
- **🤖 Autônoma**: Capacidade de operar, aprender e evoluir independentemente

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                      INTERFACES                              │
│              CLI Unificada  │  API Server (OpenAI)          │
├─────────────────────────────────────────────────────────────┤
│                 CAMADA DE RACIOCÍNIO (Python)               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐   │
│  │  Agent  │ │ Memory  │ │  Tools  │ │   Life Daemon   │   │
│  ├─────────┤ ├─────────┤ ├─────────┤ ├─────────────────┤   │
│  │  Swarm  │ │ Vector  │ │Ouroboros│ │   Multimodal    │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                 MOTOR DE PERFORMANCE (Rust)                  │
│        LLM Inference  │  Quantization  │  GPU/CPU Backend   │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Instalação

### Requisitos Mínimos (para modelos quantizados pequenos)
- **CPU**: Qualquer x64 ou ARM64
- **RAM**: 4GB (modelos 1-3B parâmetros)
- **Disco**: 2GB + tamanho do modelo

### Requisitos Recomendados
- **RAM**: 8-16GB (modelos 7-13B parâmetros)
- **GPU**: NVIDIA com 4GB+ VRAM (opcional, mas 5-10x mais rápido)

### Instalação Rápida

```bash
# Clone o repositório
git clone https://github.com/gabrielima7/OmniAGI.git
cd OmniAGI

# Instalação padrão (com uv - recomendado)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Ou com pip
pip install -e .

# Com suporte CUDA (GPU NVIDIA)
pip install -e ".[cuda]"

# Com suporte Metal (Apple Silicon)
pip install -e ".[metal]"

# Instalação mínima (sistemas com pouca RAM)
pip install -e ".[minimal]"
```

### Modelos Recomendados por Hardware

| Hardware | Modelo Recomendado | RAM Necessária |
|----------|-------------------|----------------|
| 4GB RAM | Qwen2.5-1.5B-Q4 | ~2GB |
| 8GB RAM | Llama-3.2-3B-Q4 | ~3GB |
| 8GB RAM | Mistral-7B-Q4 | ~4.5GB |
| 16GB RAM | Llama-3.1-8B-Q4 | ~5GB |
| 16GB+ RAM | Mixtral-8x7B-Q4 | ~26GB |

> 💡 **Dica**: Use modelos quantizados em Q4_K_M para melhor equilíbrio entre qualidade e eficiência.

## 📖 Uso

### CLI Interativa

```bash
# Chat interativo
omni chat --model /caminho/para/modelo.gguf

# Com modelo pequeno para sistemas limitados
omni chat --model qwen2.5-1.5b-instruct-q4_k_m.gguf

# Iniciar o Life Daemon (modo autônomo)
omni daemon start

# Iniciar servidor API
omni serve --port 8000
```

### API Server (OpenAI-compatible)

```bash
# Iniciar servidor
omni serve --port 8000

# Testar API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "Olá!"}]
  }'
```

### Configuração via Variáveis de Ambiente

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

| Componente | Descrição |
|------------|-----------|
| **Core Engine** | Motor de inferência LLM (llama.cpp + Rust) |
| **Agent System** | Framework de agentes autônomos com ReAct |
| **Memory** | Memória vetorial (ChromaDB) + persistente |
| **Tools** | Filesystem, Code Sandbox, Web, Git |
| **Life Daemon** | Ciclo de vida autônomo |
| **Swarm** | Arquitetura multi-agente |
| **Multimodal** | Vision (PIL) + Audio (Whisper) |

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