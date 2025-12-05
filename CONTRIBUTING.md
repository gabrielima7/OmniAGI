# Contribuindo para o OmniAGI

Obrigado pelo interesse em contribuir! 🎉

## 🚀 Como Começar

### Setup de Desenvolvimento

```bash
# Clone o repositório
git clone https://github.com/gabrielima7/OmniAGI.git
cd OmniAGI

# Instale uv (gerenciador de pacotes rápido)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Instale dependências de desenvolvimento
uv sync --dev

# Configure pre-commit hooks
uv run pre-commit install
```

### Rodando Testes

```bash
# Todos os testes
uv run pytest

# Com cobertura
uv run pytest --cov=omniagi

# Testes específicos
uv run pytest tests/test_agent.py -v
```

### Linting e Formatação

```bash
# Verificar código
uv run ruff check .

# Formatar código
uv run ruff format .

# Type checking
uv run mypy omniagi/
```

## 📋 Fluxo de Contribuição

1. **Fork** o repositório
2. Crie uma **branch** para sua feature: `git checkout -b feature/nome-da-feature`
3. Faça **commits** com mensagens claras
4. Rode os **testes** e **linting**
5. Abra um **Pull Request**

## 💡 Áreas que Precisam de Ajuda

### Alta Prioridade
- [ ] Implementação do Ouroboros (auto-melhoria)
- [ ] Testes unitários e integração
- [ ] Documentação e exemplos
- [ ] Otimizações de memória

### Média Prioridade
- [ ] Integração com mais modelos (LLaVA, Whisper)
- [ ] UI web para monitoramento
- [ ] Plugins e extensões

### Sempre Bem-vindas
- Correção de bugs
- Melhorias de performance
- Traduções
- Feedback e sugestões

## 📝 Padrões de Código

### Python
- Use type hints
- Docstrings em funções públicas
- Siga o estilo ruff/black

### Rust
- `cargo fmt` antes de commits
- Sem warnings do `clippy`

### Commits
```
tipo(escopo): descrição curta

Corpo opcional explicando o que e porquê
```

Tipos: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## 🐛 Reportando Bugs

Use o template de issue e inclua:
- Versão do Python/OS
- Passos para reproduzir
- Comportamento esperado vs atual
- Logs relevantes

## 💬 Dúvidas?

Abra uma issue com a tag `question` ou inicie uma discussão.

---

Obrigado por ajudar a construir o futuro da IA! 🧠
