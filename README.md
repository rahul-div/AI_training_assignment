# AI Agent Collection: Research & E-commerce

This repository contains two powerful AI agents built with modern Python frameworks and LLM integration:

## 🔬 Research Agent
**Location:** `/main.py` (root directory)

An intelligent research assistant that can search Wikipedia, arXiv papers, and provide comprehensive answers with real-time streaming responses.

### Features
- 📚 **Multi-source Research**: Wikipedia articles and arXiv academic papers
- ⚡ **Streaming Responses**: Real-time progressive answer display using Server-Sent Events (SSE)
- 🌐 **Modern Web UI**: Clean, responsive interface built with FastHTML + PicoCSS
- 🔧 **Error Resilient**: Handles API failures gracefully with proper error messages
- 📊 **Token Usage Tracking**: Monitor API consumption in real-time

### Tech Stack
- **Framework**: FastHTML
- **LLM**: Pydantic-AI with Google Gemini
- **APIs**: Wikipedia API, arXiv API
- **Frontend**: HTMX, Server-Sent Events, PicoCSS
- **Observability**: Logfire integration

---

## 🛒 E-commerce Cart Agent
**Location:** `/E-comm/main.py`

A state-aware shopping cart assistant that combines AI-powered natural language interaction with traditional e-commerce UI controls.

### Features
- 🛍️ **Interactive Product Catalog**: Sneaker products with images and prices
- 🛒 **Dual Control Methods**:
  - Manual controls: Add/remove buttons, quantity adjustments (+/-)
  - AI Assistant: Natural language cart management
- 🤖 **Intelligent Cart Management**: AI agent with tools for cart operations
- 🎨 **Modern UI**: Responsive design with product images and smooth interactions
- 📊 **Real-time Updates**: HTMX-powered dynamic cart updates
- 🔍 **State Introspection**: AI can view and manage cart state accurately

### Tech Stack
- **Framework**: FastHTML
- **LLM**: Pydantic-AI with Google Gemini 2.5 Flash
- **Frontend**: HTMX, PicoCSS
- **State Management**: In-memory cart store with session support
- **Observability**: Logfire for tracking tool calls and token usage

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install fasthtml pydantic-ai logfire python-dotenv
```

### Environment Setup
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
LOGFIRE_TOKEN=your_logfire_token_here  # Optional
```

### Running the Research Agent
```bash
# From root directory
python main.py
```
Visit: http://localhost:5000

### Running the E-commerce Agent
```bash
# From root directory
cd E-comm
python main.py
```
Visit: http://localhost:5001/shop

---

## 🎯 Usage Examples

### Research Agent
- "What are the latest developments in quantum computing?"
- "Find research papers about machine learning in healthcare"
- "Explain the concept of transformers in AI"

### E-commerce Agent
- **Manual**: Click "Add to Cart" buttons, use +/- for quantities
- **AI Chat**: 
  - "Add 2 pairs of Nike shoes to my cart"
  - "What's currently in my cart?"
  - "Remove Adidas shoes and increase Puma to 3 pairs"

---

## 📁 Project Structure

```
Research_Agent/
├── main.py                 # Research Agent (FastHTML + Pydantic-AI)
├── E-comm/
│   └── main.py            # E-commerce Cart Agent
├── .env                   # Environment variables
├── .gitignore
└── README.md              # This file
```

---

## 🛠️ Key Features Comparison

| Feature | Research Agent | E-commerce Agent |
|---------|---------------|------------------|
| **LLM Integration** | ✅ Pydantic-AI | ✅ Pydantic-AI |
| **Streaming UI** | ✅ SSE | ❌ |
| **Tool Usage** | ✅ Search APIs | ✅ Cart Operations |
| **State Management** | ❌ | ✅ Persistent Cart |
| **HTMX Integration** | ✅ | ✅ |
| **Real-time Updates** | ✅ | ✅ |
| **Logfire Observability** | ✅ | ✅ |

---

## 🔧 Configuration

### Port Configuration
- **Research Agent**: Port 5000 (configurable in main.py)
- **E-commerce Agent**: Port 5001 (configurable in E-comm/main.py)

### Logfire Setup (Optional)
1. Sign up at [Logfire](https://logfire.pydantic.dev/)
2. Create a new project
3. Add your `LOGFIRE_TOKEN` to `.env`
4. View real-time agent interactions, tool calls, and token usage

---

## 🚀 Deployment

Both agents are production-ready and can be deployed to:
- **Docker**: Containerize with Python 3.11+ base image
- **Cloud Platforms**: Deploy to Render, Railway, Heroku, or similar
- **VPS**: Run with gunicorn or uvicorn for production serving

### Example Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000 5001
# Choose which agent to run
CMD ["python", "main.py"]  # Research Agent
# CMD ["python", "E-comm/main.py"]  # E-commerce Agent
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **FastHTML**: Modern web framework for Python
- **Pydantic-AI**: Type-safe LLM agent framework
- **Google Gemini**: Powerful language model
- **Logfire**: Observability and monitoring
- **PicoCSS**: Minimal CSS framework for clean UI

---

## 📞 Support

For questions or issues:
1. Check existing [GitHub Issues](../../issues)
2. Create a new issue with detailed description
3. Include error logs and environment details

---

**Built with ❤️ using FastHTML + Pydantic-AI**

*Two agents showcasing different AI interaction patterns: research/information retrieval and stateful e-commerce management.*
