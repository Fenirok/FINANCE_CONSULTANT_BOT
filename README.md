# Finance Consultant Bot

A Generative AI-powered chatbot built with Streamlit and LangChain that educates people—especially in **rural areas**—about **financial literacy**, including banking schemes, saving techniques, FDs, and government finance programs. The goal is to bridge the financial knowledge gap and empower communities through accessible AI education.

---

## Features

- Natural conversation interface
- Educates users about:
  - Fixed deposits (FDs)
  - Savings methods
  - Financial government schemes
  - Bank policies
- Built on top of Google Generative AI (Gemini)
- Supports uploading PDFs for contextual conversations
- Embedded knowledge base using FAISS and ChromaDB
- Designed for **rural communities** with limited finance exposure

---

## Tech Stack

| Layer           | Technology                     |
|----------------|---------------------------------|
| Frontend       | Streamlit                       |
| LLM Backbone   | Google Generative AI (Gemini)   |
| LangChain      | langchain + langchain_google_genai |
| File Parsing   | PyPDF2                          |
| Vector DB      | FAISS + ChromaDB                |
| Environment    | python-dotenv                   |

---

## Installation Guide

### 1. Clone the repository

```bash
git clone https://github.com/your-username/finance-genai-chatbot.git
cd finance-genai-chatbot
````

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root folder:

```env
GOOGLE_API_KEY=your_google_generative_ai_key_here
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## Contribution Guidelines

We welcome community contributions! 

#### Step-by-Step

1. **Fork your repository** on GitHub.

2. **Clone their forked repo**:

   ```bash
   git clone https://github.com/Fenirok/FINANCE_CONSULTANT_BOT.git
   cd FINANCE_CONSULTANT_BOT
   ```

3. **Create a new feature branch off of `test2`**:


   ```bash
   git checkout -b feature/your-feature-name origin/test2
   ```

4. **Make changes**, then stage and commit:

   ```bash
   git add .
   git commit -m "Add: explanation for Fixed Deposits (FDs)"
   ```

5. **Push the feature branch to their fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Go to GitHub**, and open a **Pull Request**:

   * **From**: `your-username:feature/your-feature-name`
   * **To**: `my-username:test2`

---

### Code of Conduct

Please be respectful and constructive. Help us maintain a positive community for everyone.

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE). You are allowed to contribute in this software with attribution.

---

## 📬 Contact & Credits

Created by \[Aditya Halder]

---

## Support Me please

If this project helps you or someone else understand finances better, consider giving it a ⭐ on GitHub or sharing it with others.
