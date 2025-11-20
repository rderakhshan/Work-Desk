# 🎨 AI Workdesk - Gradio UI Guide

## 🚀 Quick Start

### Launching the UI

```bash
# From your activated virtual environment
uv run ai-workdesk-ui
```

The UI will start at: **http://127.0.0.1:7860**

## 🔐 Login Credentials

### Default Users

| Username | Password  | Role  |
|----------|-----------|-------|
| `admin`  | `admin123`| Admin |
| `demo`   | `demo123` | Demo  |

> 💡 **Tip**: To change credentials, edit `src/ai_workdesk/ui/gradio_app.py` and modify the `USERS` dictionary.

## 📱 Features

### 1. 💬 Chat Tab
- **AI Assistant**: Interactive chat with GPT models
- **Real-time Responses**: Powered by OpenAI
- **Chat History**: Maintains conversation context
- **Copy Responses**: Built-in copy button for AI responses

#### Settings Panel
- **Model Selection**: Choose from GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Temperature Control**: Adjust creativity (0-2)
  - `0.0` = Very focused and deterministic
  - `0.7` = Balanced (default)
  - `2.0` = Very creative and random
- **Max Tokens**: Control response length (100-4096)

#### Controls
- **Send Button**: Submit your message
- **Clear Chat**: Reset conversation history
- **Retry Last**: (Future feature)

### 2. 📊 Status Tab
- **API Service Status**: Real-time authentication status
- **Configuration Display**: View current settings
- **Refresh Button**: Update status information

Shows status for:
- ✅ OpenAI (Configured)
- ❌ Anthropic (Not Configured)
- ❌ Google AI (Not Configured)
- ❌ Cohere (Not Configured)

### 3. ℹ️ About Tab
- Version information
- Feature list
- Supported services
- Tech stack details
- Quick tips

## 🎨 UI Design

The interface features:
- **Modern Glassmorphism**: Elegant, translucent design
- **Gradient Headers**: Eye-catching color gradients
- **Responsive Layout**: Adapts to different screen sizes
- **Dark Mode Compatible**: Soft theme with indigo/purple accents
- **Clean Typography**: Inter font family

## 💻 Usage Examples

### Basic Chat
1. Login with your credentials
2. Navigate to the **Chat** tab
3. Type your question in the message box
4. Click **Send** or press Enter
5. View AI response in the chat window

### Adjusting Settings
1. Use the **Settings** panel on the right
2. Select your preferred model
3. Adjust temperature for creativity
4. Set max tokens for response length
5. Settings apply to next message

### Example Conversations

**Creative Writing** (Temperature: 1.5-2.0)
```
User: Write a creative story about a robot learning to paint
AI: [Creative, imaginative response]
```

**Technical Help** (Temperature: 0.0-0.3)
```
User: Explain how Python decorators work
AI: [Precise, focused explanation]
```

**General Chat** (Temperature: 0.7, Default)
```
User: What's the weather like today?
AI: [Balanced, helpful response]
```

## 🔧 Advanced Configuration

### Customizing User Authentication

Edit `src/ai_workdesk/ui/gradio_app.py`:

```python
USERS = {
    "your_username": "your_password",
    "another_user": "another_password",
}
```

### Changing the Port

```python
ui.launch(
    share=False,
    server_port=8080,  # Change this
    auth=True,
)
```

### Creating Public Link

```python
ui.launch(
    share=True,  # Creates public Gradio link
    server_port=7860,
    auth=True,
)
```

### Disabling Authentication (Not Recommended)

```bash
# For development only
ui.launch(
    share=False,
    server_port=7860,
    auth=False,  # Disable login
)
```

## 🚨 Troubleshooting

### Issue: Login Not Working
- **Solution**: Check credentials match `USERS` dictionary
- Verify you're using correct username/password
- Check console for authentication logs

### Issue: AI Not Responding
- **Solution**: Verify OpenAI API key in `.env`
- Check Status tab for API configuration
- Look at logs in `logs/ai_workdesk.log`

### Issue: Port Already in Use
```bash
# Use a different port
Error: Address already in use
Solution: Change server_port to 7861, 7862, etc.
```

### Issue: Module Not Found
```bash
# Reinstall dependencies
uv sync --extra all
```

## 📝 Tips & Best Practices

### 1. Token Management
- Monitor token usage in logs
- Use lower max_tokens for quick responses
- Higher max_tokens for detailed answers

### 2. Temperature Guidelines
- **0.0-0.3**: Code, math, factual questions
- **0.4-0.8**: General conversation, explanations
- **0.9-1.5**: Creative writing, brainstorming
- **1.6-2.0**: Very creative, experimental

### 3. Chat History
- Clear history when changing topics
- History affects context and token usage
- Longer history = more context but higher tokens

### 4. Security
- Always use authentication in production
- Change default passwords immediately
- Don't expose publicly without proper security
- Keep API keys in `.env`, never commit them

## 🎯 Common Use Cases

### Code Assistant
```
Settings: Model = gpt-4o, Temperature = 0.2
Query: "Write a Python function to calculate fibonacci numbers"
```

### Creative Writing
```
Settings: Model = gpt-4o, Temperature = 1.5
Query: "Write a short story about time travel paradox"
```

### Data Analysis Help
```
Settings: Model = gpt-4o, Temperature = 0.3
Query: "Explain pandas DataFrame operations"
```

### General Q&A
```
Settings: Model = gpt-4o-mini, Temperature = 0.7
Query: "What are the benefits of using UV package manager?"
```

## 🔗 Quick Links

- **Local URL**: http://127.0.0.1:7860
- **Logs**: `logs/ai_workdesk.log`
- **Configuration**: `.env`
- **Source Code**: `src/ai_workdesk/ui/gradio_app.py`

## 📞 Getting Help

If you encounter issues:
1. Check `logs/ai_workdesk.log` for errors
2. Verify API keys in `.env`
3. Ensure all dependencies installed: `uv sync --extra all`
4. Check the Status tab in the UI

---

**Made with ❤️ using AI Workdesk v0.1.0**
