import gradio as gr
import torch
import black
import autopep8
import ast
import re
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# 1. CONFIGURATION
class ChatbotConfig:
    MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"
    DEFAULT_TEMP = 0.7
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_TOP_P = 0.95

# 2. SURGERY FUNCTIONS
def detect_python_code(text):
    patterns = [r'def\s+\w+\s*\(', r'class\s+\w+', r'import\s+\w+', r'if\s+.:', r'for\s+.\s+in\s+', r'print\(.*\)']
    return any(re.search(p, text) for p in patterns)

def check_syntax(code):
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax Error (line {e.lineno}): {e.msg}"

def fix_python_code(code):
    code = code.expandtabs(4).strip()
    try:
        fixed_code = black.format_str(code, mode=black.FileMode())
    except:
        try:
            fixed_code = autopep8.fix_code(code)
        except:
            fixed_code = code
    return fixed_code

# 3. EXPORT FUNCTIONS
def export_to_pdf(history):
    filename = f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    y = 750
    c.drawString(100, y, "The Syntax Surgeon - Chat History")
    y -= 30
    for user, bot in history:
        user_txt = str(user)[:50]
        bot_txt = str(bot)[:50]
        c.drawString(50, y, f"User: {user_txt}...")
        y -= 20
        c.drawString(50, y, f"AI: {bot_txt}...")
        y -= 40
        if y < 100:
            c.showPage()
            y = 750
    c.save()
    return filename

# 4. MODEL LOADING (Wait for 5-10 mins)
print("Loading IBM Granite Model... Innum konja nerathula start aayidum.")
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-generation", model=ChatbotConfig.MODEL_NAME, device=device, torch_dtype=torch.float16 if device == 0 else torch.float32)

# 5. UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🩺 The Syntax Surgeon\n*Team:* P.Ulaganathan, Subash, Veeraiya, Shajith Khan, Vignesh")
    chatbot = gr.Chatbot(label="Conversation History")
    msg = gr.Textbox(label="Enter code or message")
    
    with gr.Row():
        temp = gr.Slider(0.1, 1.0, value=0.7, label="Temperature")
        tokens = gr.Slider(50, 1024, value=512, label="Max Tokens")
    
    with gr.Row():
        submit = gr.Button("Send")
        export_btn = gr.Button("Download PDF")
        pdf_file = gr.File(label="Output PDF")

    def respond(message, chat_history, t, tok):
        if detect_python_code(message):
            valid, err = check_syntax(message)
            if not valid:
                fixed = fix_python_code(message)
                bot_message = f"🩹 *Surgery Done!\n\nError:* {err}\n\n*Fixed Code:*\npython\n{fixed}\n"
            else:
                bot_message = "✅ Code is correct!"
        else:
            out = pipe(message, max_new_tokens=tok, temperature=t, do_sample=True)
            bot_message = out[0]['generated_text']
        
        chat_history.append((message, bot_message))
        return "", chat_history

    submit.click(respond, [msg, chatbot, temp, tokens], [msg, chatbot])
    export_btn.click(export_to_pdf, [chatbot], pdf_file)

demo.launch(share=True)
