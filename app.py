from flask import Flask, render_template, request, jsonify
import os
import time
from langgraph.types import Command, interrupt
from werkzeug.utils import secure_filename
from qna2 import agent, log_messages  # ✅ Replace with actual import

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}
pdf_path = None
is_interrupted = False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global pdf_path
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or unsupported file'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(save_path)
        print(f"[UPLOAD] File saved at: {save_path}")
        pdf_path = save_path
        
        return jsonify({'filepath': save_path})  # Send full path back
    except Exception as e:
        return jsonify({'error': f'File save failed: {str(e)}'}), 500
    
@app.route('/logs', methods=['GET'])
def get_logs():
    global log_messages
    logs_to_send = log_messages.copy()
    log_messages.clear()  # Optional: clear after sending
    return jsonify({'logs': logs_to_send})



@app.route('/ask', methods=['POST'])
def ask():
    global pdf_path, is_interrupted
    
    data = request.get_json()
    question = data.get('question', '').strip()
#     pdf_path = data.get('filepath', None)
    path = pdf_path
    pdf_path = None
    
    if not question:
        return jsonify({'answer': '❗ Please enter a question.'}), 400
    
    try:
        thread_config = {"configurable": {"thread_id": "some_id"}}
        print(f"[ASK] Question: {question}")
        print(f"[ASK] PDF Path: {path}")
        
        if not is_interrupted:
            result = agent.invoke({
                "query": question,
                "pdf_path": path,  # None if not provided
                "result": "",
                "imgs": [],
                "paper_url": None,
                "next_node": None,
            }, config=thread_config)
            state = agent.get_state(thread_config)
            
            #delay for log messages
            time.sleep(1)
            if state and state.tasks and state.tasks[0].interrupts:
                is_interrupted = True
                return jsonify({'answer':f"{result['result']}\n\n{state.tasks[0].interrupts[0].value['query']}"})
            return jsonify({'answer': result['result']})
        else:
            result = agent.invoke(Command(resume=question),config=thread_config)
            is_interrupted = False
            return jsonify({'answer': result['result']})
    except Exception as e:
#         print(f"result:{result}")
        return jsonify({'answer': f"❌ Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
