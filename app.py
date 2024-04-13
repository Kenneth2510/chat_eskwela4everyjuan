from flask import Flask, request, jsonify
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from collections import defaultdict
import shutil
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
session_data = defaultdict(dict)


def get_pdf_text(pdf_docs, skip_images=True):
    texts = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            texts += page.extract_text()
    return texts

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

session_vector_stores = defaultdict(FAISS)

def get_vectorstore(session_id, text_chunks):
    if session_id in session_vector_stores:
        # Reset existing vector store
        del session_vector_stores[session_id]

    embeddings = OpenAIEmbeddings()
    session_vector_stores[session_id] = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    
    return session_vector_stores[session_id]


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def process_message(session_id, user_message, course, lesson):

    if session_id not in session_vector_stores:
        return jsonify({'error': 'Session ID not found'})

    vectorstore = session_vector_stores[session_id]
    

    conversation_chain = get_conversation_chain(vectorstore)
    
    user_message_with_instructions = f"!instructions! Act like a friendly tutor. Keep responses concise. When discussing the learner's info (e.g., business or course performance), refer to their specific data. Mention the learner's personal and business information when answering questions. The course is {course}, lesson {lesson}. This is the question: {user_message}"


    response = conversation_chain({
        "question": user_message_with_instructions, 
    })
    
    
    chat_history = response['chat_history']
    
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            user_question = message.content
        else:
            ai_response = message.content

    return ai_response



@app.route('/upload/<session_id>', methods=['POST'])
def upload(session_id):
    # Create a folder for the specified session ID if it doesn't exist
    session_folder = os.path.join('data', session_id)
    os.makedirs(session_folder, exist_ok=True)

    uploaded_files = request.files.getlist('files')
    file_names = []
    extracted_texts = ""
    file_paths = []
    for file in uploaded_files:
        # Save the file to the session folder
        file_path = os.path.join(session_folder, file.filename)
        file.save(file_path)
        # Add the file name to the list
        file_names.append(file.filename)
        file_paths.append(file_path)
        
    extracted_texts = get_pdf_text(file_paths)

    # Get the text chunks
    chunks = get_text_chunks(extracted_texts)

    # Create vector store
    vectorstore = get_vectorstore(session_id, chunks)

    # Store the PDF files and vector store for this session
    session_data[session_id]['files'] = file_names
    session_data[session_id]['vectorstore'] = vectorstore

    return jsonify({
        'file_names': file_names,
        'extracted_texts': extracted_texts,
    })


@app.route('/process_session/<session_id>', methods=['POST'])
def process_files(session_id):
    session_folder = os.path.join('sessionFolder', session_id)
    session_files = os.listdir(session_folder)
    file_paths = [os.path.join(session_folder, file) for file in session_files]

    # Get the text chunks from the session files
    extracted_texts = get_pdf_text(file_paths)
    new_chunks = get_text_chunks(extracted_texts)

    # Create a new vector store with all chunks
    vectorstore = get_vectorstore(session_id, new_chunks)

    # Update the session data with the new vector store
    session_data[session_id]['vectorstore'] = vectorstore
    
    return jsonify({'message': 'Files Processed successfully'})

@app.route('/add_file/<session_id>', methods=['POST'])
def add_file(session_id):


    # Get the uploaded files
    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        return jsonify({'error': 'No file uploaded'})

    session_folder = os.path.join('sessionFolder', session_id)
    os.makedirs(session_folder, exist_ok=True)
    
    file_names = []
    
    for file in uploaded_files:
        # Save the file to the session folder
        file_path = os.path.join(session_folder, file.filename)
        file.save(file_path)
        # Add the file name to the list
        file_names.append(file.filename)


    return jsonify({'message': 'Files added successfully'})



@app.route('/reset/<session_id>', methods=['POST'])
def reset(session_id):
    # Check if the session ID exists in the session_vector_stores
    if session_id in session_vector_stores:
        # Delete the session data
        del session_vector_stores[session_id]

        # Delete the session directory
        session_folder = os.path.join('sessionFolder', session_id)
        if os.path.exists(session_folder):
            # Remove files in the directory
            for filename in os.listdir(session_folder):
                file_path = os.path.join(session_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    return jsonify({'error': f'Failed to delete file: {str(e)}'})

            # Remove the directory
            try:
                os.rmdir(session_folder)
            except Exception as e:
                return jsonify({'error': f'Failed to delete directory: {str(e)}'})

            return jsonify({'message': f'Session {session_id} reset successfully'})
        else:
            return jsonify({'error': 'Session directory not found'})
    else:
        return jsonify({'error': 'Session ID not found'})



@app.route('/delete_all_files', methods=['POST'])
def delete_all_files():
    data_folder = 'sessionFolder'
    if os.path.exists(data_folder):
        # Remove files in the directory
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                return jsonify({'error': f'Failed to delete file: {str(e)}'})

        return jsonify({'message': 'All files deleted successfully'})
    else:
        return jsonify({'error': 'Data folder not found'})


@app.route('/list_all_files', methods=['GET'])
def list_all_files():
    data_folder = 'sessionFolder'
    if os.path.exists(data_folder):
        files_info = []
        for root, dirs, files in os.walk(data_folder):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(file_path, data_folder)
                files_info.append({'file_name': file_name, 'sub_folder': os.path.dirname(relative_path)})

        return jsonify({'files': files_info})
    else:
        return jsonify({'error': 'Data folder not found'})

@app.route('/chat/<session_id>', methods=['POST'])
def chat(session_id):
    # Check if the session ID exists in the session_vector_stores
    if session_id not in session_vector_stores:
        return jsonify({'error': 'Session ID not found'})

    # Get the user's message and additional parameters from the request
    user_message = request.json.get('question', '')
    course = request.json.get('course', '')
    lesson = request.json.get('lesson', '')

    # Process the message
    ai_response = process_message(session_id, user_message, course, lesson)

    # Return the AI's response
    return jsonify({'message': ai_response})

if __name__ == '__main__':
    app.run(debug=True)
