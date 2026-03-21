from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import difflib
import sqlite3
from datetime import datetime
import os
import json

PORTFOLIO_DATA_FILE = "portfolio_data.json"

app = Flask(__name__)
CORS(app)

# Load knowledge base + models
try:
    kb = pd.read_csv("data/knowledge_base.csv")
    vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
    matrix = pickle.load(open("models/tfidf_matrix.pkl", "rb"))
    from src.preprocessing import TextPreprocessor
    pre = TextPreprocessor()
except Exception as e:
    print(f"Warning: Could not load models: {e}")
    kb = None
    vectorizer = None
    matrix = None
    pre = None

# Database setup
DB_NAME = "portfolio.db"

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Table for chat conversations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT
        )
    """)
    
    # Table for contact form submissions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contact_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT,
            status TEXT DEFAULT 'new'
        )
    """)
    
    conn.commit()
    conn.close()
    print("✓ Database initialized successfully")

# Initialize database on startup
init_db()

# Cache for recent questions
recent_cache = {}

# Similarity threshold
SIMILARITY_THRESHOLD = 0.15

def fuzzy_match(user_q, questions, cutoff=0.45):
    """Use fuzzy matching to find close question matches"""
    matches = difflib.get_close_matches(user_q, questions, n=1, cutoff=cutoff)
    if matches:
        return questions.index(matches[0])
    return None

def save_chat_message(session_id, user_msg, bot_response, ip_address):
    """Save chat message to database"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO chat_conversations (session_id, user_message, bot_response, ip_address)
            VALUES (?, ?, ?, ?)
        """, (session_id, user_msg, bot_response, ip_address))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error saving chat message: {e}")

def save_contact_submission(name, email, message, ip_address):
    """Save contact form submission to database"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO contact_submissions (name, email, message, ip_address)
            VALUES (?, ?, ?, ?)
        """, (name, email, message, ip_address))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error saving contact submission: {e}")
        return False

def no_cache(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route("/", methods=["GET"])
def index():
    """Serve the portfolio HTML"""
    r = make_response(send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_advanced.html")))
    return no_cache(r)

@app.route("/admin", methods=["GET"])
def admin():
    """Serve the admin dashboard HTML"""
    r = make_response(send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "admin_dashboard.html")))
    return no_cache(r)

@app.route("/cv", methods=["GET"])
def cv():
    """Serve the CV HTML"""
    r = make_response(send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cv.html")))
    return no_cache(r)

@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    """Chatbot endpoint - handles questions and stores conversations"""
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return "", 200
    user_q = request.json.get("question", "").strip()
    session_id = request.json.get("session_id", "unknown")
    ip_address = request.remote_addr
    
    if not user_q:
        return jsonify({"answer": "Veuillez poser une question."})
    
    # Check cache
    if user_q in recent_cache:
        answer = recent_cache[user_q]
        save_chat_message(session_id, user_q, answer, ip_address)
        return jsonify({"answer": answer})
    
    # Use ML models if available
    if kb is not None and vectorizer is not None and matrix is not None and pre is not None:
        try:
            # Preprocess and vectorize
            processed = pre.preprocess(user_q)
            q_vec = vectorizer.transform([processed])
            
            # Compute similarity
            scores = cosine_similarity(q_vec, matrix)[0]
            max_score = np.max(scores)
            
            if max_score >= SIMILARITY_THRESHOLD:
                idx = np.argmax(scores)
                answer = kb.iloc[idx]["response"]
            else:
                # Fallback to fuzzy matching
                idx = fuzzy_match(user_q, kb["question"].tolist())
                if idx is not None:
                    answer = kb.iloc[idx]["response"]
                else:
                    answer = "J'ai pas compris la question 😅. Pouvez-vous reformuler?"
        except Exception as e:
            print(f"Error in chatbot processing: {e}")
            answer = "Désolé, une erreur s'est produite. Essayez à nouveau."
    else:
        # Fallback responses if models not loaded
        answer = "Le chatbot est en cours de configuration. Veuillez contacter directement Yassine."
    
    # Cache and save
    recent_cache[user_q] = answer
    save_chat_message(session_id, user_q, answer, ip_address)
    
    return jsonify({"answer": answer})

@app.route("/contact", methods=["POST"])
def contact():
    """Contact form endpoint - stores client information"""
    try:
        data = request.json
        name = data.get("name", "").strip()
        email = data.get("email", "").strip()
        message = data.get("message", "").strip()
        ip_address = request.remote_addr
        
        # Validation
        if not name or not email:
            return jsonify({
                "success": False,
                "message": "Le nom et l'email sont obligatoires."
            }), 400
        
        # Basic email validation
        if "@" not in email or "." not in email:
            return jsonify({
                "success": False,
                "message": "Veuillez fournir une adresse email valide."
            }), 400
        
        # Save to database
        success = save_contact_submission(name, email, message, ip_address)
        
        if success:
            return jsonify({
                "success": True,
                "message": "Merci pour votre message ! Yassine vous contactera bientôt."
            })
        else:
            return jsonify({
                "success": False,
                "message": "Une erreur s'est produite. Veuillez réessayer."
            }), 500
            
    except Exception as e:
        print(f"Error in contact endpoint: {e}")
        return jsonify({
            "success": False,
            "message": "Erreur serveur. Veuillez réessayer."
        }), 500

@app.route("/conversations", methods=["GET"])
def get_conversations():
    """Get all chat conversations (admin endpoint)"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, session_id, user_message, bot_response, timestamp, ip_address
            FROM chat_conversations
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        rows = cursor.fetchall()
        conn.close()
        
        conversations = []
        for row in rows:
            conversations.append({
                "id": row[0],
                "session_id": row[1],
                "user_message": row[2],
                "bot_response": row[3],
                "timestamp": row[4],
                "ip_address": row[5]
            })
        
        return jsonify({
            "success": True,
            "count": len(conversations),
            "conversations": conversations
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route("/contacts", methods=["GET"])
def get_contacts():
    """Get all contact form submissions (admin endpoint)"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, name, email, message, timestamp, ip_address, status
            FROM contact_submissions
            ORDER BY timestamp DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        
        contacts = []
        for row in rows:
            contacts.append({
                "id": row[0],
                "name": row[1],
                "email": row[2],
                "message": row[3],
                "timestamp": row[4],
                "ip_address": row[5],
                "status": row[6]
            })
        
        return jsonify({
            "success": True,
            "count": len(contacts),
            "contacts": contacts
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route("/stats", methods=["GET"])
def get_stats():
    """Get statistics (admin endpoint)"""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Total conversations
        cursor.execute("SELECT COUNT(*) FROM chat_conversations")
        total_chats = cursor.fetchone()[0]
        
        # Total contacts
        cursor.execute("SELECT COUNT(*) FROM contact_submissions")
        total_contacts = cursor.fetchone()[0]
        
        # New contacts (status = 'new')
        cursor.execute("SELECT COUNT(*) FROM contact_submissions WHERE status = 'new'")
        new_contacts = cursor.fetchone()[0]
        
        # Chats today
        cursor.execute("""
            SELECT COUNT(*) FROM chat_conversations 
            WHERE DATE(timestamp) = DATE('now')
        """)
        chats_today = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            "success": True,
            "stats": {
                "total_chats": total_chats,
                "total_contacts": total_contacts,
                "new_contacts": new_contacts,
                "chats_today": chats_today
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if os.path.exists(DB_NAME) else "not found"
    })

DEFAULT_PORTFOLIO_DATA = {
    "personal": {
        "name": "Yassine Hafidi",
        "avail": "DISPONIBLE POUR OPPORTUNITÉS",
        "school": "ESTK Khenifra",
        "field": "Intelligence Artificielle & Développement",
        "subtitle": "Je crée des solutions intelligentes qui fusionnent code, données et IA.",
        "about": "Passionné par l'intelligence artificielle et le développement, je construis des projets qui allient technologie et créativité.",
        "aboutCards": [
            {"icon": "🧠", "title": "Intelligence Artificielle", "desc": "Expert en machine learning, deep learning et NLP. Je développe des modèles qui apprennent, comprennent et prédisent à partir de données complexes."},
            {"icon": "💻", "title": "Développement Full Stack", "desc": "Maîtrise du développement web frontend (HTML/CSS/JS) et backend (Python/Flask). Je crée des applications web modernes et performantes."},
            {"icon": "🗄️", "title": "Bases de Données", "desc": "Conception et gestion de bases de données SQL (MySQL) et NoSQL (MongoDB). Architecture de données robuste et optimisée."},
            {"icon": "🎓", "title": "Formation IDIA", "desc": "Étudiant en DUT Informatique, Développement et Intelligence Artificielle à l'ESTK Khenifra. Formation complète en développement et IA."}
        ]
    },
    "contact": {
        "email": "hafidiyassine83@gmail.com",
        "phone": "+212 613 612 618",
        "loc": "Khenifra, Maroc",
        "github": "yasine2006",
        "githubUrl": "https://github.com/yasine2006",
        "linkedinUrl": "https://www.linkedin.com/in/hafidi-yassine-5623a7352/",
        "instagramUrl": "https://www.instagram.com/yassine__hafidi_2_/"
    },
    "projects": [
        {"id": 1, "emoji": "🤖", "title": "Chatbot Intelligent IA", "desc": "Chatbot conversationnel utilisant TF-IDF et cosine similarity pour des réponses contextuelles. Backend Flask avec NLP et machine learning.", "tags": ["Python", "Flask", "NLP", "ML"], "link": "#"},
        {"id": 2, "emoji": "🌐", "title": "Portfolio Interactif", "desc": "Site web moderne avec animations avancées, design futuriste et intégration chatbot IA. Expérience utilisateur immersive.", "tags": ["HTML", "CSS", "JavaScript"], "link": "#"},
        {"id": 3, "emoji": "🚗", "title": "Agence Automobile", "desc": "Application web complète pour agence automobile avec gestion de véhicules, géolocalisation et chatbot intégré. Backend Python + SQLite.", "tags": ["Python", "SQLite", "Web"], "link": "#"}
    ],
    "skills": [
        {"name": "Machine Learning", "level": 85}, {"name": "Deep Learning", "level": 80},
        {"name": "Python", "level": 90}, {"name": "HTML & CSS", "level": 88},
        {"name": "JavaScript", "level": 75}, {"name": "NLP", "level": 78},
        {"name": "SQL / NoSQL", "level": 82}, {"name": "Flask", "level": 80},
        {"name": "C / C++", "level": 72}, {"name": "Linux / Unix", "level": 75},
        {"name": "Git & GitHub", "level": 85}, {"name": "Data Mining", "level": 76}
    ]
}

@app.route("/portfolio-data", methods=["GET"])
def get_portfolio_data():
    """Get stored portfolio data — returns saved data or defaults"""
    try:
        if os.path.exists(PORTFOLIO_DATA_FILE):
            with open(PORTFOLIO_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"success": True, "data": data})
        else:
            return jsonify({"success": True, "data": DEFAULT_PORTFOLIO_DATA})
    except Exception as e:
        return jsonify({"success": True, "data": DEFAULT_PORTFOLIO_DATA})

@app.route("/portfolio-data", methods=["POST"])
def save_portfolio_data():
    """Save portfolio data from admin"""
    try:
        data = request.json
        with open(PORTFOLIO_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ── APPLAUD ──
@app.route("/applaud", methods=["GET", "POST"])
def applaud():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS applauds (id INTEGER PRIMARY KEY, count INTEGER DEFAULT 0)""")
    cur.execute("SELECT count FROM applauds WHERE id=1")
    row = cur.fetchone()
    if not row:
        cur.execute("INSERT INTO applauds (id, count) VALUES (1, 0)")
        count = 0
    else:
        count = row[0]
    if request.method == "POST":
        count += 1
        cur.execute("UPDATE applauds SET count=? WHERE id=1", (count,))
        conn.commit()
    conn.close()
    return jsonify({"count": count})

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Portfolio Backend Server Starting...")
    print("=" * 60)
    print(f"✓ Database: {DB_NAME}")
    print(f"✓ Models loaded: {kb is not None and vectorizer is not None}")
    print("✓ Endpoints:")
    print("  - POST /ask          → Chatbot")
    print("  - POST /contact      → Contact form")
    print("  - GET  /conversations → View chat history")
    print("  - GET  /contacts     → View contact submissions")
    print("  - GET  /stats        → View statistics")
    print("  - GET  /health       → Health check")
    print("=" * 60)
    print("📡 Server running on http://127.0.0.1:5003")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5003, debug=True)
