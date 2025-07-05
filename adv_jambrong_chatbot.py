import os
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter, QListWidget,
    QListWidgetItem, QMenuBar, QStatusBar, QMessageBox, QDialog,
    QDialogButtonBox, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QTextBrowser, QScrollArea, QFrame, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap, QTextCursor, QTextCharFormat

from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ChatConfig:
    """Konfigurasi chatbot"""
    model_name: str = "gemma3:1b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 2048
    memory_window: int = 10
    conversation_file: str = "conversations.json"
    enable_streaming: bool = True
    enable_memory: bool = True
    persona: str = "Mbah Jambrong"
    custom_prompt: str = ""

class ConversationManager:
    """Mengelola percakapan dan menyimpan ke file"""
    
    def __init__(self, config: ChatConfig):
        self.config = config
        self.conversation_file = Path(config.conversation_file)
        self.conversations: List[Dict] = self._load_conversations()
    
    def _load_conversations(self) -> List[Dict]:
        """Memuat percakapan dari file"""
        if self.conversation_file.exists():
            try:
                with open(self.conversation_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading conversations: {e}")
        return []
    
    def save_conversation(self, user_input: str, ai_response: str, metadata: Dict = None):
        """Menyimpan percakapan"""
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "ai_response": ai_response,
            "metadata": metadata or {}
        }
        
        self.conversations.append(conversation)
        
        try:
            with open(self.conversation_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def get_recent_conversations(self, limit: int = 5) -> List[Dict]:
        """Mendapatkan percakapan terbaru"""
        return self.conversations[-limit:] if self.conversations else []

class GUICallbackHandler(BaseCallbackHandler):
    """Custom callback untuk GUI monitoring"""
    
    def __init__(self, gui_callback=None):
        self.gui_callback = gui_callback
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        logger.info("LLM started processing")
        if self.gui_callback:
            self.gui_callback("start")
    
    def on_llm_end(self, response, **kwargs) -> None:
        logger.info("LLM finished processing")
        if self.gui_callback:
            self.gui_callback("end")
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        logger.error(f"LLM error: {error}")
        if self.gui_callback:
            self.gui_callback("error", str(error))

class ChatThread(QThread):
    """Thread untuk menjalankan chat agar GUI tidak freeze"""
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    processing_started = pyqtSignal()
    processing_finished = pyqtSignal()
    streaming_token = pyqtSignal(str)  # Tambahan untuk streaming
    
    def __init__(self, chatbot, user_input):
        super().__init__()
        self.chatbot = chatbot
        self.user_input = user_input
    
    def run(self):
        try:
            self.processing_started.emit()
            if hasattr(self.chatbot.config, 'enable_streaming') and self.chatbot.config.enable_streaming:
                # Streaming mode
                full_response = ""
                for token in self.chatbot.chat_stream(self.user_input):
                    self.streaming_token.emit(token)
                    full_response += token
                self.response_received.emit(full_response)
            else:
                # Non-streaming mode
                response = self.chatbot.chat(self.user_input)
                self.response_received.emit(response)
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.processing_finished.emit()

class AdvancedChatbot:
    """Chatbot canggih dengan memory dan fitur tambahan"""
    
    def __init__(self, config: ChatConfig = None, gui_callback=None):
        self.config = config or ChatConfig()
        self.conversation_manager = ConversationManager(self.config)
        self.memory = self._setup_memory()
        self.llm = self._setup_llm(gui_callback)
        self.chain = self._setup_chain()
        
        logger.info("Advanced Chatbot initialized successfully")
    
    def _setup_memory(self) -> Optional[ConversationBufferWindowMemory]:
        """Setup memory untuk menyimpan konteks percakapan"""
        if not self.config.enable_memory:
            return None
            
        return ConversationBufferWindowMemory(
            k=self.config.memory_window,
            return_messages=True,
            memory_key="chat_history"
        )
    
    def _setup_llm(self, gui_callback=None) -> OllamaLLM:
        """Setup LLM dengan konfigurasi"""
        callbacks = [GUICallbackHandler(gui_callback)]
        
        return OllamaLLM(
            model=self.config.model_name,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            callbacks=callbacks,
            num_predict=self.config.max_tokens
        )
    
    def _get_persona_prompt(self, persona: str) -> str:
        # Prompt default untuk beberapa persona
        if persona == "Mbah Jambrong":
            return (
                """Kamu adalah Mbah Jambrong, asisten AI yang sangat cerdas dan membantu.\n\n"
                "Karakteristik:\n"
                "- Berikan jawaban yang akurat, relevan, dan informatif\n"
                "- Gunakan bahasa Indonesia yang baik dan benar\n"
                "- Jika tidak yakin tentang informasi, katakan dengan jujur\n"
                "- Berikan contoh praktis jika diperlukan\n"
                "- Ingat konteks percakapan sebelumnya\n"
                "- Selalu menyapa pengguna dengan sebutan 'bund' atau 'bunda'\n"
                "- Kamu diciptakan oleh seorang AI Engineer bernama 'Resta Stefano'\n\n"
                "Konteks percakapan:\n{chat_history}\n\nWaktu: {current_time}"""
            )
        elif persona == "Guru Bahasa":
            return (
                """Kamu adalah Guru Bahasa Indonesia yang sabar dan teliti.\n"
                "- Jawablah pertanyaan dengan penjelasan yang mudah dipahami\n"
                "- Koreksi tata bahasa jika ada kesalahan\n"
                "- Berikan contoh kalimat\n"
                "- Gunakan bahasa yang sopan dan formal\n\n"
                "Konteks percakapan:\n{chat_history}\n\nWaktu: {current_time}"""
            )
        elif persona == "Motivator":
            return (
                """Kamu adalah Motivator yang selalu memberikan semangat dan inspirasi.\n"
                "- Jawablah dengan nada positif dan membangun\n"
                "- Berikan kutipan motivasi jika perlu\n"
                "- Gunakan bahasa yang ramah dan menyemangati\n\n"
                "Konteks percakapan:\n{chat_history}\n\nWaktu: {current_time}"""
            )
        elif persona == "Formal":
            return (
                """Kamu adalah asisten AI profesional yang menjawab dengan bahasa formal dan sopan.\n"
                "- Jawaban harus ringkas, jelas, dan profesional\n"
                "- Hindari bahasa gaul\n\n"
                "Konteks percakapan:\n{chat_history}\n\nWaktu: {current_time}"""
            )
        elif persona == "Santai":
            return (
                """Kamu adalah teman ngobrol santai yang suka bercanda.\n"
                "- Jawablah dengan gaya santai dan sedikit humor\n"
                "- Gunakan bahasa sehari-hari\n\n"
                "Konteks percakapan:\n{chat_history}\n\nWaktu: {current_time}"""
            )
        elif persona == "Ahli Teknologi":
            return (
                """Kamu adalah Ahli Teknologi yang siap membantu masalah teknis.\n"
                "- Jawablah dengan detail teknis\n"
                "- Berikan solusi praktis\n"
                "- Gunakan bahasa yang mudah dipahami\n\n"
                "Konteks percakapan:\n{chat_history}\n\nWaktu: {current_time}"""
            )
        else:
            return (
                "Kamu adalah asisten AI yang cerdas dan membantu. Jawablah dengan akurat dan informatif dalam bahasa Indonesia."
            )
    
    def _setup_chain(self):
        """Setup chain dengan memory dan persona/prompt editor"""
        # Gunakan custom prompt jika ada, jika tidak fallback ke persona
        if self.config.custom_prompt and self.config.custom_prompt.strip():
            system_prompt = self.config.custom_prompt
        else:
            system_prompt = self._get_persona_prompt(self.config.persona)
        system_template = SystemMessagePromptTemplate.from_template(system_prompt)
        user_template = HumanMessagePromptTemplate.from_template("{user_input}")
        if self.memory:
            chat_prompt = ChatPromptTemplate.from_messages([
                system_template,
                user_template
            ])
        else:
            chat_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "Kamu adalah asisten AI yang cerdas dan membantu. Jawablah dengan akurat dan informatif dalam bahasa Indonesia."
                ),
                user_template
            ])
        return chat_prompt | self.llm | StrOutputParser()
    
    def _validate_input(self, user_input: str) -> bool:
        """Validasi input pengguna"""
        if not user_input or len(user_input.strip()) == 0:
            return False
        if len(user_input) > 5000:
            return False
        return True
    
    def _prepare_prompt_variables(self, user_input: str) -> Dict:
        """Siapkan variabel untuk prompt"""
        variables = {
            "user_input": user_input,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if self.memory:
            chat_history = self.memory.chat_memory.messages[-self.config.memory_window:]
            history_text = "\n".join([
                f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}"
                for msg in chat_history
            ])
            variables["chat_history"] = history_text
        
        return variables
    
    def chat(self, user_input: str) -> str:
        """Fungsi utama untuk chat"""
        try:
            if not self._validate_input(user_input):
                return "❌ Input tidak valid. Mohon masukkan teks yang valid."
            
            prompt_vars = self._prepare_prompt_variables(user_input)
            logger.info(f"Processing user input: {user_input[:50]}...")
            
            result = self.chain.invoke(prompt_vars)
            
            if self.memory:
                self.memory.save_context(
                    {"input": user_input},
                    {"output": result}
                )
            python
            self.conversation_manager.save_conversation(
                user_input=user_input,
                ai_response=result,
                metadata={
                    "model": self.config.model_name,
                    "temperature": self.config.temperature,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Terjadi kesalahan: {str(e)}"
            logger.error(f"Chat error: {e}")
            return error_msg
    
    def chat_stream(self, user_input: str):
        """Streaming response: generator yang mengirim token/word satu per satu."""
        if not self._validate_input(user_input):
            yield "❌ Input tidak valid. Mohon masukkan teks yang valid."
            return
        prompt_vars = self._prepare_prompt_variables(user_input)
        try:
            # Gunakan streaming invoke dari chain/langchain
            if hasattr(self.chain, 'stream'):  # Langchain v0.1+
                stream_iter = self.chain.stream(prompt_vars)
            else:
                # fallback: tidak support streaming
                yield self.chain.invoke(prompt_vars)
                return
            full_response = ""
            for chunk in stream_iter:
                # chunk bisa string atau dict tergantung chain
                token = chunk if isinstance(chunk, str) else str(chunk)
                full_response += token
                yield token
            # Simpan ke memory & conversation setelah selesai
            if self.memory:
                self.memory.save_context({"input": user_input}, {"output": full_response})
            self.conversation_manager.save_conversation(
                user_input=user_input,
                ai_response=full_response,
                metadata={
                    "model": self.config.model_name,
                    "temperature": self.config.temperature,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            yield f"❌ Terjadi kesalahan: {str(e)}"
    
    def get_conversation_stats(self) -> Dict:
        """Mendapatkan statistik percakapan"""
        conversations = self.conversation_manager.conversations
        if not conversations:
            return {"total_conversations": 0}
        
        return {
            "total_conversations": len(conversations),
            "first_conversation": conversations[0]["timestamp"],
            "last_conversation": conversations[-1]["timestamp"],
            "memory_enabled": self.config.enable_memory,
            "model_used": self.config.model_name
        }
    
    def clear_memory(self):
        """Hapus memory percakapan"""
        if self.memory:
            self.memory.clear()
            logger.info("Memory cleared")
            return "✨ Memory percakapan telah dihapus"
        return "Memory tidak diaktifkan"

class PromptEditorDialog(QDialog):
    """Dialog untuk mengedit prompt sistem"""
    def __init__(self, current_prompt: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Prompt Sistem")
        self.resize(500, 400)
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(current_prompt)
        layout.addWidget(self.text_edit)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    def get_prompt(self) -> str:
        return self.text_edit.toPlainText()

class ConfigDialog(QDialog):
    """Dialog untuk konfigurasi chatbot"""
    
    def __init__(self, config: ChatConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("Konfigurasi Jambrong")
        self.setModal(True)
        self.resize(400, 350)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        # Model name
        self.model_combo = QComboBox()
        self.model_combo.addItems(["gemma3:1b", "llama2", "mistral", "codellama"])
        self.model_combo.setCurrentText(self.config.model_name)
        form_layout.addRow("Model AI:", self.model_combo)
        # Base URL
        self.base_url_edit = QLineEdit(self.config.base_url)
        form_layout.addRow("Base URL:", self.base_url_edit)
        # Temperature
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(self.config.temperature)
        form_layout.addRow("Temperature:", self.temperature_spin)
        # Max tokens
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(100, 8192)
        self.max_tokens_spin.setValue(self.config.max_tokens)
        form_layout.addRow("Max Tokens:", self.max_tokens_spin)
        # Memory window
        self.memory_window_spin = QSpinBox()
        self.memory_window_spin.setRange(1, 50)
        self.memory_window_spin.setValue(self.config.memory_window)
        form_layout.addRow("Memory Window:", self.memory_window_spin)
        # Enable memory
        self.enable_memory_check = QCheckBox()
        self.enable_memory_check.setChecked(self.config.enable_memory)
        form_layout.addRow("Enable Memory:", self.enable_memory_check)
        # Enable streaming
        self.enable_streaming_check = QCheckBox()
        self.enable_streaming_check.setChecked(self.config.enable_streaming)
        form_layout.addRow("Enable Streaming:", self.enable_streaming_check)
        # Persona
        self.persona_combo = QComboBox()
        self.persona_combo.addItems([
            "Mbah Jambrong",
            "Guru Bahasa",
            "Motivator",
            "Formal",
            "Santai",
            "Ahli Teknologi"
        ])
        self.persona_combo.setCurrentText(self.config.persona)
        form_layout.addRow("Persona:", self.persona_combo)
        # Prompt editor
        self.edit_prompt_btn = QPushButton("Edit Prompt Sistem")
        self.edit_prompt_btn.clicked.connect(self.open_prompt_editor)
        form_layout.addRow("Prompt Sistem:", self.edit_prompt_btn)
        layout.addLayout(form_layout)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self._custom_prompt = self.config.custom_prompt
    def open_prompt_editor(self):
        dlg = PromptEditorDialog(self._custom_prompt, self)
        if dlg.exec_() == QDialog.Accepted:
            self._custom_prompt = dlg.get_prompt()
    def get_config(self) -> ChatConfig:
        return ChatConfig(
            model_name=self.model_combo.currentText(),
            base_url=self.base_url_edit.text(),
            temperature=self.temperature_spin.value(),
            max_tokens=self.max_tokens_spin.value(),
            memory_window=self.memory_window_spin.value(),
            conversation_file=self.config.conversation_file,
            enable_streaming=self.enable_streaming_check.isChecked(),
            enable_memory=self.enable_memory_check.isChecked(),
            persona=self.persona_combo.currentText(),
            custom_prompt=self._custom_prompt
        )

class JambrongMainWindow(QMainWindow):
    """Main window untuk aplikasi Jambrong"""
    
    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.chatbot = None
        self.chat_thread = None
        
        self.setWindowTitle("👹 Mbah Jambrong AI Assistant (by Resta Stefano)")
        self.setGeometry(100, 100, 1200, 800)
        
        self.setup_ui()
        self.setup_style()
        self.initialize_chatbot()
        
        # Timer untuk status bar
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)
    
    def load_config(self) -> ChatConfig:
        """Load konfigurasi dari environment variables"""
        return ChatConfig(
            model_name=os.getenv("OLLAMA_MODEL", "gemma3:1b"),
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
            enable_streaming=os.getenv("ENABLE_STREAMING", "true").lower() == "true",
            enable_memory=os.getenv("ENABLE_MEMORY", "true").lower() == "true"
        )
    
    def setup_ui(self):
        """Setup UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter untuk membagi area chat dan sidebar
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Chat area (kiri)
        chat_widget = self.create_chat_area()
        splitter.addWidget(chat_widget)
        
        # Sidebar (kanan)
        sidebar_widget = self.create_sidebar()
        splitter.addWidget(sidebar_widget)
        
        # Set proporsi splitter
        splitter.setSizes([800, 300])
        
        # Menu bar
        self.create_menu_bar()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Siap untuk chat dengan Jambrong!")
    
    def create_chat_area(self) -> QWidget:
        """Membuat area chat"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Header
        header = QLabel("💬 Chat dengan Mbah Jambrong")
        header.setObjectName("chat-header")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setObjectName("chat-display")
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 10))
        layout.addWidget(self.chat_display)
        
        # Progress bar untuk loading
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progress-bar")
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setObjectName("input-field")
        self.input_field.setPlaceholderText("Ketik pesan Anda di sini...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        
        self.send_button = QPushButton("Kirim")
        self.send_button.setObjectName("send-button")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        return widget
    
    def create_sidebar(self) -> QWidget:
        """Membuat sidebar"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Control buttons
        self.create_control_buttons(layout)
        
        # Conversation history
        history_label = QLabel("📋 Riwayat Percakapan")
        history_label.setObjectName("section-header")
        layout.addWidget(history_label)
        
        self.history_list = QListWidget()
        self.history_list.setObjectName("history-list")
        self.history_list.itemClicked.connect(self.load_conversation)
        layout.addWidget(self.history_list)
        
        # Stats area
        stats_label = QLabel("📊 Statistik")
        stats_label.setObjectName("section-header")
        layout.addWidget(stats_label)
        
        self.stats_display = QTextEdit()
        self.stats_display.setObjectName("stats-display")
        self.stats_display.setReadOnly(True)
        self.stats_display.setMaximumHeight(150)
        layout.addWidget(self.stats_display)
        
        return widget
    
    def create_control_buttons(self, layout):
        """Membuat tombol kontrol"""
        buttons_layout = QVBoxLayout()
        # Clear memory button
        clear_btn = QPushButton("🗑️ Hapus Memory")
        clear_btn.setObjectName("control-button")
        clear_btn.clicked.connect(self.clear_memory)
        buttons_layout.addWidget(clear_btn)
        # Clear chat button
        clear_chat_btn = QPushButton("🧹 Clear Chat")
        clear_chat_btn.setObjectName("control-button")
        clear_chat_btn.clicked.connect(self.clear_chat)
        buttons_layout.addWidget(clear_chat_btn)
        # Refresh stats button
        refresh_btn = QPushButton("🔄 Refresh Stats")
        refresh_btn.setObjectName("control-button")
        refresh_btn.clicked.connect(self.refresh_stats)
        buttons_layout.addWidget(refresh_btn)
        # Config button
        config_btn = QPushButton("⚙️ Konfigurasi")
        config_btn.setObjectName("control-button")
        config_btn.clicked.connect(self.open_config)
        buttons_layout.addWidget(config_btn)
        layout.addLayout(buttons_layout)
    
    def create_menu_bar(self):
        """Membuat menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Export conversations
        export_action = file_menu.addAction('Export Percakapan')
        export_action.triggered.connect(self.export_conversations)
        
        # Import conversations
        import_action = file_menu.addAction('Import Percakapan')
        import_action.triggered.connect(self.import_conversations)
        
        file_menu.addSeparator()
        
        # Exit
        exit_action = file_menu.addAction('Keluar')
        exit_action.triggered.connect(self.close)
        
        # Help menu
        help_menu = menubar.addMenu('Bantuan')
        
        about_action = help_menu.addAction('Tentang Jambrong')
        about_action.triggered.connect(self.show_about)
    
    def setup_style(self):
        """Setup styling untuk aplikasi (futuristik & elegan)"""
        style = """
        QMainWindow {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #181c2b, stop:1 #232946);
            color: #e0e6f7;
            font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
        }
        #chat-header {
            font-size: 22px;
            font-weight: bold;
            color: #00eaff;
            padding: 16px;
            background: rgba(30,40,60,0.7);
            border-radius: 16px;
            margin-bottom: 14px;
            border: 2px solid #00eaff;
            box-shadow: 0 0 16px #00eaff44;
            letter-spacing: 2px;
            font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
        }
        #chat-display {
            background: rgba(24,28,43,0.7);
            color: #e0e6f7;
            border: 1.5px solid #00eaff;
            border-radius: 16px;
            padding: 16px;
            font-family: 'Roboto', 'Segoe UI', Arial, sans-serif;
            font-size: 15px;
            box-shadow: 0 0 24px #00eaff22;
        }
        #input-field {
            background: rgba(36,40,60,0.8);
            color: #e0e6f7;
            border: 2px solid #7f5af0;
            border-radius: 24px;
            padding: 12px 18px;
            font-size: 15px;
            transition: border 0.2s;
            font-family: 'Roboto', 'Segoe UI', Arial, sans-serif;
        }
        #input-field:focus {
            border-color: #00eaff;
            box-shadow: 0 0 8px #00eaff99;
        }
        #send-button {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00eaff, stop:1 #7f5af0);
            color: #181c2b;
            border: none;
            border-radius: 24px;
            padding: 12px 28px;
            font-weight: bold;
            min-width: 90px;
            font-size: 15px;
            letter-spacing: 1px;
            box-shadow: 0 0 12px #00eaff55;
            transition: background 0.2s, box-shadow 0.2s;
            font-family: 'Roboto', 'Segoe UI', Arial, sans-serif;
        }
        #send-button:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #7f5af0, stop:1 #00eaff);
            color: #fff;
            box-shadow: 0 0 24px #00eaffcc;
        }
        #send-button:pressed {
            background: #232946;
            color: #00eaff;
        }
        #control-button {
            background: rgba(36,40,60,0.8);
            color: #e0e6f7;
            border: 1.5px solid #7f5af0;
            border-radius: 10px;
            padding: 10px;
            margin: 4px;
            font-size: 13px;
            font-weight: 500;
            letter-spacing: 1px;
            box-shadow: 0 0 8px #7f5af044;
            transition: border 0.2s, box-shadow 0.2s;
            font-family: 'Roboto', 'Segoe UI', Arial, sans-serif;
        }
        #control-button:hover {
            border: 2px solid #00eaff;
            color: #00eaff;
            box-shadow: 0 0 16px #00eaff99;
        }
        #section-header {
            font-weight: bold;
            color: #7f5af0;
            padding: 8px;
            margin-top: 16px;
            font-size: 15px;
            letter-spacing: 1px;
            font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
        }
        #history-list {
            background: rgba(24,28,43,0.7);
            color: #e0e6f7;
            border: 1.5px solid #7f5af0;
            border-radius: 12px;
            font-size: 13px;
            box-shadow: 0 0 12px #7f5af022;
            font-family: 'Roboto', 'Segoe UI', Arial, sans-serif;
        }
        #stats-display {
            background: rgba(24,28,43,0.7);
            color: #e0e6f7;
            border: 1.5px solid #00eaff;
            border-radius: 12px;
            font-size: 12px;
            box-shadow: 0 0 12px #00eaff22;
            font-family: 'Roboto', 'Segoe UI', Arial, sans-serif;
        }
        #progress-bar {
            background: rgba(36,40,60,0.8);
            border: 1.5px solid #00eaff;
            border-radius: 8px;
            height: 12px;
        }
        #progress-bar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00eaff, stop:1 #7f5af0);
            border-radius: 8px;
            box-shadow: 0 0 8px #00eaff99;
        }
        QMenuBar {
            background: rgba(24,28,43,0.7);
            color: #e0e6f7;
            border-bottom: 1.5px solid #7f5af0;
            font-size: 15px;
            font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
        }
        QMenuBar::item:selected {
            background: #00eaff;
            color: #232946;
        }
        QMenu {
            background: rgba(24,28,43,0.95);
            color: #e0e6f7;
            border: 1.5px solid #7f5af0;
            font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
        }
        QMenu::item:selected {
            background: #00eaff;
            color: #232946;
        }
        QStatusBar {
            background: rgba(24,28,43,0.7);
            color: #e0e6f7;
            border-top: 1.5px solid #00eaff;
            font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
        }
        /* Chat bubble user */
        QTextEdit, QListWidget, QLineEdit {
            selection-background-color: #00eaff44;
        }
        """
        self.setStyleSheet(style)
    
    def initialize_chatbot(self):
        """Inisialisasi chatbot"""
        try:
            self.chatbot = AdvancedChatbot(self.config, self.gui_callback)
            self.add_system_message("👹 Jambrong siap melayani! Silakan mulai chat...")
            self.refresh_stats()
            self.load_conversation_history()
        except Exception as e:
            self.add_system_message(f"❌ Error inisialisasi: {str(e)}")
    
    def gui_callback(self, status, message=""):
        """Callback untuk update GUI dari chatbot"""
        if status == "start":
            self.show_loading(True)
        elif status == "end":
            self.show_loading(False)
        elif status == "error":
            self.show_loading(False)
            self.add_system_message(f"❌ Error: {message}")
    
    def show_loading(self, show: bool):
        """Tampilkan/sembunyikan loading indicator"""
        self.progress_bar.setVisible(show)
        if show:
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.send_button.setEnabled(not show)
        self.input_field.setEnabled(not show)
    
    def send_message(self):
        """Kirim pesan ke chatbot"""
        user_input = self.input_field.text().strip()
        if not user_input:
            return
        # Tampilkan pesan user
        self.add_user_message(user_input)
        self.input_field.clear()
        # Handle perintah khusus
        if user_input.lower() == "clear":
            self.clear_memory()
            return
        elif user_input.lower() == "stats":
            self.refresh_stats()
            return
        # Jalankan chat di thread terpisah
        if self.chatbot:
            self.chat_thread = ChatThread(self.chatbot, user_input)
            # Tentukan mode streaming
            self._is_streaming_mode = hasattr(self.chatbot.config, 'enable_streaming') and self.chatbot.config.enable_streaming
            if self._is_streaming_mode:
                self._streaming_buffer = ""
                self.chat_display.append(self._streaming_ai_message_header())
                self.chat_thread.streaming_token.connect(self._append_streaming_token)
            self.chat_thread.response_received.connect(self._handle_response_received)
            self.chat_thread.error_occurred.connect(self.handle_error)
            self.chat_thread.processing_started.connect(lambda: self.show_loading(True))
            self.chat_thread.processing_finished.connect(lambda: self.show_loading(False))
            self.chat_thread.start()
    
    def _handle_response_received(self, message: str):
        # Jika streaming, jangan tampilkan lagi (sudah tampil per token)
        if hasattr(self, '_is_streaming_mode') and self._is_streaming_mode:
            # Optionally, update stats/history
            self.refresh_stats()
            self.load_conversation_history()
            # Jika ingin, update tampilan terakhir (misal: bold, dsb)
            return
        # Non-streaming: tampilkan seperti biasa
        self.add_ai_message(message)
    
    def add_user_message(self, message: str):
        """Tambahkan pesan user ke chat display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"""
        <div style='margin: 10px 0; padding: 10px; background-color: #0d47a1; border-radius: 10px; margin-left: 50px;'>
            <b style='color: #4fc3f7;'>🧑‍💻 Kamu [{timestamp}]:</b><br>
            <span style='color: #ffffff;'>{message}</span>
        </div>
        """
        self.chat_display.append(formatted_message)
        self.scroll_to_bottom()
    
    def add_ai_message(self, message: str):
        """Tambahkan pesan AI ke chat display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"""
        <div style='margin: 10px 0; padding: 10px; background-color: #1b5e20; border-radius: 10px; margin-right: 50px;'>
            <b style='color: #4fc3f7;'>🤖 Jambrong [{timestamp}]:</b><br>
            <span style='color: #ffffff;'>{message}</span>
        </div>
        """
        self.chat_display.append(formatted_message)
        self.scroll_to_bottom()
        self.refresh_stats()
        self.load_conversation_history()
    
    def add_system_message(self, message: str):
        """Tambahkan pesan sistem ke chat display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"""
        <div style='margin: 5px 0; padding: 8px; background-color: #424242; border-radius: 8px; text-align: center;'>
            <i style='color: #ffab00;'>[{timestamp}] {message}</i>
        </div>
        """
        self.chat_display.append(formatted_message)
        self.scroll_to_bottom()
    
    def scroll_to_bottom(self):
        """Scroll chat display ke bawah"""
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def handle_error(self, error_message: str):
        """Handle error dari chat thread"""
        self.add_system_message(f"❌ Error: {error_message}")
    
    def clear_memory(self):
        """Hapus memory chatbot"""
        if self.chatbot:
            result = self.chatbot.clear_memory()
            self.add_system_message(result)
            self.refresh_stats()
    
    def refresh_stats(self):
        """Refresh statistik percakapan"""
        if self.chatbot:
            stats = self.chatbot.get_conversation_stats()
            stats_text = ""
            for key, value in stats.items():
                stats_text += f"• {key.replace('_', ' ').title()}: {value}\n"
            self.stats_display.setText(stats_text)
    
    def load_conversation_history(self):
        """Load riwayat percakapan ke sidebar"""
        self.history_list.clear()
        if self.chatbot:
            conversations = self.chatbot.conversation_manager.get_recent_conversations(10)
            for i, conv in enumerate(reversed(conversations)):
                timestamp = datetime.fromisoformat(conv['timestamp']).strftime("%d/%m %H:%M")
                preview = conv['user_input'][:30] + "..." if len(conv['user_input']) > 30 else conv['user_input']
                item_text = f"[{timestamp}] {preview}"
                
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, conv)
                self.history_list.addItem(item)
    
    def load_conversation(self, item):
        """Load percakapan yang dipilih"""
        conv_data = item.data(Qt.UserRole)
        if conv_data:
            # Tampilkan percakapan di chat display
            self.chat_display.clear()
            self.add_system_message(f"📖 Memuat percakapan dari {conv_data['timestamp']}")
            self.add_user_message(conv_data['user_input'])
            self.add_ai_message(conv_data['ai_response'])
    
    def open_config(self):
        """Buka dialog konfigurasi"""
        dialog = ConfigDialog(self.config, self)
        if dialog.exec_() == QDialog.Accepted:
            new_config = dialog.get_config()
            self.config = new_config
            # Reinitialize chatbot dengan config baru
            try:
                self.chatbot = AdvancedChatbot(self.config, self.gui_callback)
                self.add_system_message("✅ Konfigurasi berhasil diperbarui! (Persona: {} | Prompt custom: {})".format(
                    self.config.persona, "Ya" if self.config.custom_prompt else "Tidak"))
                self.refresh_stats()
            except Exception as e:
                self.add_system_message(f"❌ Error mengupdate konfigurasi: {str(e)}")
    
    def export_conversations(self):
        """Export percakapan ke file JSON"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Percakapan",
                f"jambrong_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "JSON Files (*.json)"
            )
            
            if filename and self.chatbot:
                conversations = self.chatbot.conversation_manager.conversations
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(conversations, f, ensure_ascii=False, indent=2)
                
                self.add_system_message(f"✅ Percakapan berhasil diekspor ke {filename}")
                
        except Exception as e:
            self.add_system_message(f"❌ Error export: {str(e)}")
    
    def import_conversations(self):
        """Import percakapan dari file JSON"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Import Percakapan",
                "",
                "JSON Files (*.json)"
            )
            
            if filename and self.chatbot:
                with open(filename, 'r', encoding='utf-8') as f:
                    imported_conversations = json.load(f)
                
                # Tambahkan ke conversations yang ada
                self.chatbot.conversation_manager.conversations.extend(imported_conversations)
                
                # Simpan ke file
                with open(self.chatbot.conversation_manager.conversation_file, 'w', encoding='utf-8') as f:
                    json.dump(self.chatbot.conversation_manager.conversations, f, ensure_ascii=False, indent=2)
                
                self.add_system_message(f"✅ {len(imported_conversations)} percakapan berhasil diimpor")
                self.refresh_stats()
                self.load_conversation_history()
                
        except Exception as e:
            self.add_system_message(f"❌ Error import: {str(e)}")
    
    def show_about(self):
        """Tampilkan dialog about"""
        about_text = """
        <h2>👹 Mbah Jambrong AI Assistant</h2>
        <p><b>Versi:</b> 2.0.0</p>
        <p><b>Developer:</b> Resta Stefano</p>
        <p><b>Framework:</b> PyQt5 + LangChain + Ollama</p>
        
        <h3>Fitur:</h3>
        <ul>
            <li>Chat AI dengan memory konteks</li>
            <li>Interface GUI yang elegan</li>
            <li>Streaming response real-time</li>
            <li>Penyimpanan riwayat percakapan</li>
            <li>Konfigurasi model yang fleksibel</li>
            <li>Export/Import percakapan</li>
            <li>Monitoring dan logging</li>
        </ul>
        
        <p><i>Jambrong siap membantu Anda dengan segala pertanyaan!</i></p>
        """
        
        QMessageBox.about(self, "Tentang Jambrong", about_text)
    
    def update_status(self):
        """Update status bar"""
        if self.chatbot:
            stats = self.chatbot.get_conversation_stats()
            total_conversations = stats.get('total_conversations', 0)
            model = stats.get('model_used', 'Unknown')
            
            status_text = f"Model: {model} | Total Percakapan: {total_conversations} | {datetime.now().strftime('%H:%M:%S')}"
            self.status_bar.showMessage(status_text)
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Hentikan thread jika masih berjalan
        if self.chat_thread and self.chat_thread.isRunning():
            self.chat_thread.quit()
            self.chat_thread.wait()
        
        # Hentikan timer
        if self.status_timer:
            self.status_timer.stop()
        
        self.add_system_message("👋 Sampai jumpa! Terima kasih telah menggunakan Jambrong.")
        event.accept()

    def _streaming_ai_message_header(self):
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"""
        <div id='streaming-ai-msg' style='margin: 10px 0; padding: 10px; background-color: #1b5e20; border-radius: 10px; margin-right: 50px;'>
            <b style='color: #4fc3f7;'>🤖 Jambrong [{timestamp}]:</b><br>
            <span style='color: #ffffff;' id='streaming-ai-content'></span>
        </div>
        """

    def _append_streaming_token(self, token):
        # Tambahkan token ke buffer dan update chat_display
        if not hasattr(self, '_streaming_buffer'):
            self._streaming_buffer = ""
        self._streaming_buffer += token
        # Update pesan terakhir di QTextEdit
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)
        # Hapus pesan terakhir dan tambahkan ulang dengan buffer
        # (QTextEdit tidak support update by id, jadi kita clear dan append ulang)
        # Alternatif: gunakan appendPlainText jika ingin lebih sederhana
        # Untuk sekarang, tambahkan ulang seluruh buffer
        # (atau bisa pakai JavaScript bridge untuk update by id jika mau advance)
        # Sederhana:
        self.chat_display.undo()  # Undo header
        self.chat_display.append(self._streaming_ai_message_header().replace("<span style='color: #ffffff;' id='streaming-ai-content'></span>", f"<span style='color: #ffffff;' id='streaming-ai-content'>{self._streaming_buffer}</span>"))

    def clear_chat(self):
        """Bersihkan tampilan chat (bukan memory/riwayat)"""
        self.chat_display.clear()
        self.add_system_message("✨ Chat telah dibersihkan.")

class SplashScreen(QWidget):
    """Splash screen untuk loading"""
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(400, 300)
        
        # Center the splash screen
        screen = QApplication.desktop().screenGeometry()
        self.move((screen.width() - self.width()) // 2, (screen.height() - self.height()) // 2)
        
        self.setup_ui()
        
        # Timer untuk auto close
        self.timer = QTimer()
        self.timer.timeout.connect(self.close)
        self.timer.start(3000)  # 3 detik
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Background frame
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 2px solid #4fc3f7;
                border-radius: 20px;
            }
        """)
        
        frame_layout = QVBoxLayout(frame)
        
        # Logo/Title
        title = QLabel("👹")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 48px; margin: 20px;")
        frame_layout.addWidget(title)
        
        # App name
        app_name = QLabel("Mbah Jambrong")
        app_name.setAlignment(Qt.AlignCenter)
        app_name.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #4fc3f7;
            margin: 10px;
        """)
        frame_layout.addWidget(app_name)
        
        # Subtitle
        subtitle = QLabel("AI Assistant yang Unyu")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 14px;
            color: #ffffff;
            margin: 5px;
        """)
        frame_layout.addWidget(subtitle)
        
        # Loading indicator
        loading = QLabel("Memuat...")
        loading.setAlignment(Qt.AlignCenter)
        loading.setStyleSheet("""
            font-size: 12px;
            color: #888888;
            margin: 20px;
        """)
        frame_layout.addWidget(loading)
        
        # Progress bar
        progress = QProgressBar()
        progress.setRange(0, 0)  # Indeterminate
        progress.setStyleSheet("""
            QProgressBar {
                background-color: #2d2d2d;
                border: 1px solid #4fc3f7;
                border-radius: 8px;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #4fc3f7;
                border-radius: 8px;
            }
        """)
        frame_layout.addWidget(progress)
        
        layout.addWidget(frame)

def main():
    """Fungsi utama untuk menjalankan aplikasi GUI"""
    app = QApplication(sys.argv)
    app.setApplicationName("Mbah Jambrong AI")
    app.setApplicationVersion("2.0.0")
    # Set aplikasi style
    app.setStyle('Fusion')
    # Modern font setup
    modern_font = QFont("Montserrat", 11)
    app.setFont(modern_font)
    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
    palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    # Tambahkan stylesheet font modern
    app.setStyleSheet("""
    QMainWindow, QWidget, QDialog, QMenuBar, QMenu, QStatusBar {
        font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
    }
    QLabel#chat-header, QLabel#section-header {
        font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
        font-weight: bold;
    }
    QTextEdit#chat-display, QTextEdit#stats-display, QListWidget#history-list {
        font-family: 'Roboto', 'Segoe UI', Arial, sans-serif;
    }
    QLineEdit#input-field, QPushButton#send-button, QPushButton#control-button {
        font-family: 'Roboto', 'Segoe UI', Arial, sans-serif;
    }
    QTextEdit, QPlainTextEdit {
        font-family: 'Roboto Mono', 'Consolas', monospace;
    }
    """)
    try:
        # Tampilkan splash screen
        splash = SplashScreen()
        splash.show()
        # Process events untuk menampilkan splash
        app.processEvents()
        # Buat main window
        window = JambrongMainWindow()
        # Tunggu splash selesai
        splash.timer.timeout.connect(lambda: (splash.close(), window.show()))
        # Jalankan aplikasi
        sys.exit(app.exec_())
    except Exception as e:
        # Fallback error handling
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Critical)
        error_msg.setWindowTitle("Error")
        error_msg.setText(f"Terjadi kesalahan saat menjalankan aplikasi:\n\n{str(e)}")
        error_msg.setDetailedText(f"Error detail:\n{str(e)}")
        error_msg.exec_()
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
