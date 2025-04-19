import streamlit as st

# Set page config as the first Streamlit command
st.set_page_config(page_title="NoteBot", page_icon="📝", layout="wide")

import whisper
import google.generativeai as genai
import os
from dotenv import load_dotenv
import tempfile
import json
from datetime import datetime
import traceback
import pandas as pd
from pathlib import Path

# Load environment variables
load_dotenv()

# Tạo thư mục data nếu chưa tồn tại
DATA_DIR = "data"
NOTES_FILE = os.path.join(DATA_DIR, "notes.csv")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

# Debug log function
def debug_log(message):
    print(f"[DEBUG] {datetime.now()}: {message}")
    st.write(f"[DEBUG] {message}")

# Tạo các thư mục cần thiết
debug_log(f"Tạo thư mục data: {DATA_DIR}")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Tạo file notes.csv nếu chưa tồn tại với header
if not os.path.exists(NOTES_FILE):
    debug_log(f"Tạo file CSV mới: {NOTES_FILE}")
    df = pd.DataFrame(columns=['subject', 'title', 'content', 'summary', 'date', 'audio_file'])
    df.to_csv(NOTES_FILE, index=False, encoding='utf-8-sig')
    debug_log("Đã tạo file CSV với header")

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Định nghĩa cấu trúc tóm tắt cho từng môn học
SUBJECT_TEMPLATES = {
    "Toán học": """
    1. KHÁI NIỆM & ĐỊNH LÝ:
    - Các định nghĩa và khái niệm mới
    - Các định lý và công thức quan trọng
    - Điều kiện áp dụng

    2. PHƯƠNG PHÁP & KỸ THUẬT:
    - Các phương pháp giải chính
    - Kỹ thuật tính toán
    - Các bước giải quan trọng

    3. VÍ DỤ & BÀI TẬP MẪU:
    - Phân tích các ví dụ tiêu biểu
    - Các dạng bài tập điển hình

    4. GHI CHÚ HỌC TẬP:
    - Các lỗi thường gặp cần tránh
    - Mẹo và thủ thuật giải nhanh
    - Liên hệ với các chủ đề khác
    """,
    
    "Vật lý": """
    1. NGUYÊN LÝ & ĐỊNH LUẬT:
    - Các định luật vật lý mới
    - Nguyên lý hoạt động
    - Các công thức quan trọng

    2. HIỆN TƯỢNG & ỨNG DỤNG:
    - Giải thích hiện tượng
    - Ứng dụng thực tế
    - Thí nghiệm liên quan

    3. PHÂN TÍCH ĐỊNH LƯỢNG:
    - Các đại lượng và đơn vị
    - Quan hệ giữa các đại lượng
    - Phương pháp giải bài tập

    4. GHI CHÚ HỌC TẬP:
    - Các điểm cần lưu ý
    - Liên hệ với các chương khác
    - Câu hỏi ôn tập quan trọng
    """,
    
    "Hóa học": """
    1. KHÁI NIỆM & PHẢN ỨNG:
    - Định nghĩa và khái niệm mới
    - Các phản ứng hóa học chính
    - Điều kiện phản ứng

    2. CƠ CHẾ & QUY LUẬT:
    - Cơ chế phản ứng
    - Các quy luật quan trọng
    - Yếu tố ảnh hưởng

    3. THỰC HÀNH & ỨNG DỤNG:
    - Phương pháp thí nghiệm
    - Ứng dụng trong thực tế
    - Các bài toán thực tế

    4. GHI CHÚ HỌC TẬP:
    - Các công thức cần nhớ
    - Phương pháp giải bài tập
    - Lưu ý an toàn thí nghiệm
    """,
    
    "Sinh học": """
    1. CẤU TRÚC & CHỨC NĂNG:
    - Cấu tạo và đặc điểm
    - Chức năng và vai trò
    - Mối quan hệ cấu trúc-chức năng

    2. QUÁ TRÌNH & CƠ CHẾ:
    - Các quá trình sinh học
    - Cơ chế hoạt động
    - Các yếu tố ảnh hưởng

    3. PHÂN LOẠI & ĐẶC ĐIỂM:
    - Tiêu chí phân loại
    - Đặc điểm nhận dạng
    - So sánh và phân biệt

    4. GHI CHÚ HỌC TẬP:
    - Thuật ngữ chuyên ngành
    - Sơ đồ và hình vẽ quan trọng
    - Câu hỏi trọng tâm
    """,
    
    "Văn học": """
    1. TÁC PHẨM & TÁC GIẢ:
    - Thông tin về tác giả
    - Hoàn cảnh sáng tác
    - Ý nghĩa tác phẩm

    2. PHÂN TÍCH & ĐÁNH GIÁ:
    - Nội dung chính
    - Nghệ thuật đặc sắc
    - Ý nghĩa văn học - xã hội

    3. CHỦ ĐỀ & TƯ TƯỞNG:
    - Chủ đề chính
    - Tư tưởng nổi bật
    - Giá trị nhân văn

    4. GHI CHÚ HỌC TẬP:
    - Dàn ý phân tích
    - Các dẫn chứng tiêu biểu
    - Câu hỏi thảo luận
    """,
    
    "Lịch sử": """
    1. SỰ KIỆN & NHÂN VẬT:
    - Thời gian và địa điểm
    - Nhân vật lịch sử
    - Diễn biến chính

    2. NGUYÊN NHÂN & HỆ QUẢ:
    - Bối cảnh lịch sử
    - Nguyên nhân sự kiện
    - Kết quả và tác động

    3. Ý NGHĨA & ĐÁNH GIÁ:
    - Ý nghĩa lịch sử
    - Bài học kinh nghiệm
    - Đánh giá khách quan

    4. GHI CHÚ HỌC TẬP:
    - Mốc thời gian quan trọng
    - Sơ đồ diễn biến
    - Câu hỏi ôn tập
    """,
    
    "Địa lý": """
    1. ĐẶC ĐIỂM & PHÂN BỐ:
    - Vị trí địa lý
    - Đặc điểm tự nhiên
    - Phân bố không gian

    2. MỐI QUAN HỆ & TÁC ĐỘNG:
    - Quan hệ nhân-quả
    - Tác động qua lại
    - Ảnh hưởng đến đời sống

    3. THỰC TRẠNG & XU HƯỚNG:
    - Hiện trạng phát triển
    - Xu hướng biến đổi
    - Dự báo tương lai

    4. GHI CHÚ HỌC TẬP:
    - Số liệu quan trọng
    - Bản đồ và biểu đồ
    - Các vấn đề thực tế
    """,
    
    "Khác": """
    1. KHÁI NIỆM CHÍNH:
    - Định nghĩa và thuật ngữ
    - Phạm vi áp dụng
    - Ý nghĩa quan trọng

    2. NỘI DUNG TRỌNG TÂM:
    - Các điểm chính
    - Mối liên hệ
    - Ứng dụng thực tế

    3. PHÂN TÍCH & ĐÁNH GIÁ:
    - Ưu điểm và hạn chế
    - So sánh và phân biệt
    - Nhận xét tổng hợp

    4. GHI CHÚ HỌC TẬP:
    - Các điểm cần nhớ
    - Câu hỏi ôn tập
    - Hướng nghiên cứu thêm
    """
}

# Initialize Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("medium")

def transcribe_audio(audio_file):
    """Transcribe audio file using Whisper"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name

        model = load_whisper_model()
        result = model.transcribe(tmp_file_path, language="vi")
        
        os.unlink(tmp_file_path)
        
        return result["text"]
    except Exception as e:
        st.error(f"Lỗi khi chuyển đổi âm thanh: {str(e)}")
        return None

def summarize_text(text, subject):
    """Summarize text using Google's Gemini model with subject-specific template"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        template = SUBJECT_TEMPLATES.get(subject, SUBJECT_TEMPLATES["Khác"])
        
        prompt = f"""Với tư cách là một trợ lý học tập chuyên môn về {subject}, 
        hãy phân tích và tóm tắt nội dung sau đây theo cấu trúc dành cho môn {subject}:

        NỘI DUNG:
        {text}

        Hãy tổ chức bản tóm tắt theo cấu trúc sau:
        {template}

        Hãy trình bày rõ ràng, súc tích và dễ hiểu bằng tiếng Việt."""

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Lỗi khi tạo tóm tắt: {str(e)}")
        return None

def generate_title(text, subject):
    """Generate a title from the content"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""Dựa vào nội dung bài giảng sau đây, hãy tạo một tiêu đề ngắn gọn (tối đa 10 từ) phản ánh chủ đề chính của bài:

        {text[:500]}...  # Chỉ lấy 500 ký tự đầu để tạo tiêu đề

        Lưu ý:
        - Tiêu đề phải ngắn gọn, súc tích
        - Không cần ghi "Bài giảng về" hoặc các từ mở đầu tương tự
        - Chỉ trả về tiêu đề, không thêm giải thích"""

        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        # Nếu không tạo được tiêu đề, dùng thời gian làm tiêu đề
        return f"Bài ghi {datetime.now().strftime('%d/%m/%Y %H:%M')}"

def save_audio_file(audio_file):
    """Save uploaded audio file"""
    try:
        file_extension = os.path.splitext(audio_file.name)[1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}{file_extension}"
        filepath = os.path.join(AUDIO_DIR, filename)
        
        with open(filepath, "wb") as f:
            f.write(audio_file.getvalue())
        
        st.success(f"Đã lưu file âm thanh: {filepath}")
        return filepath
    except Exception as e:
        st.error(f"Lỗi khi lưu file âm thanh: {str(e)}")
        st.error(traceback.format_exc())
        return None

def save_note(subject, content, summary, audio_path=None):
    """Save note to CSV file"""
    try:
        debug_log("Bắt đầu lưu ghi chú mới")
        
        # Tạo tiêu đề tự động từ nội dung
        title = generate_title(content, subject)
        debug_log(f"Đã tạo tiêu đề: {title}")
        
        # Tạo ghi chú mới
        new_note = {
            'subject': subject,
            'title': title,
            'content': content,
            'summary': summary,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'audio_file': audio_path
        }
        debug_log("Đã tạo dictionary ghi chú mới")
        
        # Tạo DataFrame mới với một dòng
        new_df = pd.DataFrame([new_note])
        debug_log("Đã tạo DataFrame mới")
        
        # Đọc CSV hiện có hoặc tạo mới nếu không tồn tại
        try:
            if os.path.exists(NOTES_FILE):
                debug_log("Đọc file CSV hiện có")
                df = pd.read_csv(NOTES_FILE, encoding='utf-8-sig')
                debug_log(f"Số ghi chú hiện có: {len(df)}")
            else:
                debug_log("Tạo DataFrame mới vì file không tồn tại")
                df = pd.DataFrame(columns=['subject', 'title', 'content', 'summary', 'date', 'audio_file'])
        except Exception as e:
            debug_log(f"Lỗi khi đọc CSV: {str(e)}")
            df = pd.DataFrame(columns=['subject', 'title', 'content', 'summary', 'date', 'audio_file'])
        
        # Thêm ghi chú mới
        df = pd.concat([df, new_df], ignore_index=True)
        debug_log(f"Đã thêm ghi chú mới, tổng số ghi chú: {len(df)}")
        
        # Lưu lại file CSV
        debug_log(f"Lưu file CSV: {NOTES_FILE}")
        df.to_csv(NOTES_FILE, index=False, encoding='utf-8-sig')
        
        # Tạo file text riêng cho ghi chú
        note_dir = os.path.join(DATA_DIR, "notes")
        os.makedirs(note_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        note_file = os.path.join(note_dir, f"note_{timestamp}.txt")
        
        debug_log(f"Lưu file text: {note_file}")
        with open(note_file, "w", encoding="utf-8") as f:
            f.write(f"""TIÊU ĐỀ: {title}
MÔN HỌC: {subject}
THỜI GIAN: {new_note['date']}

TÓM TẮT:
{summary}

NỘI DUNG ĐẦY ĐỦ:
{content}
""")
        
        # Kiểm tra xem file đã được lưu thành công chưa
        if os.path.exists(NOTES_FILE):
            with open(NOTES_FILE, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                debug_log(f"Nội dung file CSV sau khi lưu: {content[:200]}...")
        
        st.success(f"""
        ✅ Đã lưu ghi chú thành công!
        - Tiêu đề: {title}
        - Môn học: {subject}
        - Thời gian: {new_note['date']}
        - File CSV: {os.path.abspath(NOTES_FILE)}
        - File ghi chú: {note_file}
        {f'- File âm thanh: {audio_path}' if audio_path else ''}
        """)
        
        return True
    except Exception as e:
        debug_log(f"❌ Lỗi khi lưu ghi chú: {str(e)}")
        st.error("❌ Lỗi khi lưu ghi chú!")
        st.error(f"Chi tiết lỗi: {str(e)}")
        st.error(traceback.format_exc())
        return False

def load_notes():
    """Load saved notes"""
    try:
        debug_log("Bắt đầu đọc ghi chú")
        if os.path.exists(NOTES_FILE):
            debug_log(f"Đọc file CSV: {NOTES_FILE}")
            df = pd.read_csv(NOTES_FILE, encoding='utf-8-sig')
            notes = df.to_dict('records')
            debug_log(f"Đã đọc được {len(notes)} ghi chú")
            return notes
    except Exception as e:
        debug_log(f"Lỗi khi đọc ghi chú: {str(e)}")
        st.error(f"Lỗi khi đọc ghi chú: {str(e)}")
        st.error(traceback.format_exc())
    return []

def get_storage_info():
    """Get storage information"""
    try:
        # Đọc CSV file
        if os.path.exists(NOTES_FILE):
            df = pd.read_csv(NOTES_FILE, encoding='utf-8-sig')
            num_notes = len(df)
        else:
            num_notes = 0
        
        # Đếm file âm thanh
        audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(('.mp3', '.wav'))]
        num_audio = len(audio_files)
        
        # Đếm file text
        note_dir = os.path.join(DATA_DIR, "notes")
        if os.path.exists(note_dir):
            text_files = [f for f in os.listdir(note_dir) if f.endswith('.txt')]
            num_text = len(text_files)
        else:
            num_text = 0
        
        # Tính tổng dung lượng
        total_size = 0
        for f in audio_files:
            total_size += os.path.getsize(os.path.join(AUDIO_DIR, f))
            
        return {
            "num_notes": num_notes,
            "num_audio": num_audio,
            "num_text": num_text,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }
    except Exception:
        return {"num_notes": 0, "num_audio": 0, "num_text": 0, "total_size_mb": 0}

def correct_text(text):
    """Sửa lỗi và cải thiện chất lượng văn bản từ speech-to-text"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""Hãy sửa lỗi chính tả và cải thiện chất lượng văn bản sau đây, giữ nguyên ý nghĩa nhưng làm cho văn bản mạch lạc và dễ hiểu hơn:

        Văn bản gốc:
        {text}

        Yêu cầu:
        1. Sửa lỗi chính tả và ngữ pháp
        2. Thêm dấu câu phù hợp
        3. Điều chỉnh các từ ngữ không rõ ràng
        4. Giữ nguyên thuật ngữ chuyên môn
        5. Không thay đổi ý nghĩa của văn bản

        Chỉ trả về văn bản đã sửa, không cần giải thích."""

        response = model.generate_content(prompt)
        corrected_text = response.text.strip()
        debug_log("Đã sửa lỗi và cải thiện chất lượng văn bản")
        return corrected_text
    except Exception as e:
        debug_log(f"Lỗi khi sửa văn bản: {str(e)}")
        return text

def main():
    st.title("📝 NoteBot - Trợ lý Học tập")
    st.subheader("Chuyển đổi ghi chú âm thanh thành văn bản và tạo tóm tắt học thuật")

    # Tạo hai cột chính
    col1, col2 = st.columns([2, 1])

    with col1:
        # Chọn môn học
        subject = st.selectbox(
            "Chọn môn học",
            list(SUBJECT_TEMPLATES.keys())
        )

        # File upload
        audio_file = st.file_uploader("Tải lên file ghi âm bài giảng của bạn", type=["mp3", "wav"])

        if audio_file:
            st.audio(audio_file)
            
            if st.button("Chuyển đổi và Tạo Tóm tắt Học thuật"):
                # Lưu file âm thanh
                audio_path = save_audio_file(audio_file)
                
                with st.spinner("Đang chuyển đổi âm thanh thành văn bản..."):
                    transcription = transcribe_audio(audio_file)
                    
                if transcription:
                    # Hiển thị văn bản gốc
                    st.subheader("📄 Văn bản Gốc")
                    st.write(transcription)
                    
                    # Sửa lỗi và cải thiện chất lượng văn bản
                    with st.spinner("Đang sửa lỗi và cải thiện chất lượng văn bản..."):
                        corrected_text = correct_text(transcription)
                    
                    # Hiển thị văn bản đã sửa
                    st.subheader("📝 Văn bản Đã Sửa")
                    st.write(corrected_text)
                    
                    # So sánh sự khác biệt
                    if corrected_text != transcription:
                        st.info("ℹ️ Văn bản đã được sửa và cải thiện chất lượng.")
                    
                    with st.spinner("Đang tạo tóm tắt học thuật..."):
                        summary = summarize_text(corrected_text, subject)
                    
                    if summary:
                        st.subheader(f"✨ Tóm tắt Học thuật - {subject}")
                        st.write(summary)
                        
                        # Lưu ghi chú
                        debug_log("Hiển thị nút lưu ghi chú")
                        if st.button("Lưu ghi chú này"):
                            debug_log("Nút lưu ghi chú được nhấn")
                            save_success = save_note(subject, corrected_text, summary, audio_path)
                            
                            if save_success:
                                st.success("✅ Đã lưu ghi chú thành công!")
                            else:
                                st.error("❌ Lưu ghi chú thất bại!")

    with col2:
        st.header("📚 Ghi chú đã lưu")
        
        # Hiển thị thông tin lưu trữ
        storage_info = get_storage_info()
        st.info(f"""
        📊 **Thông tin lưu trữ:**
        - Số lượng ghi chú: {storage_info['num_notes']}
        - Số file âm thanh: {storage_info['num_audio']}
        - Số file text: {storage_info['num_text']}
        - Dung lượng: {storage_info['total_size_mb']} MB
        
        📂 **Vị trí lưu trữ:**
        - Ghi chú: `{os.path.abspath(NOTES_FILE)}`
        - File âm thanh: `{os.path.abspath(AUDIO_DIR)}`
        - File text: `{os.path.abspath(os.path.join(DATA_DIR, "notes"))}`
        """)
        
        notes = load_notes()
        if notes:
            st.success(f"Đã tải {len(notes)} ghi chú")
            
            # Lọc theo môn học
            filter_subject = st.selectbox("Lọc theo môn học", ["Tất cả"] + list(SUBJECT_TEMPLATES.keys()))
            
            filtered_notes = notes
            if filter_subject != "Tất cả":
                filtered_notes = [note for note in notes if note["subject"] == filter_subject]
            
            for note in filtered_notes:
                with st.expander(f"{note['title']} ({note['subject']}) - {note['date']}"):
                    st.write("**Tóm tắt:**")
                    st.write(note["summary"])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Xem nội dung đầy đủ", key=f"content_{note['date']}"):
                            st.write("**Nội dung đầy đủ:**")
                            st.write(note["content"])
                    
                    with col2:
                        if note.get("audio_file") and os.path.exists(note["audio_file"]):
                            with open(note["audio_file"], "rb") as audio_file:
                                st.audio(audio_file.read(), format="audio/wav")
        else:
            st.info("Chưa có ghi chú nào được lưu.")

    # Sidebar với hướng dẫn
    with st.sidebar:
        st.header("Hướng dẫn Sử dụng")
        st.write("""
        **Các bước sử dụng:**
        1. Chọn môn học phù hợp
        2. Tải lên file ghi âm bài giảng
        3. Nhấn 'Chuyển đổi và Tạo Tóm tắt Học thuật'
        4. Lưu ghi chú để xem lại sau
        
        **Tính năng nâng cao:**
        - Tự động tạo tiêu đề từ nội dung
        - Tóm tắt được tùy chỉnh theo từng môn học
        - Lưu trữ và quản lý ghi chú theo môn học
        - Tìm kiếm và lọc ghi chú dễ dàng
        - Lưu trữ cả file âm thanh gốc
        """)
        
        st.header("Lưu ý")
        st.write("""
        - Nên ghi âm ở môi trường yên tĩnh
        - Có thể tải lên file MP3 hoặc WAV
        - Mỗi môn học có cấu trúc tóm tắt riêng
        - Tiêu đề được tự động tạo từ nội dung bài giảng
        - File được lưu trong thư mục data của ứng dụng
        """)

if __name__ == "__main__":
    main() 