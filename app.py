import os
import json
import random
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from langchain.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from openai import AzureOpenAI

load_dotenv()

# Mock data
destinations_data = [
    "Hà Nội: Thủ đô ngàn năm văn hiến, hồ Hoàn Kiếm, phố cổ, Văn Miếu Quốc Tử Giám.",
    "Hạ Long: Di sản thiên nhiên thế giới, nổi tiếng với hàng ngàn đảo đá vôi và hang động kỳ thú.",
    "Sapa: Thị trấn vùng cao, khí hậu mát mẻ, ruộng bậc thang và đỉnh Fansipan.",
    "Hà Giang: Cao nguyên đá Đồng Văn, đèo Mã Pì Lèng, lễ hội hoa tam giác mạch.",
    "Ninh Bình: Tràng An, Tam Cốc – Bích Động, chùa Bái Đính.",
    "Mộc Châu: Cao nguyên chè, hoa mận, thác Dải Yếm.",
    "Cát Bà: Vườn quốc gia Cát Bà, bãi tắm Cát Cò, làng chài.",
    "Lạng Sơn: Động Tam Thanh, chợ Đông Kinh, núi Mẫu Sơn.",
    "Yên Bái: Hồ Thác Bà, Mù Cang Chải, ruộng bậc thang.",
    "Lào Cai: Bắc Hà, chợ phiên, dinh Hoàng A Tưởng.",
    "Quảng Ninh: Yên Tử, đảo Cô Tô, đảo Quan Lạn.",
    "Nam Định: Nhà thờ Phú Nhai, biển Thịnh Long.",
    "Thái Bình: Biển Đồng Châu, làng vườn Bách Thuận.",
    "Cao Bằng: Thác Bản Giốc, động Ngườm Ngao.",
    "Huế: Cố đô, Đại Nội, lăng tẩm vua Nguyễn.",
    "Đà Nẵng: Bà Nà Hills, cầu Rồng, biển Mỹ Khê.",
    "Hội An: Phố cổ UNESCO, đèn lồng, ẩm thực đặc sắc.",
    "Quảng Bình: Hang Sơn Đoòng, động Phong Nha - Kẻ Bàng.",
    "Nha Trang: Thành phố biển, Tháp Bà Ponagar, đảo Hòn Mun.",
    "Đà Lạt: Thành phố ngàn hoa, hồ Xuân Hương, vườn dâu.",
    "TP.HCM: Chợ Bến Thành, Nhà thờ Đức Bà, phố đi bộ Nguyễn Huệ.",
    "Vũng Tàu: Bãi Sau, bãi Trước, tượng Chúa Kitô Vua.",
    "Cần Thơ: Chợ nổi Cái Răng, miệt vườn trái cây.",
    "Phú Quốc: Bãi Sao, VinWonders, lặn ngắm san hô.",
    "Cà Mau: Mũi Cà Mau, rừng U Minh Hạ.",
    "Bến Tre: Xứ dừa, cồn Phụng, du lịch miệt vườn."
]

mock_docs = destinations_data

message_history = [
    {
        "role": "system",
        "content": """
            Bạn là một hướng dẫn viên du lịch Việt Nam.
            Luôn trả lời thật chi tiết, bao gồm:
            - Giới thiệu tổng quan điểm đến (lịch sử, văn hóa, khí hậu, điểm nổi bật).
            - Các hoạt động gợi ý cho từng ngày kèm mô tả cụ thể (địa điểm, giờ đi, chi phí dự kiến, lưu ý đặc biệt).
            - Gợi ý ăn uống (tên món, nhà hàng/quán nổi tiếng).
            - Mẹo và kinh nghiệm khi đi (thời tiết, phương tiện, vé tham quan).
            Không được trả lời quá ngắn gọn hay chỉ liệt kê.
            Trình bày rõ ràng theo từng mục và ngày.
        """
    }
]

# Mock functions
def get_flight_price(destination: str) -> str:
    price = random.randint(1500000, 5000000)
    return f"Giá vé khứ hồi đến {destination} khoảng {price:,} VND."

def get_hotel_price(destination: str) -> str:
    price = random.randint(500000, 2000000)
    return f"Giá khách sạn ở {destination} từ {price:,} VND/đêm."

def get_itinerary(destination: str, days: int) -> str:
    activities = [
        "Tham quan địa danh nổi tiếng",
        "Thưởng thức đặc sản địa phương",
        "Chụp ảnh check-in",
        "Mua sắm quà lưu niệm",
        "Tham gia hoạt động ngoài trời",
        "Khám phá văn hóa bản địa"
    ]
    plan = []
    for day in range(1, max(1, int(days))+1):
        plan.append(f"Ngày {day}: {random.choice(activities)} tại {destination}")
    return "\n".join(plan)

# Embeddings + FAISS
embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("AZURE_DEPLOYMENT_NAME_EBD3"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_EBD3"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_version="2024-07-01-preview"
)

vectorstore = FAISS.from_texts(mock_docs, embedding=embeddings)

# LangChain Azure Chat (used for RAG conversational retriever)
chat = AzureChatOpenAI(
    deployment_name=os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_GPT4"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-07-01-preview"
)

retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=chat,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# AzureOpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_GPT4"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-07-01-preview",
    azure_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4")
)

# Functions schema
functions = [
    {
        "name": "get_flight_price",
        "description": "Lấy giá vé máy bay khứ hồi đến điểm du lịch",
        "parameters": {
            "type": "object",
            "properties": {"destination": {"type": "string"}},
            "required": ["destination"]
        }
    },
    {
        "name": "get_hotel_price",
        "description": "Lấy giá khách sạn trung bình ở điểm du lịch",
        "parameters": {
            "type": "object",
            "properties": {"destination": {"type": "string"}},
            "required": ["destination"]
        }
    },
    {
        "name": "get_itinerary",
        "description": "Gợi ý lịch trình du lịch",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {"type": "string"},
                "days": {"type": "integer"}
            },
            "required": ["destination", "days"]
        }
    }
]

# Few-shot prompting
few_shots = [
    # Ví dụ 1: hỏi về địa điểm
    {
        "role": "user",
        "content": "Tôi muốn đi du lịch Đà Nẵng 3 ngày, gợi ý giúp tôi lịch trình chi tiết."
    },
    {
        "role": "assistant",
        "content": """
            Đà Nẵng là thành phố biển nổi tiếng miền Trung, có cầu Rồng, biển Mỹ Khê, Bà Nà Hills, ẩm thực phong phú.
            **Lịch trình gợi ý:**
            Ngày 1: Tham quan Bà Nà Hills (vé 900k), buổi chiều dạo biển Mỹ Khê, tối ăn hải sản ở Bé Mặn.
            Ngày 2: Sáng tham quan Ngũ Hành Sơn, chiều đi Hội An (vé vào phố cổ 80k), tối ngắm đèn lồng.
            Ngày 3: Sáng chợ Hàn mua quà, ăn mì Quảng ếch Bà Mua, ra sân bay.
            **Mẹo:** Mang kem chống nắng, đặt vé Bà Nà trước 1-2 ngày, thuê xe máy di chuyển thuận tiện.
        """
    },
    # Ví dụ 2: hỏi giá vé
    {
        "role": "user",
        "content": "Giá vé máy bay khứ hồi đến Hà Nội là bao nhiêu?"
    },
    {
        "role": "assistant",
        "content": "Giá vé khứ hồi đến Hà Nội khoảng 2,500,000 VND, tùy hãng và thời điểm đặt."
    },
    # Ví dụ 3: hỏi khách sạn
    {
        "role": "user",
        "content": "Khách sạn ở Nha Trang giá bao nhiêu?"
    },
    {
        "role": "assistant",
        "content": "Giá khách sạn ở Nha Trang từ 800,000 VND/đêm cho khách sạn 3 sao, cao hơn với resort ven biển."
    }
]

# Flask app
app = Flask(__name__)

# helper: list of destination names from mock_docs (left part before ":")
DEST_NAMES = [d.split(":")[0].strip() for d in mock_docs]

def call_model_with_functions(messages):
    """
        Gọi AzureOpenAI client để model decide function call.
        Trả về message object (choices[0].message)
    """
    resp = client.chat.completions.create(
        model=os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4"),
        messages=messages,
        functions=functions,
        function_call="auto",
        temperature=0.1
    )
    return resp.choices[0].message

def fallback_rag(question, chat_history):
    """Dùng LangChain retrieval_chain để lấy answer nếu mock không đủ"""
    rag = retrieval_chain({"question": question, "chat_history": chat_history})
    # retrieval_chain trả về dict: {'answer': ..., 'source_documents': [...]}
    return rag.get("answer", "").strip(), rag.get("source_documents", [])

def is_tourism_related(question: str) -> bool:
    """
    Kiểm tra xem câu hỏi có liên quan đến du lịch Việt Nam không.
    Trả về True nếu liên quan, False nếu không.
    """
    check_messages = [
        {
            "role": "system",
            "content": "Bạn là bộ lọc phân loại câu hỏi. Trả lời chỉ 'yes' nếu câu hỏi liên quan đến du lịch Việt Nam (địa điểm, lịch trình, ẩm thực, mẹo du lịch). Trả lời 'no' nếu không liên quan."
        },
        {"role": "user", "content": question}
    ]
    try:
        resp = client.chat.completions.create(
            model=os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4"),
            messages=check_messages,
            temperature=0,
            max_tokens=2
        )
        answer = resp.choices[0].message.content.strip().lower()
        return answer.startswith("y")
    except Exception as e:
        print("Error in topic classification:", e)
        return True

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    user_message = data.get("message", "")
    chat_history = data.get("history", [])  # list of pairs [(q, a), ...]

    if not is_tourism_related(user_message):
        reply = "Tôi là chatbot về du lịch, hãy hỏi câu hỏi liên quan đến địa điểm, lịch trình du lịch ở Việt Nam."
        chat_history.append((user_message, reply))
        return jsonify({"reply": reply, "sources": [], "history": chat_history})

    messages = [
        *few_shots,
        *message_history,
        {
            "role": "system",
            "content": """
                Bạn là một hướng dẫn viên du lịch Việt Nam.
                Luôn trả lời thật chi tiết, bao gồm:
                - Giới thiệu tổng quan điểm đến (lịch sử, văn hóa, khí hậu, điểm nổi bật).
                - Các hoạt động gợi ý cho từng ngày kèm mô tả cụ thể (địa điểm, giờ đi, chi phí dự kiến, lưu ý đặc biệt).
                - Gợi ý ăn uống (tên món, nhà hàng/quán nổi tiếng).
                - Mẹo và kinh nghiệm khi đi (thời tiết, phương tiện, vé tham quan).
                Không được trả lời quá ngắn gọn hay chỉ liệt kê.
                Trình bày rõ ràng theo từng mục và ngày.
                Trả lời chỉ bằng văn bản thuần (plain text), không dùng Markdown, không dùng ký hiệu #, -, *, **.
            """
        }
    ]

    for q, a in chat_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_message})

    # call model to decide function or direct answer
    try:
        message = call_model_with_functions(messages)
    except Exception as e:
        return jsonify({"error": f"Model call failed: {e}"}), 500

    # if model wants to call function
    if getattr(message, "function_call", None):
        func_name = message.function_call.name
        # arguments is string JSON
        try:
            args = json.loads(message.function_call.arguments)
        except Exception:
            args = {}

        dest = args.get("destination", "").strip()
        # If destination not in mock list, fallback to RAG
        if dest and dest not in DEST_NAMES:
            rag_answer, sources = fallback_rag(user_message, chat_history)
            reply = rag_answer or f"Xin lỗi, không tìm thấy thông tin về {dest} trong dữ liệu của tôi."
            chat_history.append((user_message, reply))
            return jsonify({"reply": reply, "sources": [s.metadata if hasattr(s, 'metadata') else str(s) for s in sources], "history": chat_history})

        # destination exists in mock -> execute local function
        if func_name == "get_flight_price":
            result = get_flight_price(dest)
        elif func_name == "get_hotel_price":
            result = get_hotel_price(dest)
        elif func_name == "get_itinerary":
            days = args.get("days", 1)
            result = get_itinerary(dest, days)
        else:
            result = "Không tìm thấy chức năng."

        chat_history.append((user_message, result))
        return jsonify({"reply": result, "sources": [], "history": chat_history})

    # else not function_call -> if model returned plain content, use it
    reply = getattr(message, "content", "") or ""

    # if reply seems empty or "I don't know", fallback RAG
    if not reply.strip() or "không biết" in reply.lower():
        rag_answer, sources = fallback_rag(user_message, chat_history)
        reply = rag_answer or reply or "Xin lỗi, tôi chưa có dữ liệu."
        chat_history.append((user_message, reply))
        return jsonify({"reply": reply, "sources": [s.metadata if hasattr(s, 'metadata') else str(s) for s in sources], "history": chat_history})

    chat_history.append((user_message, reply))
    return jsonify({"reply": reply, "sources": [], "history": chat_history})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
