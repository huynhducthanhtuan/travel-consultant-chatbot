# from pinecone import Pinecone, ServerlessSpec
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI
from dotenv import load_dotenv
import requests
import random
import json
import os

load_dotenv()

# Mock data
destinations_data = [
    "Hà Nội: Thủ đô ngàn năm văn hiến. Địa điểm nổi bật: hồ Hoàn Kiếm, phố cổ, Văn Miếu Quốc Tử Giám, Lăng Chủ tịch Hồ Chí Minh. Lịch trình gợi ý 2N1Đ: Ngày 1 tham quan phố cổ, hồ Hoàn Kiếm, thưởng thức phở Hà Nội và bún chả; Ngày 2 tham quan Lăng Bác, chùa Trấn Quốc, mua quà lưu niệm.",
    "Hạ Long: Di sản thiên nhiên thế giới, nổi tiếng với hàng ngàn đảo đá vôi và hang động kỳ thú. Lịch trình gợi ý 2N1Đ: Ngày 1 đi du thuyền tham quan vịnh, thăm hang Sửng Sốt, chèo kayak; Ngày 2 tham quan Sun World, chợ Hạ Long. Món ngon: chả mực, sá sùng.",
    "Sapa: Thị trấn vùng cao, khí hậu mát mẻ. Địa điểm: ruộng bậc thang, đỉnh Fansipan, bản Cát Cát, Tả Van. Lịch trình 3N2Đ: Ngày 1 khám phá thị trấn, chợ đêm; Ngày 2 leo Fansipan hoặc đi cáp treo, thăm bản làng; Ngày 3 mua quà lưu niệm. Món ngon: thắng cố, cá hồi Sapa.",
    "Hà Giang: Cao nguyên đá Đồng Văn, đèo Mã Pì Lèng, lễ hội hoa tam giác mạch. Lịch trình 3N2Đ: Ngày 1 đi Quản Bạ, Yên Minh; Ngày 2 thăm Đồng Văn, Mã Pì Lèng; Ngày 3 quay về Hà Nội. Món ngon: thắng cố, cháo ấu tẩu.",
    "Ninh Bình: Tràng An, Tam Cốc – Bích Động, chùa Bái Đính. Lịch trình 2N1Đ: Ngày 1 đi Tam Cốc, Bích Động; Ngày 2 Tràng An, Bái Đính. Món ngon: dê núi, cơm cháy.",
    "Mộc Châu: Cao nguyên chè, hoa mận, thác Dải Yếm. Lịch trình 2N1Đ: Ngày 1 thăm đồi chè, thác Dải Yếm; Ngày 2 rừng thông bản Áng, thưởng thức đặc sản sữa bò. Món ngon: bê chao, cá hồi.",
    "Cát Bà: Vườn quốc gia Cát Bà, bãi tắm Cát Cò, làng chài. Lịch trình 2N1Đ: Ngày 1 tắm biển, thăm làng chài; Ngày 2 trekking trong vườn quốc gia. Món ngon: hải sản tươi, tu hài.",
    "Lạng Sơn: Động Tam Thanh, chợ Đông Kinh, núi Mẫu Sơn. Lịch trình 2N1Đ: Ngày 1 thăm động Tam Thanh, mua sắm chợ Đông Kinh; Ngày 2 leo núi Mẫu Sơn. Món ngon: vịt quay, phở chua.",
    "Yên Bái: Hồ Thác Bà, Mù Cang Chải, ruộng bậc thang. Lịch trình 3N2Đ: Ngày 1 thăm hồ Thác Bà; Ngày 2 – 3 khám phá Mù Cang Chải. Món ngon: xôi ngũ sắc, cá nướng.",
    "Lào Cai: Bắc Hà, chợ phiên, dinh Hoàng A Tưởng. Lịch trình 2N1Đ: Ngày 1 tham quan dinh Hoàng A Tưởng, chợ phiên; Ngày 2 khám phá bản làng. Món ngon: mèn mén, rượu ngô.",
    "Quảng Ninh: Yên Tử, đảo Cô Tô, đảo Quan Lạn. Lịch trình 3N2Đ: Ngày 1 leo núi Yên Tử; Ngày 2 đi tàu ra Cô Tô; Ngày 3 tắm biển Quan Lạn. Món ngon: sá sùng, cù kỳ.",
    "Nam Định: Nhà thờ Phú Nhai, biển Thịnh Long. Lịch trình 2N1Đ: Ngày 1 tham quan nhà thờ, thưởng thức phở bò Nam Định; Ngày 2 tắm biển Thịnh Long. Món ngon: bánh xíu páo.",
    "Thái Bình: Biển Đồng Châu, làng vườn Bách Thuận. Lịch trình 2N1Đ: Ngày 1 tắm biển; Ngày 2 tham quan làng vườn. Món ngon: cá kho làng Vũ Đại.",
    "Cao Bằng: Thác Bản Giốc, động Ngườm Ngao. Lịch trình 2N1Đ: Ngày 1 thăm thác Bản Giốc; Ngày 2 động Ngườm Ngao. Món ngon: vịt quay 7 vị, phở chua.",
    "Huế: Cố đô, Đại Nội, lăng tẩm vua Nguyễn. Lịch trình 3N2Đ: Ngày 1 Đại Nội, chùa Thiên Mụ; Ngày 2 lăng Minh Mạng, lăng Tự Đức; Ngày 3 chợ Đông Ba. Món ngon: bún bò Huế, chè cung đình.",
    "Đà Nẵng: Bà Nà Hills, cầu Rồng, biển Mỹ Khê. Lịch trình 2N1Đ: Ngày 1 Bà Nà Hills; Ngày 2 biển Mỹ Khê, chợ Hàn. Món ngon: mì Quảng, bánh tráng cuốn thịt heo.",
    "Hội An: Phố cổ UNESCO, đèn lồng, ẩm thực đặc sắc. Lịch trình 2N1Đ: Ngày 1 tham quan phố cổ, chùa Cầu, ăn cao lầu, cơm gà; Ngày 2 đi làng gốm Thanh Hà, làng rau Trà Quế. Món ngon: cao lầu, bánh mì Phượng.",
    "Quảng Bình: Hang Sơn Đoòng, động Phong Nha - Kẻ Bàng. Lịch trình 3N2Đ: Ngày 1 động Phong Nha; Ngày 2 hang Sơn Đoòng; Ngày 3 suối nước Moọc. Món ngon: cháo canh, khoai deo.",
    "Nha Trang: Thành phố biển, Tháp Bà Ponagar, đảo Hòn Mun. Lịch trình 3N2Đ: Ngày 1 tắm biển, Tháp Bà Ponagar; Ngày 2 đảo Hòn Mun, lặn ngắm san hô; Ngày 3 chợ Đầm. Món ngon: nem nướng Ninh Hòa, bún sứa.",
    "Đà Lạt: Thành phố ngàn hoa, hồ Xuân Hương, vườn dâu. Lịch trình 3N2Đ: Ngày 1 hồ Xuân Hương, vườn hoa; Ngày 2 đồi chè Cầu Đất, thác Datanla; Ngày 3 chợ Đà Lạt. Món ngon: lẩu gà lá é, bánh tráng nướng.",
    "TP.HCM: Chợ Bến Thành, Nhà thờ Đức Bà, phố đi bộ Nguyễn Huệ. Lịch trình 2N1Đ: Ngày 1 tham quan trung tâm, ăn cơm tấm; Ngày 2 đi chợ Bến Thành, bảo tàng. Món ngon: hủ tiếu, bánh mì.",
    "Vũng Tàu: Bãi Sau, bãi Trước, tượng Chúa Kitô Vua. Lịch trình 2N1Đ: Ngày 1 tắm biển Bãi Sau; Ngày 2 tham quan tượng Chúa, mua hải sản. Món ngon: bánh khọt, lẩu cá đuối.",
    "Cần Thơ: Chợ nổi Cái Răng, miệt vườn trái cây. Lịch trình 2N1Đ: Ngày 1 tham quan chợ nổi; Ngày 2 đi miệt vườn. Món ngon: lẩu mắm, cá lóc nướng trui.",
    "Phú Quốc: Bãi Sao, VinWonders, lặn ngắm san hô. Lịch trình 3N2Đ: Ngày 1 VinWonders, Safari; Ngày 2 bãi Sao, bãi Dài; Ngày 3 chợ đêm. Món ngon: ghẹ Hàm Ninh, gỏi cá trích.",
    "Cà Mau: Mũi Cà Mau, rừng U Minh Hạ. Lịch trình 2N1Đ: Ngày 1 đi Mũi Cà Mau; Ngày 2 tham quan rừng U Minh Hạ. Món ngon: ba khía muối, lẩu mắm.",
    "Bến Tre: Xứ dừa, cồn Phụng, du lịch miệt vườn. Lịch trình 2N1Đ: Ngày 1 tham quan cồn Phụng, ăn đặc sản từ dừa; Ngày 2 chèo xuồng kênh rạch. Món ngon: kẹo dừa, cá kho tộ."
]

mock_docs = destinations_data

# USING PINECONE INSTEAD OF FAISS
# ebd_client = AzureOpenAI( 
#     api_version="2024-07-01-preview", 
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
#     api_key=os.getenv("AZURE_DEPLOYMENT_NAME_EBD3")
# ) 
# index_name = "destinations"
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY")) 

# if index_name not in [index["name"] for index in pc.list_indexes()]: 
#     pc.create_index(name=index_name, dimension=1536, spec=ServerlessSpec(cloud="aws", region="us-east-1")) 
        
# index = pc.Index(index_name) 

# vectors = []
# for item in destinations_data:
#     emb = ebd_client.embeddings.create(
#         model=os.getenv("AZURE_DEPLOYMENT_NAME_EBD3"),
#         input=item["text"]
#     ).data[0].embedding
#     vectors.append({
#         "id": item["id"],
#         "values": emb,
#         "metadata": {"text": item["text"]}
#     })

# # Upsert vào Pinecone
# if len(vectors) > 0:
#     index.upsert(vectors)

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
            Trả lời chỉ bằng văn bản thuần (plain text), không dùng Markdown, không dùng ký hiệu #, -, *, **.
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

def get_weather_city(city: str) -> str:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": os.getenv("OPEN_WEATHER_MAP_API_KEY"),
        "units": "metric",
        "lang": "vi"
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if response.status_code != 200:
            print(f"Error: {data.get('message', 'Failed to fetch weather data')}")
            return f"Error: {data.get('message', 'Failed to fetch weather data')}"

        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        return f"Thời tiết ở thành phố {city} is {weather}, nhiệt độ là {temp}°C (cảm giác như {feels_like}°C)."

    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error: {str(e)}"

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
    },
    {
        "name": "get_weather_city",
        "description": "Lấy thời tiết ở thành phố",
        "parameters": {
            "type": "object",
            "properties": {"destination": {"type": "string"}},
            "required": ["destination"]
        }
    },
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
            Lịch trình gợi ý:
            Ngày 1: Tham quan Bà Nà Hills (vé 900k), buổi chiều dạo biển Mỹ Khê, tối ăn hải sản ở Bé Mặn.
            Ngày 2: Sáng tham quan Ngũ Hành Sơn, chiều đi Hội An (vé vào phố cổ 80k), tối ngắm đèn lồng.
            Ngày 3: Sáng chợ Hàn mua quà, ăn mì Quảng ếch Bà Mua, ra sân bay.
            Mẹo: Mang kem chống nắng, đặt vé Bà Nà trước 1-2 ngày, thuê xe máy di chuyển thuận tiện.
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
    Kiểm tra xem câu hỏi có liên quan đến du lịch Việt Nam hoặc thời tiết tại các tỉnh/thành Việt Nam không.
    Trả về True nếu liên quan, False nếu không.
    """
    check_messages = [
        {
            "role": "system",
            "content": (
                "Bạn là bộ lọc phân loại câu hỏi. "
                "Trả lời chỉ 'yes' nếu câu hỏi liên quan đến: "
                "1) Du lịch Việt Nam (địa điểm, lịch trình, ẩm thực, mẹo du lịch) "
                "hoặc 2) Thời tiết ở các tỉnh/thành phố của Việt Nam. "
                "Trả lời 'no' nếu không liên quan."
            )
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

def is_vietnamese_language(text: str) -> bool:
    """
    Dùng Azure GPT để xác định câu có phải tiếng Việt không.
    Trả về True nếu là tiếng Việt, False nếu không.
    """
    check_messages = [
        {
            "role": "system",
            "content": "Bạn là bộ lọc ngôn ngữ. Trả lời duy nhất 'yes' nếu câu hỏi là tiếng Việt, 'no' nếu không."
        },
        {"role": "user", "content": text}
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
        print("Error in language detection:", e)
        # Nếu lỗi thì cho qua (hoặc mặc định False)
        return True

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    user_message = data.get("message", "")
    chat_history = data.get("history", [])

    if not is_vietnamese_language(user_message):
        reply = "Vui lòng hỏi đáp bằng tiếng Việt."
        chat_history.append((user_message, reply))
        return jsonify({"reply": reply, "sources": [], "history": chat_history})

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
        elif func_name == "get_weather_city":
            result = get_weather_city(dest)
        else:
            result = "Không tìm thấy chức năng."

        chat_history.append((user_message, result))
        return jsonify({"reply": result, "sources": [], "history": chat_history})

    # else not function_call -> if model returned plain content, use it
    reply = getattr(message, "content", "") or ""

    # Fallback RAG
    if not reply.strip() or "không biết" in reply.lower():
        rag_answer, sources = fallback_rag(user_message, chat_history)
        reply = rag_answer or reply or "Xin lỗi, tôi chưa có dữ liệu."
        chat_history.append((user_message, reply))
        return jsonify({"reply": reply, "sources": [s.metadata if hasattr(s, 'metadata') else str(s) for s in sources], "history": chat_history})

    chat_history.append((user_message, reply))
    return jsonify({"reply": reply, "sources": [], "history": chat_history})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
