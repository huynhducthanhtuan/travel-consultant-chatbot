from pinecone import Pinecone, ServerlessSpec
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import requests
import random
import json
import os

load_dotenv()

# Init Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name="vietnam-tourism"
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index( 
        name=index_name,
        dimension=1536, 
        spec=ServerlessSpec( 
            cloud='aws',  
            region='us-east-1' 
        ) 
    )

# Check & upsert embeddings into Pinecone
index=pc.Index(index_name)
stats = index.describe_index_stats()
if stats["total_vector_count"] == 0:
    with open("vietnam_tourism.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    tourism_texts = []
    for province, places in data.items():
        sentence = f"Tỉnh {province} có các điểm du lịch: {', '.join(places)}"
        tourism_texts.append({"province": province, "text": sentence})
    
    client = OpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY_EBD3")
    )

    vectors = []
    for i, item in enumerate(tourism_texts):
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=item["text"]
        ).data[0].embedding
        
        vectors.append((
            str(i),  # id
            emb,     # vector embedding
            {"province": item["province"], "text": item["text"]}  # metadata
        ))
    
    # Lưu vào Pinecone
    index.upsert(vectors)
    print("Upsert embeddings into Pinecone successfully")
else:
    print("Dữ liệu đã tồn tại, bỏ qua upsert")

# Short-term memory message
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

# Functions calling
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

def usd_to_vnd(usd_price):
    try:
        usd_value = float(usd_price)
        vnd_value = usd_value * 24500
        return f"{vnd_value:,.0f} VND"
    except Exception:
        return None

def get_hotel_price(destination: str) -> str:
    url = "https://agoda-travel.p.rapidapi.com/agoda-app/hotels/search-day-use-by-location"
    querystring = {"location": destination}
 
    headers = {
        "x-rapidapi-host": "agoda-travel.p.rapidapi.com",
        "x-rapidapi-key": os.getenv("RAPIDAPI_KEY"),
    }
 
    print(f"Fetching hotel prices for {destination} from {url}")
    try:
        res = requests.get(url, headers=headers, params=querystring, timeout=10)
 
        if res.status_code != 200:
            return f"Xin lỗi, không lấy được giá khách sạn ở {destination}."
 
        data = res.json()
        hotels = data.get("data", {}).get("properties", [])
        if not hotels:
            return f"Không tìm thấy khách sạn nào ở {destination}."
 
        hotel_list = []
        for h in hotels[:3]:
            name = (
                h.get("content", {})
                .get("informationSummary", {})
                .get("localeName", "Không rõ tên")
            )
            try:
                price = (
                    h.get("pricing", {})
                    .get("offers", [])[0]
                    .get("roomOffers", [])[0]
                    .get("room", {})
                    .get("pricing", [])[0]
                    .get("price", {})
                    .get("perBook", {})
                    .get("inclusive", {})
                    .get("display")
                )
            except:
                price = None
 
            if price:
                price_vnd = usd_to_vnd(price)
                hotel_list.append(f"{name}: {price_vnd}/đêm")
            else:
                hotel_list.append(f"{name}: Giá chưa có")
 
        return f"Giá khách sạn ở {destination} (tham khảo):\n" + "\n".join(hotel_list)
 
    except Exception as e:
        return f"Lỗi khi lấy dữ liệu giá khách sạn: {str(e)}"

# Embeddings
# embeddings = AzureOpenAIEmbeddings(
#     model=os.getenv("AZURE_DEPLOYMENT_NAME_EBD3"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY_EBD3"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     openai_api_version="2024-07-01-preview"
# )
# vectorstore = PineconeVectorStore(
#     index, 
#     embedding=embeddings, 
#     text_key="text"
# )

# LangChain Azure Chat (used for RAG conversational retriever)
# chat = AzureChatOpenAI(
#     deployment_name=os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY_GPT4"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version="2024-07-01-preview"
# )
# retrieval_chain = ConversationalRetrievalChain.from_llm(
#     llm=chat,
#     retriever=vectorstore.as_retriever(),
#     return_source_documents=True
# )

# AzureOpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY_GPT4"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-07-01-preview",
    azure_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4")
)

# Functions calling schema
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
# DEST_NAMES = [d.split(":")[0].strip() for d in mock_docs]

def search_tourism(query, top_k=1):
    client = OpenAI(
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY_EBD3")
    )

    # Sinh embedding từ câu hỏi
    query_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
 
    # Query Pinecone
    results = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True
    )
 
    # Xử lý kết quả
    response = []
    for match in results["matches"]:
        province = match["metadata"]["province"]
        text = match["metadata"]["text"]
        response.append(f"[{province}] {text}")
    return "\n".join(response)

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

# def fallback_rag(question, chat_history):
#     """Dùng LangChain retrieval_chain để lấy answer nếu mock không đủ"""
#     rag = retrieval_chain({"question": question, "chat_history": chat_history})
#     # retrieval_chain trả về dict: {'answer': ..., 'source_documents': [...]}
#     return rag.get("answer", "").strip(), rag.get("source_documents", [])

# Helper functions
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
def home():
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

    # search in Pinecone
    search_result = search_tourism(user_message, top_k=1)

    messages = [
        *few_shots,
        *message_history,
        {
            "role": "system",
            "content": f"""
                Bạn là một hướng dẫn viên du lịch Việt Nam.
                {f"Đây là dữ liệu du lịch lấy từ cơ sở dữ liệu: {search_result}" if search_result else ""}
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
    print(messages)

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
        # if dest and dest not in DEST_NAMES:
        #     rag_answer, sources = fallback_rag(user_message, chat_history)
        #     reply = rag_answer or f"Xin lỗi, không tìm thấy thông tin về {dest} trong dữ liệu của tôi."
        #     chat_history.append((user_message, reply))
        #     return jsonify({"reply": reply, "sources": [s.metadata if hasattr(s, 'metadata') else str(s) for s in sources], "history": chat_history})

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
        elif func_name == "get_hotel_price":
            result = get_hotel_price(dest)
        else:
            result = "Không tìm thấy chức năng."

        chat_history.append((user_message, result))
        return jsonify({"reply": result, "sources": [], "history": chat_history})

    # else not function_call -> if model returned plain content, use it
    reply = getattr(message, "content", "") or ""

    # Fallback RAG
    # if not reply.strip() or "không biết" in reply.lower():
    #     rag_answer, sources = fallback_rag(user_message, chat_history)
    #     reply = rag_answer or reply or "Xin lỗi, tôi chưa có dữ liệu."
    #     chat_history.append((user_message, reply))
    #     return jsonify({"reply": reply, "sources": [s.metadata if hasattr(s, 'metadata') else str(s) for s in sources], "history": chat_history})

    chat_history.append((user_message, reply))
    return jsonify({"reply": reply, "sources": [], "history": chat_history})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
