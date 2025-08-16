from pinecone import Pinecone, ServerlessSpec
from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
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
            Quy tắc đặc biệt:
                - Nếu câu hỏi liên quan đến **giá khách sạn**, luôn gọi tool get_hotel_prices.
                - Nếu câu hỏi liên quan đến **giá vé máy bay**, luôn gọi tool get_flight_prices.
                - Nếu câu hỏi liên quan đến **thời tiết**, luôn gọi tool get_weather_city.
                - Các câu hỏi khác (thời tiết, địa điểm du lịch, mẹo du lịch, sự kiện) thì trả lời trực tiếp.
                - Trả lời ngắn gọn, dễ hiểu, văn bản thuần, không dùng Markdown.
        """
    }
]

# Functions calling
@tool
def get_flight_price(origin: str, destination: str, trip_type: str, currency: str) -> str:
    """
    Lấy giá vé máy bay giữa hai thành phố.
 
    Args:
        origin (str): The name of the origin city or country.
        destination (str): The name of the destination city or country.
        trip_type (str): Loại chuyến đi, nhận một trong hai giá trị:
            - "one_way"  : chuyến bay một chiều
            - "round_trip": chuyến bay khứ hồi
        currency (str): Đơn vị tiền tệ muốn hiển thị giá (mặc định là "VND").
 
    Returns:
        str: A formatted string with the estimated round-trip price in VND.
    """
    api_endpoint = "https://www.mocky.io/v2/5c3c7d2d300000980a055e76"  # Thay bằng URL Mocky.io thực tế
    trip_name = "khứ hồi" if trip_type == "round_trip" else "một chiều"
   
    try:
        response = requests.get(api_endpoint, timeout=5)
        response.raise_for_status()
       
        data = response.json()
        flights = data.get("flights", [])
        # Tìm chuyến bay khớp với origin, destination, và trip_type
        for flight in flights:
            if (flight.get("origin", "").strip().encode('utf-8').decode('utf-8') == origin.strip().encode('utf-8').decode('utf-8') and
                flight.get("destination", "").strip().encode('utf-8').decode('utf-8') == destination.strip().encode('utf-8').decode('utf-8') and
                flight.get("trip_type", "").strip().encode('utf-8').decode('utf-8') == trip_type.strip().encode('utf-8').decode('utf-8')):
                price = flight.get("price", random.randint(1500000, 5000000))
                return f"Giá vé máy bay {trip_name} từ {origin} đến {destination} khoảng {price:,} {currency}."
       
        # Nếu không tìm thấy chuyến bay khớp
        return f"Không tìm thấy chuyến bay {trip_name} từ {origin} đến {destination}."
   
    except requests.exceptions.RequestException as e:
        try:
            # Fallback: Read from local mock_data.json
            with open(os.path.join("data", "mock_data.json"), "r", encoding="utf-8") as file:
                data = json.load(file)
            flights = data.get("flights", [])
            print("flights:", flights)
            for flight in flights:
                if (flight.get("origin", "").strip().encode('utf-8').decode('utf-8') == origin.strip().encode('utf-8').decode('utf-8') and
                    flight.get("destination", "").strip().encode('utf-8').decode('utf-8') == destination.strip().encode('utf-8').decode('utf-8') and
                    flight.get("trip_type", "").strip().encode('utf-8').decode('utf-8') == trip_type.strip().encode('utf-8').decode('utf-8')):
                    price = flight.get("price", random.randint(1500000, 5000000))
                    return f"Giá vé máy bay {trip_name} từ {origin} đến {destination} khoảng {price:,} {currency}."
           
            price = random.randint(1500000, 5000000)
            return f"Không tìm thấy chuyến bay {trip_name} từ {origin} đến {destination}."
       
        except (FileNotFoundError, json.JSONDecodeError) as file_error:
            return f"Giá vé máy bay {trip_name} từ {origin} đến {destination} khoảng {random.randint(1500000, 5000000):,} {currency} (dữ liệu dự phòng do lỗi khi đọc file mock_data.json: {str(file_error)})."


@tool
def get_itinerary(destination: str, days: int) -> str:
    """
    Tạo một hành trình du lịch đơn giản cho một điểm đến và số ngày nhất định.
 
    Args:
        destination (str): The name of the destination city or country.
        days (int): Number of days for the itinerary.
 
    Returns:
        str: A multi-line string describing daily activities at the destination.
    """
 
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
 
@tool
def get_weather_city(city: str) -> str:
    """
    Lấy thông tin thời tiết hiện tại của một thành phố nhất định bằng API OpenWeatherMap.
 
    Args:
        city (str): The name of the city to get weather information for.
 
    Returns:
        str: A formatted string describing the weather, temperature, and "feels like" temperature.
             If an error occurs, returns an error message.
    """
 
    url = "https://api.openweathermap.org/data/2.5/weather"
    # url = "https://api.openweathermap.org/data/2.5/onecall"
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
        return f"Thời tiết ở thành phố {city} là {weather}, nhiệt độ là {temp}°C (cảm giác như {feels_like}°C)."
 
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"Error: {str(e)}"
   
@tool
def usd_to_vnd(usd_price: float) -> str | None:
    """
    Chuyển đổi một số tiền USD nhất định sang VND bằng tỷ giá hối đoái cố định.
 
    Args:
        usd_price (float): The amount in USD to convert.
 
    Returns:
        str | None: The converted amount formatted in VND (e.g., "24,500 VND").
                    Returns None if the input cannot be converted to float.
    """
    try:
        usd_value = float(usd_price)
        vnd_value = usd_value * 24_500
        return f"{vnd_value:,.0f} VND"
    except Exception:
        return None
   
@tool
def get_hotel_price(destination: str) -> str:
    """
    Lấy giá khách sạn theo thành phố
 
    Args:
        destination (str): The name of the destination city.
 
    Returns:
        str: A formatted list of up to 3 hotels with their prices in VND.
             If no hotels or prices are found, returns a fallback message.
             If an error occurs, returns an error message.
    """
   
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
        "description": "Lấy giá vé máy bay từ điểm đi đến điểm du lịch (khứ hồi hoặc một chiều)",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Địa điểm khởi hành"},
                "destination": {"type": "string", "description": "Địa điểm đến"},
                "trip_type": {
                    "type": "string",
                    "description": "Loại chuyến đi: 'round_trip' cho khứ hồi, 'one_way' cho một chiều",
                    "enum": ["round_trip", "one_way"],
                    "default": "round_trip"
                }
            },
            "required": ["origin", "destination"]
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
    # Địa điểm du lịch
    {
        "role": "user",
        "content": "Gợi ý cho tôi một số địa điểm du lịch nổi tiếng ở Đà Nẵng."
    },
    {
        "role": "assistant",
        "content": "Một số địa điểm nổi tiếng ở Đà Nẵng gồm: Bà Nà Hills, biển Mỹ Khê, cầu Rồng, Ngũ Hành Sơn, chợ Hàn, bán đảo Sơn Trà, và phố cổ Hội An (cách Đà Nẵng khoảng 30km)."
    },
 
    # Thời tiết
    {
        "role": "user",
        "content": "Thời tiết Đà Nẵng hôm nay thế nào?"
    },
    {
        "role": "assistant",
        "content": "Hôm nay Đà Nẵng có nắng nhẹ, nhiệt độ khoảng 28-32°C, độ ẩm 70%, gió nhẹ, thích hợp cho các hoạt động ngoài trời."
    },
 
    # Giá vé máy bay
    {
        "role": "user",
        "content": "Giá vé máy bay khứ hồi từ TP.HCM đến Đà Nẵng là bao nhiêu?"
    },
    {
        "role": "assistant",
        "content": "Giá vé khứ hồi từ TP.HCM đến Đà Nẵng khoảng 1,500,000 - 2,500,000 VND, tùy hãng bay và thời điểm đặt."
    },
 
    # Giá phòng khách sạn
    {
        "role": "user",
        "content": "Giá phòng khách sạn ở Đà Nẵng là bao nhiêu?"
    },
    {
        "role": "assistant",
        "content": "Giá phòng khách sạn ở Đà Nẵng dao động từ 500,000 VND/đêm cho khách sạn 2-3 sao, từ 1,500,000 VND/đêm cho resort hoặc khách sạn ven biển cao cấp."
    },


     # Mẹo du lịch
    {
        "role": "user",
        "content": "Đi du lịch Đà Nẵng thì nên lưu ý gì?"
    },
    {
        "role": "assistant",
        "content": "Nên mang theo kem chống nắng, mũ, kính râm. Đặt vé Bà Nà Hills trước để tránh hết chỗ. Buổi tối cuối tuần có thể xem cầu Rồng phun lửa, nhớ đi sớm để chọn chỗ đẹp."
    },


    # So sánh thành phố
    {
        "role": "user",
        "content": "Đi Nha Trang hay Đà Nẵng thì hợp hơn?"
    },
    {
        "role": "assistant",
        "content": """
        - Đà Nẵng: có nhiều điểm tham quan nổi tiếng như Bà Nà Hills, cầu Rồng, gần Hội An, thích hợp cho cả gia đình.
        - Nha Trang: nổi tiếng về biển đảo, lặn ngắm san hô, VinWonders, phù hợp cho ai thích nghỉ dưỡng biển.
        Tùy bạn thích trải nghiệm văn hóa hay nghỉ dưỡng mà chọn.
        """
    },
    # Khách sạn cao cấp
    {
        "role": "user",
        "content": "Gợi ý cho tôi khách sạn 5 sao ở Đà Nẵng."
    },
    {
        "role": "assistant",
        "content": "Một số khách sạn 5 sao ở Đà Nẵng: InterContinental Danang Sun Peninsula Resort (~9,000,000 VND/đêm), Furama Resort (~3,500,000 VND/đêm), Hyatt Regency (~4,000,000 VND/đêm)."
    },


    # Phương tiện di chuyển
    {
        "role": "user",
        "content": "Ở Đà Nẵng nên di chuyển bằng gì tiện nhất?"
    },
    {
        "role": "assistant",
        "content": "Bạn có thể thuê xe máy (~120k-150k/ngày) để đi lại linh hoạt. Nếu đi gia đình thì nên thuê taxi hoặc Grab. Ngoài ra còn có xe bus đi Hội An giá rẻ."
    },


    # Sự kiện đặc biệt
    {
        "role": "user",
        "content": "Đà Nẵng có lễ hội gì đặc biệt?"
    },
    {
        "role": "assistant",
        "content": "Đà Nẵng nổi tiếng với Lễ hội pháo hoa quốc tế (DIFF) thường diễn ra vào tháng 6-7 hằng năm, thu hút nhiều du khách trong và ngoài nước."
    },
   


]

# Flask app
app = Flask(__name__)

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
                "hoặc 3) Giá vé máy bay ở các tỉnh/thành phố của Việt Nam. "
                "hoặc 4) Giá phòng khách sạn ở các tỉnh/thành phố của Việt Nam. "
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


# LangChain tools
tavily_search_tool = TavilySearch(
    max_results=1,
    topic="general",
)
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME_GPT4"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2024-07-01-preview",
    api_key=os.getenv("AZURE_OPENAI_API_KEY_GPT4"),
)
tools = [
    get_flight_price,
    get_hotel_price,
    get_weather_city,
    get_itinerary,
    usd_to_vnd,
    tavily_search_tool
]
agent = create_react_agent(
    model=llm,
    tools=tools,
)


@app.route("/")
def home():
    return render_template("index.html")


# Lưu lịch sử chat theo session_id (demo)
chat_histories = {}
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    user_message = data.get("message", "")
    chat_history = data.get("history", [])
    session_id = data.get("session_id", "default")
    user_input = data.get("message", "")
    history = chat_histories.get(session_id, [])
    history.append({"role": "user", "content": user_input})


    if not is_vietnamese_language(user_message):
        reply = "Vui lòng hỏi những vấn đề du lịch bằng tiếng Việt."
        chat_history.append((user_message, reply))
        return jsonify({"reply": reply, "sources": [], "history": chat_history})


    if not is_tourism_related(user_message):
        reply = "Tôi là chatbot về du lịch, hãy hỏi câu hỏi liên quan đến địa điểm, lịch trình du lịch ở Việt Nam."
        chat_history.append((user_message, reply))
        return jsonify({"reply": reply, "sources": [], "history": chat_history})


    # search in Pinecone
    search_result = search_tourism(user_message, top_k=1)


    messages = [
        *message_history,
        *few_shots,
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
        response = agent.invoke({"messages": messages})
        ai_reply = response["messages"][-1].content


        chat_history.append((user_message, ai_reply))
        return jsonify({"reply": ai_reply, "sources": [], "history": chat_history})


    # else not function_call -> if model returned plain content, use it
    reply = getattr(message, "content", "") or ""


    chat_history.append((user_message, reply))
    return jsonify({"reply": reply, "sources": [], "history": chat_history})


if __name__ == "__main__":
    if os.getenv("IS_PRODUCTION_MODE", "").strip().lower() == "true":
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    else:
        app.run(host="0.0.0.0", port=5000, debug=True)
