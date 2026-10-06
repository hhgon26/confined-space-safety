import math
import os
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
import xgboost as xgb

# =========================================================
# 페이지 기본 설정
# =========================================================
st.set_page_config(
    page_title="밀폐공간 작업자 안전 모니터링",
    page_icon="🚨",
    layout="wide"
)

# =========================================================
# Step 1: XGBoost 모델 학습 및 저장/캐싱
# =========================================================
MODEL_FILE = 'confined_space_xgb_model.pkl'

def train_and_save_model():
    np.random.seed(42)
    n_samples = 10000

    temperature = np.random.uniform(18.0, 38.0, n_samples)
    humidity = np.random.uniform(30.0, 95.0, n_samples)
    o2_level = np.random.uniform(15.0, 21.0, n_samples)
    h2s_ppm = np.random.exponential(scale=2.5, size=n_samples)
    movement = np.random.uniform(0.0, 10.0, n_samples)

    heat_index = temperature + 0.33 * humidity - 0.70 * 4.0 - 4.0

    risk_score = (
        (o2_level < 18.0).astype(int) * 3.0 +
        (h2s_ppm >= 10.0).astype(int) * 3.0 +
        ((temperature > 32.0) & (humidity > 80.0) & (movement < 2.0)).astype(int) * 2.0 +
        (heat_index > 38.0).astype(int) * 1.0 +
        np.random.normal(0, 0.4, n_samples)
    )

    y = (risk_score >= 2.5).astype(int)

    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'o2_level': o2_level,
        'h2s_ppm': h2s_ppm,
        'movement': movement,
        'heat_index': heat_index,
        'risk_label': y
    })

    X = df.drop(columns=['risk_label'])
    y_target = df['risk_label']

    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X, y_target)
    joblib.dump(model, MODEL_FILE)
    return model

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        return train_and_save_model()
    return joblib.load(MODEL_FILE)

model = load_model()

# =========================================================
# Step 2: 위도/경도 -> 기상청 격자(nx, ny) 변환 함수
# =========================================================
def convert_to_grid(lat, lon):
    RE = 6371.00877   # 지구 반경(km)
    GRID = 5.0        # 격자 간격(km)
    SLAT1 = 30.0      # 투영 위도1(degree)
    SLAT2 = 60.0      # 투영 위도2(degree)
    OLON = 126.0      # 기준점 경도(degree)
    OLAT = 38.0       # 기준점 위도(degree)
    XO = 43           # 기준점 X좌표(GRID)
    YO = 136          # 기준점 Y좌표(GRID)

    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = math.pow(sf, sn) * math.cos(slat1) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / math.pow(ro, sn)

    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / math.pow(ra, sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    nx = math.floor(ra * math.sin(theta) + XO + 0.5)
    ny = math.floor(ro - ra * math.cos(theta) + YO + 0.5)
    return int(nx), int(ny)

# =========================================================
# Step 3: 기상청 단기예보 API 호출 함수
# =========================================================
SERVICE_KEY = "77a3d0095b4e5151f66c3dde4bd0925f328c9ccd462427268711a4a4af30f020"

def get_kma_weather(nx, ny):
    now = datetime.now()
    base_date = now.strftime("%Y%m%d")
    
    if now.minute < 40:
        if now.hour == 0:
            base_date = (now - pd.Timedelta(days=1)).strftime("%Y%m%d")
            base_time = "2300"
        else:
            base_time = f"{now.hour - 1:02d}00"
    else:
        base_time = f"{now.hour:02d}00"

    url = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
    
    params = {
        'serviceKey': SERVICE_KEY,
        'pageNo': '1',
        'numOfRows': '10',
        'dataType': 'JSON',
        'base_date': base_date,
        'base_time': base_time,
        'nx': str(nx),
        'ny': str(ny)
    }

    try:
        # ⭐ verify=False 옵션을 추가하여 SSL 인증서 검증 생략
        response = requests.get(url, params=params, timeout=5, verify=False)
        res_json = response.json()
        
        if 'response' in res_json:
            header = res_json['response']['header']
            if header.get('resultCode') == '00':
                temp, hum = None, None
                items = res_json['response']['body']['items']['item']
                for item in items:
                    if item['category'] == 'T1H':   # 기온 (°C)
                        temp = float(item['obsrValue'])
                    elif item['category'] == 'REH': # 습도 (%)
                        hum = float(item['obsrValue'])
                return temp, hum, None
            else:
                return 25.0, 60.0, f"기상청 API 오류: {header.get('resultMsg')} (코드: {header.get('resultCode')})"
        else:
            return 25.0, 60.0, f"응답 형식 오류: {response.text[:100]}"

    except Exception as e:
        return 25.0, 60.0, f"기상청 API 연동 실패: {str(e)}"

# =========================================================
# Step 4: Streamlit UI 화면 구성
# =========================================================
st.title("🚨 밀폐공간 작업자 안전 모니터링 시스템")
st.markdown("현장 센서 데이터와 기상청 실시간 데이터를 결합하여 작업장의 위험도를 실시간 예측합니다.")

# 사이드바: 입력 파라미터
st.sidebar.header("📍 위치 및 센서 설정")

st.sidebar.subheader("위치 정보 (GPS)")
lat = st.sidebar.number_input("위도 (Latitude)", value=37.5665, format="%.6f")
lon = st.sidebar.number_input("경도 (Longitude)", value=126.9780, format="%.6f")

st.sidebar.subheader("현장 센서 측정값")
o2 = st.sidebar.slider("산소 농도 (%)", min_value=10.0, max_value=25.0, value=20.9, step=0.1)
h2s = st.sidebar.number_input("황화수소 농도 (H2S ppm)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
move = st.sidebar.slider("작업자 활동량 (Movement)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

# 진단 실행
nx, ny = convert_to_grid(lat, lon)
temp, hum, api_err = get_kma_weather(nx, ny)

if api_err:
    st.warning(f"⚠️ {api_err}")

# Heat Index 계산
hi = temp + 0.33 * hum - 0.70 * 4.0 - 4.0

# 모델 예측
input_df = pd.DataFrame([{
    'temperature': temp,
    'humidity': hum,
    'o2_level': o2,
    'h2s_ppm': h2s,
    'movement': move,
    'heat_index': hi
}])

risk_prob = float(model.predict_proba(input_df)[0][1] * 100)

# 메인 화면 카드 레이아웃
st.subheader("📊 실시간 모니터링 대시보드")

col1, col2, col3, col4 = st.columns(4)
col1.metric("현재 기온", f"{temp} °C")
col2.metric("현재 습도", f"{hum} %")
col3.metric("산소 농도 (O2)", f"{o2} %")
col4.metric("황화수소 (H2S)", f"{h2s} ppm")

st.markdown("---")

# 위험도 종합 진단 표시
st.subheader("⚠️ 위험도 예측 결과")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.metric("위험 확률", f"{risk_prob:.2f} %")

with col_right:
    if risk_prob >= 75.0:
        st.error("🔴 **위험 (Immediate Danger)**: 작업자 대피 및 환기 조치가 즉시 필요합니다.")
    elif risk_prob >= 40.0:
        st.warning("🟡 **주의 (Warning)**: 환경 상태 모니터링을 강화하세요.")
    else:
        st.success("🟢 **정상 (Normal)**: 작업 환경이 안전한 상태입니다.")

# 위경도 지도 표시
st.subheader("📍 작업 현장 위치")
map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
st.map(map_data)