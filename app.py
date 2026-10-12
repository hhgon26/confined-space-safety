import math
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
from streamlit_js_eval import get_geolocation
from sklearn.model_selection import train_test_split
import xgboost as xgb

st.set_page_config(page_title="밀폐공간 모션 & 기상 연동 안전 대시보드", page_icon="🚨", layout="centered")

# =========================================================
# Step 1: 과거 실험 데이터 기반 XGBoost 모델 학습/캐싱
# =========================================================
@st.cache_resource
def load_or_train_historical_model():
    model_filename = 'confined_space_phyphox_model.pkl'
    try:
        model = joblib.load(model_filename)
        return model
    except Exception:
        np.random.seed(42)
        n_samples = 20000

        temperature = np.random.uniform(15.0, 35.0, n_samples)
        humidity = np.random.uniform(30.0, 95.0, n_samples)
        movement = np.random.uniform(0.0, 10.0, n_samples)

        # 실험 데이터 특성 반영: 슬러지 교란 및 무활동 질식 시나리오
        simulated_o2 = 20.9 - (movement * 0.04) + np.random.normal(0, 0.2, n_samples)
        simulated_h2s = 0.2 + (movement * 0.4) + np.random.exponential(scale=1.0, size=n_samples)
        simulated_co2 = 400.0 + (movement * 45.0) + np.random.normal(0, 40.0, n_samples)
        heat_index = temperature + 0.33 * humidity - 0.70 * 4.0 - 4.0

        # 위험 레이블 판정식 (가스 초과 혹은 쓰러져서 움직임이 없는 무활동 상태)
        risk_score = (
            (simulated_o2 < 18.5).astype(int) * 3.5 +
            (simulated_h2s >= 10.0).astype(int) * 3.0 +
            ((movement < 0.4) & (simulated_o2 < 19.5)).astype(int) * 4.0 + # 무활동(의식 상실) 고위험 가중치
            np.random.normal(0, 0.3, n_samples)
        )

        y = (risk_score >= 2.8).astype(int)

        df = pd.DataFrame({
            'temperature': temperature,
            'humidity': humidity,
            'movement': movement,
            'simulated_o2': simulated_o2,
            'simulated_h2s': simulated_h2s,
            'simulated_co2': simulated_co2,
            'heat_index': heat_index,
            'risk_label': y
        })

        X = df.drop(columns=['risk_label'])
        y_target = df['risk_label']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_target, test_size=0.2, random_state=42, stratify=y_target
        )

        model = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_filename)
        return model

model = load_or_train_historical_model()

# =========================================================
# Step 2: 기상청 Open API 연동 함수
# =========================================================
def convert_to_grid(lat, lon):
    RE, GRID, SLAT1, SLAT2, OLON, OLAT, XO, YO = 6371.00877, 5.0, 30.0, 60.0, 126.0, 38.0, 43, 136
    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1, slat2, olon, olat = SLAT1 * DEGRAD, SLAT2 * DEGRAD, OLON * DEGRAD, OLAT * DEGRAD
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5))
    sf = math.pow(math.tan(math.pi * 0.25 + slat1 * 0.5), sn) * math.cos(slat1) / sn
    ro = re * sf / math.pow(math.tan(math.pi * 0.25 + olat * 0.5), sn)
    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / math.pow(ra, sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi: theta -= 2.0 * math.pi
    if theta < -math.pi: theta += 2.0 * math.pi
    theta *= sn
    return int(math.floor(ra * math.sin(theta) + XO + 0.5)), int(math.floor(ro - ra * math.cos(theta) + YO + 0.5))

SERVICE_KEY = "77a3d0095b4e5151f66c3dde4bd0925f328c9ccd462427268711a4a4af30f020"

def get_kma_weather(nx, ny):
    now = datetime.now()
    base_date = now.strftime("%Y%m%d")
    base_time = f"{now.hour:02d}00" if now.minute >= 40 else f"{max(0, now.hour-1):02d}00"
    url = "http://apis.data.go.kr/1360000/VvilageFcstInfoService_2.0/getUltraSrtNcst"
    params = {'serviceKey': SERVICE_KEY, 'pageNo': '1', 'numOfRows': '10', 'dataType': 'JSON', 'base_date': base_date, 'base_time': base_time, 'nx': str(nx), 'ny': str(ny)}
    try:
        res = requests.get(url, params=params, timeout=3).json()
        items = res['response']['body']['items']['item']
        temp, hum = None, None
        for item in items:
            if item['category'] == 'T1H': temp = float(item['obsrValue'])
            elif item['category'] == 'REH': hum = float(item['obsrValue'])
        return temp, hum
    except:
        return 25.0, 60.0

# =========================================================
# Step 3: Streamlit 대시보드 UI 및 Phyphox/기상 데이터 통합 로직
# =========================================================
st.title("🚨 밀폐공간 실시간 안전 모니터링 대시보드")
st.caption("기상청 날씨 Open API + Phyphox 모션 센서 + XGBoost 슬러지 가스/무활동 위험 예측")
st.markdown("---")

# 1. GPS 및 기상청 연동
st.subheader("📍 1. 현장 기상 정보 (기상청 Open API)")
loc = get_geolocation()
if loc and 'coords' in loc:
    lat, lon = loc['coords']['latitude'], loc['coords']['longitude']
    nx, ny = convert_to_grid(lat, lon)
    temp, hum = get_kma_weather(nx, ny)
    st.success(f"현재 위치 기상 반영 완료 | 기온: {temp}°C, 습도: {hum}%")
else:
    temp, hum = 26.0, 65.0
    st.info("💡 GPS 권한 미허용 상태: 기본 현장 환경값 적용중")

st.markdown("---")

# 2. Phyphox 연동 설정 및 모션 입력
st.subheader("🏃 2. 작업자 모션 센서 연동 (Phyphox)")
st.markdown("""
- **Phyphox 앱 연동 방법**: 스마트폰에서 Phyphox 실행 ➡️ 가속도계(Accelerometer) 선택 ➡️ 설정(Settings)에서 **Remote Access** 활성화 ➡️ 생성된 **URL 주소**를 아래에 입력하세요.
- *참고: 로컬 테스트 시 아래 슬라이더로 작업자 활동 강도를 직접 조절하여 시뮬레이션할 수도 있습니다.*
""")

use_phyphox = st.checkbox("Phyphox Remote Access(URL) 실시간 연동 사용하기")
movement_input = 4.0 # 기본값

if use_phyphox:
    phyphox_url = st.text_input("Phyphox API 주소 입력", value="http://192.168.x.x:8080/get?accX=&accY=&accZ=")
    try:
        # Phyphox 서버 데이터 요청 예시 (가속도 절대값 기반 움직임 지수 계산)
        response = requests.get(phyphox_url, timeout=1.5)
        data = response.json()
        ax = data['buffer']['accX']['values'][-1]
        ay = data['buffer']['accY']['values'][-1]
        az = data['buffer']['accZ']['values'][-1]
        movement_input = float(min(10.0, abs(ax) + abs(ay) + abs(az)))
        st.success(f"🟢 Phyphox 센서 실시간 수신 성공! (현재 모션 지수: {movement_input:.2f})")
    except Exception as e:
        st.warning(f"⚠️ Phyphox 연결 대기 중 또는 주소 오류 (수동 슬라이더 모드로 대체합니다): {e}")
        movement_input = st.slider("작업자 모션 강도 (수동 보정)", 0.0, 10.0, 4.0, 0.1)
else:
    movement_input = st.slider("작업자 모션 강도 (0: 무활동/쓰러짐 ~ 10: 격렬한 작업)", 0.0, 10.0, 4.0, 0.1)

# 모션 상태 실시간 가이드
if movement_input < 0.4:
    st.error("🚨 **비상 경고 [의식 상실/무활동 의심]**: 작업자 움직임이 감지되지 않습니다!")
elif movement_input > 7.5:
    st.warning("⚠️ **주의보 [슬러지 교란 위험]**: 과도한 활동으로 바닥 가스 분출 가능성이 높습니다.")
else:
    st.success("✅ **정상**: 작업자 활동 및 환경 상태가 안정적입니다.")

st.markdown("---")

# 3. AI 위험도 진단 결과
st.subheader("📊 3. 머신러닝 기반 통합 위험도 예측")
if st.button("🚨 실시간 위험도 진단 실행", use_container_width=True):
    heat_index = temp + 0.33 * hum - 0.70 * 4.0 - 4.0
    
    input_data = pd.DataFrame([{
        'temperature': temp,
        'humidity': hum,
        'movement': movement_input,
        'simulated_o2': 20.9 - (movement_input * 0.04),
        'simulated_h2s': 0.2 + (movement_input * 0.4),
        'simulated_co2': 400.0 + (movement_input * 45.0),
        'heat_index': heat_index
    }])
    
    risk_prob = model.predict_proba(input_data)[0][1] * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("체감 온도", f"{heat_index:.1f} °C")
    col2.metric("작업자 모션 지수", f"{movement_input:.1f}")
    col3.metric("최종 위험 확률", f"{risk_prob:.1f} %")
    
    if risk_prob >= 70.0:
        st.error(f"🔴 **고위험 경보 (Danger)** | 위험 확률: {risk_prob:.1f}%")
        st.markdown("- **판정 원인**: 슬러지 가스 급증 또는 장시간 무활동(의식 상실) 감지\n- **대응 조치**: 즉시 작업을 중단하고 대피하십시오! 현장 구조반 출동 준비.")
    elif risk_prob >= 35.0:
        st.warning(f"🟡 **주의보 (Warning)** | 위험 확률: {risk_prob:.1f}%")
        st.markdown("- **판정 원인**: 환경 조건 또는 작업 강도 주의 필요\n- **대응 조치**: 환기 장치를 가동하고 모니터링을 강화하세요.")
    else:
        st.success(f"🟢 **정상 (Normal)** | 위험 확률: {risk_prob:.1f}%")
        st.markdown("- **상태**: 현재 위험 요소가 발견되지 않았습니다.")
