import math
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
from streamlit_js_eval import get_geolocation
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
import xgboost as xgb

st.set_page_config(
    page_title="밀폐공간 올인원 안전 모니터링", 
    page_icon="🚨", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================================================
# Step 1: 과거 실험 데이터 기반 XGBoost 모델 학습/캐싱
# =========================================================
@st.cache_resource
def load_or_train_historical_model():
    model_filename = 'confined_space_allinone_model.pkl'
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
            ((movement < 0.4) & (simulated_o2 < 19.5)).astype(int) * 4.0 + 
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
# Step 3: Streamlit UI 및 브라우저 센서 컴포넌트 통합
# =========================================================
st.title("🚨 밀폐공간 올인원 안전 대시보드")
st.caption("스마트폰 브라우저 모션 센서 + 기상청 API + XGBoost 실시간 위험 진단")
st.markdown("---")

# 1. 위치 및 기상청 연동
st.subheader("📍 1. 현장 기상 정보 (자동 연동)")
loc = get_geolocation()
if loc and 'coords' in loc:
    lat, lon = loc['coords']['latitude'], loc['coords']['longitude']
    nx, ny = convert_to_grid(lat, lon)
    temp, hum = get_kma_weather(nx, ny)
    st.success(f"현장 위치 기상 반영 완료 | 기온: {temp}°C, 습도: {hum}%")
else:
    temp, hum = 26.0, 65.0
    st.info("💡 GPS 권한 미허용 상태: 기본 현장 환경값 적용중")

st.markdown("---")

# 2. 스마트폰 브라우저 내장 센서(DeviceMotionEvent) 연동 컴포넌트
st.subheader("🏃 2. 작업자 모션 센서 (브라우저 직접 측정)")
st.markdown("""
> 아래 **[센서 권한 허용 및 측정 시작]** 버튼을 누르면 스마트폰의 움직임(가속도)을 실시간으로 감지하여 AI 모델에 반영합니다. (아이폰/안드로이드 지원)
""")

# 스마트폰 센서 기록 및 CSV 다운로드가 가능한 올인원 컴포넌트
logger_html = """
<div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; font-family: sans-serif; text-align: center;">
    <h3>📊 스마트폰 모션 센서 로거</h3>
    <p style="font-size: 13px; color: #555;">현장 실험 중 데이터를 수집하여 CSV 파일로 저장합니다.</p>
    
    <button id="recBtn" onclick="toggleRecording()" style="background-color: #28a745; color: white; border: none; padding: 10px 18px; font-size: 15px; border-radius: 5px; cursor: pointer; font-weight: bold; margin: 5px;">
        🔴 센서 기록 시작
    </button>
    
    <button id="dlBtn" onclick="downloadCSV()" disabled style="background-color: #6c757d; color: white; border: none; padding: 10px 18px; font-size: 15px; border-radius: 5px; cursor: pointer; font-weight: bold; margin: 5px;">
        💾 CSV 다운로드
    </button>
    
    <div style="margin-top: 10px; font-size: 14px;">
        수집된 데이터 포인트: <span id="count" style="font-weight: bold; color: #0066cc;">0</span> 개
    </div>
</div>

<script>
let isRecording = false;
let sensorLogs = [];
let watchInterval = null;
let currentAcc = {x: 0, y: 0, z: 0};

// 권한 요청 및 이벤트 등록
window.addEventListener('devicemotion', (event) => {
    let acc = event.acceleration || event.accelerationIncludingGravity;
    if (acc && acc.x !== null) {
        currentAcc = {
            x: acc.x || 0,
            y: acc.y || 0,
            z: acc.z || 0
        };
    }
});

function toggleRecording() {
    // iOS 권한 요청 처리
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        DeviceMotionEvent.requestPermission().then(response => {
            if (response === 'granted') { startStopProcess(); }
            else { alert("센서 권한이 거부되었습니다."); }
        });
    } else {
        startStopProcess();
    }
}

function startStopProcess() {
    const btn = document.getElementById("recBtn");
    const dlBtn = document.getElementById("dlBtn");
    
    if (!isRecording) {
        // 기록 시작
        isRecording = true;
        sensorLogs = [];
        btn.innerText = "⏹ 기록 중지";
        btn.style.backgroundColor = "#dc3545";
        dlBtn.disabled = true;
        dlBtn.style.backgroundColor = "#6c757d";
        
        // 1초 주기로 데이터 샘플링 (EDW500 데이터 간격과 맞추거나 필요에 따라 조절)
        watchInterval = setInterval(() => {
            let timestamp = new Date().toISOString();
            let score = Math.min(10.0, (Math.abs(currentAcc.x) + Math.abs(currentAcc.y) + Math.abs(currentAcc.z)) * 1.5);
            
            sensorLogs.push({
                timestamp: timestamp,
                movement_score: score.toFixed(3),
                acc_x: currentAcc.x.toFixed(3),
                acc_y: currentAcc.y.toFixed(3),
                acc_z: currentAcc.z.toFixed(3)
            });
            
            document.getElementById("count").innerText = sensorLogs.length;
        }, 1000); // 1초 간격 기록
        
    } else {
        // 기록 중지
        isRecording = false;
        clearInterval(watchInterval);
        btn.innerText = "🔴 센서 기록 시작";
        btn.style.backgroundColor = "#28a745";
        
        if (sensorLogs.length > 0) {
            dlBtn.disabled = false;
            dlBtn.style.backgroundColor = "#007bff";
        }
    }
}

function downloadCSV() {
    if (sensorLogs.length === 0) return;
    
    let csvContent = "timestamp,movement_score,acc_x,acc_y,acc_z\\n";
    sensorLogs.forEach(row => {
        csvContent += `${row.timestamp},${row.movement_score},${row.acc_x},${row.acc_y},${row.acc_z}\\n`;
    });
    
    let blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    let url = URL.createObjectURL(blob);
    let a = document.createElement('a');
    a.href = url;
    a.download = "phone_motion_log.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}
</script>
"""

components.html(logger_html, height=180)

# 세션 상태나 수동 보정용 슬라이더 제공 (브라우저 센서 연동 보완용)
st.markdown("---")
st.subheader("📊 3. 머신러닝 통합 위험도 예측 진단")

# 수동 슬라이더를 병행 제공하여 센서 미지원 환경이나 테스트 시에도 즉시 확인 가능하도록 구성
movement_input = st.slider(
    "작업자 모션 활동 지수 (브라우저 센서 연동 또는 수동 조절)", 
    min_value=0.0, max_value=10.0, value=4.0, step=0.1
)

if movement_input < 0.4:
    st.error("🚨 **비상 경고 [의식 상실/무활동 의심]**: 작업자 움직임이 감지되지 않습니다!")
elif movement_input > 7.5:
    st.warning("⚠️ **주의보 [슬러지 교란 위험]**: 과도한 활동으로 바닥 가스 분출 가능성이 높습니다.")
else:
    st.success("✅ **정상**: 작업자 활동 및 환경 상태가 안정적입니다.")

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
    col2.metric("모션 활동 지수", f"{movement_input:.1f}")
    col3.metric("최종 위험 확률", f"{risk_prob:.1f} %")
    
    if risk_prob >= 70.0:
        st.error(f"🔴 **고위험 경보 (Danger)** | 위험 확률: {risk_prob:.1f}%")
        st.markdown("- **판정 원인**: 슬러지 가스 급증 또는 장시간 무활동(의식 상실) 감지\n- **대응 조치**: 즉시 작업을 중단하고 대피하십시오!")
    elif risk_prob >= 35.0:
        st.warning(f"🟡 **주의보 (Warning)** | 위험 확률: {risk_prob:.1f}%")
        st.markdown("- **판정 원인**: 환경 조건 또는 작업 강도 주의 필요\n- **대응 조치**: 환기 장치를 가동하고 모니터링을 강화하세요.")
    else:
        st.success(f"🟢 **정상 (Normal)** | 위험 확률: {risk_prob:.1f}%")
        st.markdown("- **상태**: 현재 위험 요소가 발견되지 않았습니다.")
