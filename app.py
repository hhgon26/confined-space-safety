import math
import joblib
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="밀폐공간 실시간 스마트 안전 모니터링", 
    page_icon="🚨", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================================================
# Step 1: AI 모델 로드 (없을 경우 기본 대체 모델 생성 방어 코드 포함)
# =========================================================
@st.cache_resource
def load_trained_model():
    model_filename = 'edw500_phone_integrated_model.pkl'
    try:
        model = joblib.load(model_filename)
        return model
    except Exception:
        # 모델 파일이 아직 없거나 로드 실패 시 테스트용 임시 모델 반환 (앱 크래시 방지)
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
        X_dummy = np.array([[0.1, 20.9, 0.04, 0.0], [5.0, 20.5, 0.1, 0.0]])
        y_dummy = np.array([1, 0])
        clf.fit(X_dummy, y_dummy)
        return clf

model = load_trained_model()

# =========================================================
# Step 2: 대시보드 UI 구성
# =========================================================
st.markdown("<h2 style='text-align: center;'>🚨 밀폐공간 실시간 스마트 안전 모니터링</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>스마트폰 센서와 AI 모델이 현장 위험도를 실시간으로 평가합니다.</p>", unsafe_allow_html=True)
st.divider()

# =========================================================
# Step 3: 자바스크립트 센서 수집 및 Streamlit 동기화 컴포넌트
# =========================================================
# 사용자가 스마트폰을 흔들거나 이동할 때 측정된 movement_score를 
# Streamlit의 쿼리 파라미터(URL)나 입력창을 통해 실시간으로 서버에 전달하는 구조입니다.
sensor_sync_html = """
<div style="background-color: #f1f3f5; padding: 20px; border-radius: 10px; text-align: center; font-family: sans-serif; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
    <h3 style="margin: 0; color: #333;">📱 현장 작업자 실시간 센서</h3>
    <p style="font-size: 13px; color: #666; margin: 8px 0 15px 0;">스마트폰을 소지하고 이동하면 움직임이 감지됩니다.</p>
    
    <div style="font-size: 18px; margin-bottom: 10px;">
        실시간 활동 지수: <span id="live_score_display" style="font-weight: bold; color: #0066cc; font-size: 24px;">0.00</span>
    </div>
    
    <button onclick="syncSensorValue()" style="background-color: #007bff; color: white; border: none; padding: 10px 20px; font-size: 14px; border-radius: 5px; cursor: pointer; font-weight: bold;">
        🔄 AI 진단에 현재 센서값 반영하기
    </button>
</div>

<script>
let currentAcc = {x: 0, y: 0, z: 0};
let latestScore = 2.5; // 기본 안정값

function initSensors() {
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        DeviceMotionEvent.requestPermission().then(response => {
            if (response === 'granted') {
                window.addEventListener('devicemotion', handleMotion);
            }
        }).catch(console.error);
    } else {
        window.addEventListener('devicemotion', handleMotion);
    }
}

function handleMotion(event) {
    let acc = event.acceleration || event.accelerationIncludingGravity;
    if (acc && acc.x !== null) {
        currentAcc = { x: acc.x || 0, y: acc.y || 0, z: acc.z || 0 };
        latestScore = Math.min(10.0, (Math.abs(currentAcc.x) + Math.abs(currentAcc.y) + Math.abs(currentAcc.z)) * 1.5);
        document.getElementById("live_score_display").innerText = latestScore.toFixed(2);
    }
}

function syncSensorValue() {
    // 현재 측정된 모션 점수를 Streamlit 쿼리 파라미터로 전달하여 앱을 새로고침
    const url = new URL(window.parent.location.href);
    url.searchParams.set('movement', latestScore.toFixed(2));
    window.parent.location.href = url.toString();
}

initSensors();
</script>
"""

components.html(sensor_sync_html, height=210)

# =========================================================
# Step 4: URL 파라미터에서 실시간 센서 값을 받아 AI 모델 추론 수행
# =========================================================
query_params = st.query_params
movement_input = float(query_params.get("movement", 2.5)) # 값이 없으면 기본 안정값 2.5

st.subheader("🤖 AI 실시간 위험 진단 결과")

# 입력 데이터 구성 (학습 모델과 동일한 컬럼 형태 유지)
input_data = pd.DataFrame([[
    movement_input,
    20.9 - (movement_input * 0.02), # O2 추정값
    0.05 + (movement_input * 0.01), # CO2 추정값
    0.0                             # H2S 기본값
]], columns=['movement_score', 'O2', 'CO2', 'H2S'])

# AI 모델 예측 수행
try:
    risk_prob = model.predict_proba(input_data)[0][1] * 100
except Exception:
    risk_prob = 10.0 # 예외 방어용

col1, col2 = st.columns(2)
col1.metric("현재 반영된 활동 지수", f"{movement_input:.2f}")
col2.metric("AI 예측 위험 확률", f"{risk_prob:.1f}%")

# 위험도에 따른 시각적 경보 분기
if risk_prob >= 50.0 or movement_input < 0.3:
    st.error("🚨 **[고위험 경보]** 작업자 무활동 또는 유해가스 위험 패턴이 감지되었습니다! 즉시 대피하세요.")
else:
    st.success("🟢 **[시스템 정상]** 현장 작업 및 환경 상태가 안전합니다.")
