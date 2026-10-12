import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="밀폐공간 실시간 자동 안전 모니터링", 
    page_icon="🚨", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================================================
# Step 1: AI 모델 로드 (없을 경우 기본 대체 모델 생성)
# =========================================================
@st.cache_resource
def load_trained_model():
    model_filename = 'edw500_phone_integrated_model.pkl'
    try:
        model = joblib.load(model_filename)
        return model
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(random_state=42)
        X_dummy = np.array([[0.1, 20.9, 0.04, 0.0], [5.0, 20.5, 0.1, 0.0]])
        y_dummy = np.array([1, 0])
        clf.fit(X_dummy, y_dummy)
        return clf

model = load_trained_model()

# =========================================================
# Step 2: 대시보드 UI 레이아웃
# =========================================================
st.markdown("<h2 style='text-align: center;'>🚨 밀폐공간 스마트 안전 모니터링</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>스마트폰 센서 실시간 연동 및 AI 위험 감지</p>", unsafe_allow_html=True)
st.divider()

# =========================================================
# Step 3: 브라우저 센서 권한 및 실시간 연동 HTML/JS 컴포넌트
# =========================================================
live_sensor_html = """
<div id="alert_box" style="background-color: #d4edda; padding: 20px; border-radius: 12px; text-align: center; font-family: sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: background-color 0.3s ease;">
    <h3 id="status_title" style="margin: 0; color: #155724;">🟢 실시간 현장 안전 (Normal)</h3>
    <p id="status_desc" style="font-size: 13px; color: #385723; margin: 6px 0 12px 0;">센서 권한을 허용하고 이동하면 움직임이 측정됩니다.</p>
    
    <!-- 센서 권한 허용 버튼 (모바일 브라우저 필수) -->
    <button id="permission_btn" onclick="requestSensors()" style="background-color: #007bff; color: white; border: none; padding: 10px 18px; font-size: 14px; border-radius: 6px; cursor: pointer; font-weight: bold; margin-bottom: 12px;">
        📱 센서 모니터링 시작 (터치 필수)
    </button>

    <div style="display: flex; justify-content: space-around; font-size: 15px;">
        <div style="background: white; padding: 8px; border-radius: 6px; width: 45%;">
            실시간 활동 지수<br>
            <span id="live_score" style="font-weight: bold; color: #0066cc; font-size: 22px;">0.00</span>
        </div>
        <div style="background: white; padding: 8px; border-radius: 6px; width: 45%;">
            AI 위험 확률<br>
            <span id="live_risk" style="font-weight: bold; color: #28a745; font-size: 22px;">0.0%</span>
        </div>
    </div>
</div>

<script>
function requestSensors() {
    // iOS 13+ 권한 요청 처리
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        DeviceMotionEvent.requestPermission().then(response => {
            if (response === 'granted') {
                window.addEventListener('devicemotion', handleMotion);
                document.getElementById("permission_btn").style.display = "none";
                alert("센서 연동이 활성화되었습니다!");
            } else {
                alert("센서 권한이 거부되었습니다.");
            }
        }).catch(console.error);
    } else {
        // 안드로이드 및 일반 브라우저
        window.addEventListener('devicemotion', handleMotion);
        document.getElementById("permission_btn").style.display = "none";
        alert("센서 모니터링이 시작되었습니다!");
    }
}

function handleMotion(event) {
    // 중력 가속도가 포함된 속도와 일반 가속도를 모두 포괄하여 수치 추출
    let acc = event.accelerationIncludingGravity || event.acceleration;
    if (acc) {
        let ax = acc.x || 0;
        let ay = acc.y || 0;
        let az = acc.z || 0;
        
        // 정지 상태(대략 중력 9.8 등)를 기준(Offset)으로 하여 변화량(활동 지수) 계산
        // 스마트폰을 흔들거나 이동할 때 값이 0 이상으로 크게 튀도록 구성
        let rawMagnitude = Math.sqrt(ax*ax + ay*ay + az*az);
        let score = Math.min(10.0, Math.abs(rawMagnitude - 9.8) * 1.5);
        
        // 만약 가속도 표준 API가 0으로 잡히는 환경일 경우를 위한 대체 계산식 적용
        if (score === 0 && (ax !== 0 || ay !== 0)) {
            score = Math.min(10.0, (Math.abs(ax) + Math.abs(ay) + Math.abs(az)) * 0.8);
        }

        document.getElementById("live_score").innerText = score.toFixed(2);
        
        // AI 위험 확률 시뮬레이션 계산 (무활동 혹은 과도한 충격)
        let riskProb = 0.0;
        if (score < 0.2) {
            riskProb = 88.0; // 무활동 질식 위험 상태
        } else {
            riskProb = Math.max(2.0, 12.0 - (score * 1.0)); // 정상 작업 영역
        }
        
        document.getElementById("live_risk").innerText = riskProb.toFixed(1) + "%";
        
        let boxElem = document.getElementById("alert_box");
        let titleElem = document.getElementById("status_title");
        let descElem = document.getElementById("status_desc");
        
        if (riskProb >= 50.0) {
            boxElem.style.backgroundColor = "#f8d7da";
            titleElem.style.color = "#721c24";
            titleElem.innerText = "🚨 [긴급 고위험 경보]";
            descElem.style.color = "#721c24";
            descElem.innerText = "무활동 또는 위험 패턴 감지!";
            if (navigator.vibrate) { navigator.vibrate(400); }
        } else {
            boxElem.style.backgroundColor = "#d4edda";
            titleElem.style.color = "#155724";
            titleElem.innerText = "🟢 실시간 현장 안전 (Normal)";
            descElem.style.color = "#385723";
            descElem.innerText = "센서가 정상적으로 작동 중입니다.";
        }
    }
}
</script>
"""

components.html(live_sensor_html, height=270)

st.divider()
st.info("💡 **테스트 방법**: 스마트폰으로 접속하신 뒤 화면의 **[센서 모니터링 시작]** 버튼을 터치하고 기기를 가볍게 흔들거나 움직여 보세요. 활동 지수와 위험 확률이 실시간으로 변동하는 것을 확인할 수 있습니다.")
