import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정 및 스타일링
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="밀폐공간 안전 모니터링 시스템",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ 밀폐공간 작업 실시간 안전 모니터링 대시보드")
st.caption("수동 입력 매개변수 기반 종합 안전 지수 시각화 및 위험 요소 진단 시스템")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 수동 입력 제어판 (가로바 슬라이더 & 체크박스/토글)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 현장 측정 데이터 수동 입력")
st.sidebar.info("💡 현재 센서 미연동 상태로, 현장 점검 수치를 직접 조정하여 평가합니다.")

# [카테고리 1] 가스 농도 (배점 40점) - 가로바 슬라이더
st.sidebar.subheader("1. 가스 농도 측정 (슬라이더)")
o2_val = st.sidebar.slider("산소 (O₂)(%)", 10.0, 25.0, 20.9, 0.1)
co_val = st.sidebar.slider("일산화탄소 (CO)(ppm)", 0, 100, 5, 1)
h2s_val = st.sidebar.slider("황화수소 (H₂S)(ppm)", 0, 50, 0, 1)
lel_val = st.sidebar.slider("가연성가스 (LEL)(%)", 0, 50, 0, 1)

# [카테고리 2] 환기 설비 상태 - 토글 및 가로바
st.sidebar.subheader("2. 환기 설비 상태")
fan_on = st.sidebar.toggle("송풍기(환기팬) 가동 여부", value=True)
flow_rate = st.sidebar.slider("환기 풍량 (m³/h)", 0, 2000, 1500, 50)

# [카테고리 3] 작업장 환경 조건 - 가로바 슬라이더
st.sidebar.subheader("3. 작업장 환경 조건")
temp_val = st.sidebar.slider("작업장 내부 온도 (°C)", 0, 45, 24, 1)
hum_val = st.sidebar.slider("작업장 내부 습도 (%)", 20, 100, 55, 1)
lux_val = st.sidebar.slider("작업장 내부 조도 (lx)", 0, 300, 120, 5)
work_min = st.sidebar.slider("연속 작업 시간 (분)", 0, 300, 45, 5)

# [카테고리 4] 안전 절차 및 점검 - 체크박스
st.sidebar.subheader("4. 안전 절차 준수 여부 (체크)")
chk_attendant = st.sidebar.checkbox("외부 감시인(구조원) 배치 완료", value=True)
chk_comm = st.sidebar.checkbox("비상 무전/통신 상태 정상", value=True)
chk_detector = st.sidebar.checkbox("개인용 가스 측정기 착용", value=True)

# -----------------------------------------------------------------------------
# 3. 안전 점수 계산 알고리즘 엔진
# -----------------------------------------------------------------------------
def calculate_safety_score():
    gas_score = 40.0
    gas_critical = False
    critical_reasons = []

    # 가스 농도 평가
    if not (18.0 <= o2_val <= 23.5):
        gas_score -= 20
        gas_critical = True
        critical_reasons.append(f"산소 농도 부적합 ({o2_val}%)")
    
    if co_val >= 30:
        gas_score -= 15
        gas_critical = True
        critical_reasons.append(f"일산화탄소 위험 수준 ({co_val} ppm)")
    elif co_val > 15:
        gas_score -= 7

    if h2s_val >= 10:
        gas_score -= 15
        gas_critical = True
        critical_reasons.append(f"황화수소 위험 수준 ({h2s_val} ppm)")
    elif h2s_val > 5:
        gas_score -= 7

    if lel_val >= 10:
        gas_score -= 15
        gas_critical = True
        critical_reasons.append(f"가연성 가스 위험 수준 ({lel_val}% LEL)")
    elif lel_val > 5:
        gas_score -= 7

    gas_score = max(0.0, gas_score)

    # 환기 평가
    vent_score = 0.0
    if fan_on:
        vent_score += 10.0
        if flow_rate >= 1200:
            vent_score += 15.0
        elif flow_rate >= 800:
            vent_score += 8.0
    else:
        critical_reasons.append("환기팬 미가동 상태")

    # 작업 환경 평가
    env_score = 20.0
    if not (15 <= temp_val <= 28):
        env_score -= 5.0
    if hum_val > 80:
        env_score -= 5.0
    if lux_val < 75:
        env_score -= 5.0
    if work_min > 120:
        env_score -= 5.0
    env_score = max(0.0, env_score)

    # 안전 절차 평가
    proc_score = 0.0
    if chk_attendant:
        proc_score += 5.0
    else:
        critical_reasons.append("외부 감시인 미배치")
        
    if chk_comm:
        proc_score += 5.0
    if chk_detector:
        proc_score += 5.0

    total_score = gas_score + vent_score + env_score + proc_score

    # 가스 중대 위험 발생 시 45점 이하로 제한
    if gas_critical:
        total_score = min(total_score, 45.0)

    return total_score, gas_score, vent_score, env_score, proc_score, critical_reasons

total_score, gas_score, vent_score, env_score, proc_score, critical_reasons = calculate_safety_score()

# -----------------------------------------------------------------------------
# 4. 대시보드 상단 - 종합 안전 점수 게이지 시각화 & 비상 경보
# -----------------------------------------------------------------------------
col_gauge, col_info = st.columns([1.2, 1.8])

with col_gauge:
    st.subheader("🎯 종합 안전 지수 시각화")

    # 상태별 게이지 색상 지정
    if total_score >= 90 and not critical_reasons:
        gauge_color = "#2ECC71"  # 녹색
        status_text = "안전 (Safe)"
    elif total_score >= 70 and not critical_reasons:
        gauge_color = "#F1C40F"  # 노란색
        status_text = "주의 (Caution)"
    else:
        gauge_color = "#E74C3C"  # 빨간색
        status_text = "위험 (Danger)"

    # Plotly Donut Gauge Chart 생성
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=total_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{status_text}</b>", 'font': {'size': 20}},
        number={'suffix': "점", 'font': {'size': 36}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': gauge_color, 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [0, 70], 'color': '#FDEDEC'},
                {'range': [70, 90], 'color': '#FEF9E7'},
                {'range': [90, 100], 'color': '#EAFAF1'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': total_score
            }
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_info:
    st.subheader("⚠️ 실시간 안전 진단 및 경보")
    
    if total_score >= 90 and not critical_reasons:
        st.success("🟢 **작업 진행 가능**: 모든 가스 농도 및 작업 환경 기준이 적합합니다.")
    elif total_score >= 70 and not critical_reasons:
        st.warning("🟡 **주의 필요**: 환기량을 증대하고 안전 관리자의 지속 모니터링이 필요합니다.")
    else:
        st.error("🔴 **즉시 작업 중단 및 대피**: 위험 수준의 가스 감지 또는 주요 절차 미준수!")

    if critical_reasons:
        st.markdown("##### 🚨 감지된 주요 위험 요인:")
        for reason in critical_reasons:
            st.error(f"• {reason}")
    else:
        st.info("✅ 현재 입력된 가스 농도 및 안전 절차상 중대한 감점 요인이 없습니다.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. 수동 입력 수치 시각화 (4대 카테고리)
# -----------------------------------------------------------------------------
st.subheader("📊 항목별 세부 수치 및 평가 점수")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("#### 1. 가스 농도")
    st.metric(label="평가 점수", value=f"{gas_score:.0f} / 40점")
    st.write(f"• 산소(O₂): **{o2_val}%**")
    st.write(f"• 일산화탄소(CO): **{co_val} ppm**")
    st.write(f"• 황화수소(H₂S): **{h2s_val} ppm**")
    st.write(f"• 가연성가스(LEL): **{lel_val}%**")

with col2:
    st.markdown("#### 2. 환기 상태")
    st.metric(label="평가 점수", value=f"{vent_score:.0f} / 25점")
    st.write(f"• 팬 가동: **{'가동 중 (ON)' if fan_on else '중지 (OFF)'}**")
    st.write(f"• 환기 풍량: **{flow_rate} m³/h**")

with col3:
    st.markdown("#### 3. 작업 환경")
    st.metric(label="평가 점수", value=f"{env_score:.0f} / 20점")
    st.write(f"• 온도 / 습도: **{temp_val}°C** / **{hum_val}%**")
    st.write(f"• 작업장 조도: **{lux_val} lx**")
    st.write(f"• 연속 작업: **{work_min}분**")

with col4:
    st.markdown("#### 4. 안전 절차")
    st.metric(label="평가 점수", value=f"{proc_score:.0f} / 15점")
    st.write(f"• 감시인 배치: **{'완료' if chk_attendant else '미배치'}**")
    st.write(f"• 통신 상태: **{'정상' if chk_comm else '불가'}**")
    st.write(f"• 가스 측정기: **{'착용' if chk_detector else '미착용'}**")

# 가시적인 수평 바 차트 비교
chart_df = pd.DataFrame({
    "평가 항목": ["가스 농도", "환기 상태", "작업 환경", "안전 절차"],
    "취득 점수": [gas_score, vent_score, env_score, proc_score],
    "만점 기준": [40, 25, 20, 15]
})

st.markdown("<br>", unsafe_allow_html=True)
st.bar_chart(chart_df, x="평가 항목", y=["취득 점수", "만점 기준"], stack=False)
