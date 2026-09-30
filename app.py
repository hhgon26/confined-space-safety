import streamlit as st
import pandas as pd
import numpy as np
import time

# -----------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="밀폐공간 안전 모니터링 대시보드",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ 밀폐공간 작업 실시간 안전 모니터링 시스템")
st.caption("가스 농도, 환기, 환경 변수 및 안전 절차 상태를 실시간 종합 평가합니다.")
st.markdown("---")

# -----------------------------------------------------------------------------
# 2. 사이드바 - 실시간 매개변수 입력 제어판
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 현장 측정 데이터 입력")

# [카테고리 1] 가스 농도 (배점 40점)
st.sidebar.subheader("1. 가스 농도 측정")
o2_val = st.sidebar.slider("산소 (O₂)(%)", 10.0, 25.0, 20.9, 0.1)
co_val = st.sidebar.slider("일산화탄소 (CO)(ppm)", 0, 100, 5, 1)
h2s_val = st.sidebar.slider("황화수소 (H₂S)(ppm)", 0, 50, 0, 1)
lel_val = st.sidebar.slider("가연성가스 (LEL)(%)", 0, 50, 0, 1)

# [카테고리 2] 환기 및 유량 상태 (배점 25점)
st.sidebar.subheader("2. 환기 설비 상태")
fan_on = st.sidebar.toggle("송풍기(환기팬) 가동 여부", value=True)
flow_rate = st.sidebar.slider("환기 풍량 (m³/h)", 0, 2000, 1500, 50)

# [카테고리 3] 작업 환경 조건 (배점 20점)
st.sidebar.subheader("3. 작업장 환경 조건")
temp_val = st.sidebar.slider("작업장 내부 온도 (°C)", 0, 45, 24, 1)
hum_val = st.sidebar.slider("작업장 내부 습도 (%)", 20, 100, 55, 1)
lux_val = st.sidebar.slider("작업장 내부 조도 (lx)", 0, 300, 120, 5)
work_min = st.sidebar.number_input("연속 작업 시간 (분)", 0, 300, 45, 5)

# [카테고리 4] 안전 절차 및 점검 (배점 15점)
st.sidebar.subheader("4. 안전 절차 준수 여부")
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

    # 가스 농도 평가 (필수 안전 수치 초과 시 0점 및 즉시 대피)
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

    # 가스 중대 위험 발생 시 총점 50점 이하로 강제 하향 조정
    if gas_critical:
        total_score = min(total_score, 45.0)

    return total_score, gas_score, vent_score, env_score, proc_score, critical_reasons

# 점수 계산 실행
total_score, gas_score, vent_score, env_score, proc_score, critical_reasons = calculate_safety_score()

# -----------------------------------------------------------------------------
# 4. 대시보드 상단 - 종합 상태 및 메트릭
# -----------------------------------------------------------------------------
col_status1, col_status2 = st.columns([1, 2])

with col_status1:
    st.subheader("종합 안전 점수")
    st.metric(label="현재 안전 지수", value=f"{total_score:.0f} / 100점")

    if total_score >= 90 and not critical_reasons:
        st.success("🟢 **안전 (Safe)** - 작업 진행 가능")
    elif total_score >= 70 and not critical_reasons:
        st.warning("🟡 **주의 (Caution)** - 환기 강화 및 점검 필요")
    else:
        st.error("🔴 **위험 (Danger)** - 즉시 작업 중단 및 대피")

with col_status2:
    st.subheader("⚠️ 실시간 주요 점검/경보 사항")
    if critical_reasons:
        for reason in critical_reasons:
            st.error(f"🚨 **위험 요인:** {reason}")
    else:
        st.info("✅ 현재 감지된 가스/절차상의 중대한 감점 요인이 없습니다.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 5. 세부 항목별 상태 시각화
# -----------------------------------------------------------------------------
st.subheader("📊 카테고리별 세부 평가 점수")

col_g1, col_g2, col_g3, col_g4 = st.columns(4)

with col_g1:
    st.metric(label="가스 농도 (40점)", value=f"{gas_score:.0f}점")
    st.caption(f"O₂: {o2_val}% | CO: {co_val}ppm")
    st.caption(f"H₂S: {h2s_val}ppm | LEL: {lel_val}%")

with col_g2:
    st.metric(label="환기 상태 (25점)", value=f"{vent_score:.0f}점")
    st.caption(f"가동 여부: {'ON' if fan_on else 'OFF'}")
    st.caption(f"풍량: {flow_rate} m³/h")

with col_g3:
    st.metric(label="작업 환경 (20점)", value=f"{env_score:.0f}점")
    st.caption(f"온도: {temp_val}°C | 습도: {hum_val}%")
    st.caption(f"조도: {lux_val} lx | 작업시간: {work_min}분")

with col_g4:
    st.metric(label="안전 절차 (15점)", value=f"{proc_score:.0f}점")
    st.caption(f"감시인: {'완료' if chk_attendant else '미배치'}")
    st.caption(f"통신: {'정상' if chk_comm else '불가'}")

# 차트용 데이터 프레임 구축
score_df = pd.DataFrame({
    "평가 항목": ["가스 농도", "환기 상태", "작업 환경", "안전 절차"],
    "취득 점수": [gas_score, vent_score, env_score, proc_score],
    "만점 기준": [40, 25, 20, 15]
})

st.bar_chart(score_df, x="평가 항목", y=["취득 점수", "만점 기준"], stack=False)

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. 실시간 가스 농도 트렌드 시뮬레이션
# -----------------------------------------------------------------------------
st.subheader("📈 실시간 모니터링 시뮬레이션 추이 (최근 1분)")

# 임의의 시뮬레이션 데이터 생성 (현재 입력 수치를 기준으로 약간의 변동 부여)
np.random.seed(42)
time_steps = pd.date_range(end=pd.Timestamp.now(), periods=20, freq='3s')

sim_data = pd.DataFrame({
    '시간': time_steps,
    '산소 O₂ (%)': o2_val + np.random.normal(0, 0.1, 20),
    '일산화탄소 CO (ppm)': np.clip(co_val + np.random.normal(0, 0.5, 20), 0, None),
    '황화수소 H₂S (ppm)': np.clip(h2s_val + np.random.normal(0, 0.2, 20), 0, None)
}).set_index('시간')

st.line_chart(sim_data)
