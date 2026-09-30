import streamlit as st

# 1. 대시보드 제목 쓰기
st.title("🛡️ 밀폐공간 안전 모니터링 대시보드")
st.write("슬라이더를 조절하여 실시간 안전 점수를 확인하세요.")

st.divider() # 구분선

# 2. 화면을 2개의 구역으로 나누기
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ 현장 데이터 입력")
    # 슬라이더(바를 드래그해서 수치 조절) 만들기
    o2 = st.slider("산소 농도 (%)", min_value=10.0, max_value=25.0, value=20.9, step=0.1)
    co = st.slider("일산화탄소 (ppm)", min_value=0, max_value=100, value=5)
    
    # 체크박스 만들기
    fan = st.checkbox("환기 팬 가동 중", value=True)
    attendant = st.checkbox("외부 감시인 배치 완료", value=True)

# 3. 안전 점수 계산 로직 (간단 버전)
score = 100

# 가스 농도에 따른 감점
if o2 < 18.0 or o2 > 23.5:
    score -= 40
if co >= 30:
    score -= 30

# 환경 및 절차에 따른 감점
if not fan:
    score -= 15
if not attendant:
    score -= 15

# 4. 결과 출력하기
with col2:
    st.subheader("📊 안전 평가 결과")
    st.metric(label="현재 안전 점수", value=f"{score} / 100점")

    if score >= 90:
        st.success("🟢 작업 가능 (안전한 상태입니다)")
    elif score >= 70:
        st.warning("🟡 주의 필요 (환기 및 환경을 점검하세요)")
    else:
        st.error("🔴 즉시 대피 (위험 상태입니다)")