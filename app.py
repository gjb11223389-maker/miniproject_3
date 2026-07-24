import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 환경변수 로드 (.env 파일이 있는 경우)
load_dotenv()

# 2. Streamlit 페이지 기본 설정
st.set_page_config(
    page_title="Global Intellect - 다국어 기사 분석 및 질의응답",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Global Intellect: 다국어 기사 분석 및 QA 챗봇")
st.markdown("""
해외 주요 뉴스 기사나 논문 텍스트를 붙여넣고 분석 버튼을 누른 뒤, **궁금한 점을 자유롭게 질문하세요.**  
RAG 검색 기술을 바탕으로 원문을 참고하여 한국어로 답변합니다.
""")

# 3. 안전한 API 키 불러오기 함수
def get_openai_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")

# 4. 사이드바: API 키 입력 및 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    api_key_default = get_openai_api_key()
    
    openai_api_key = st.text_input(
        "OpenAI API Key",
        value=api_key_default,
        type="password"
    )
    
    # 세션 상태 초기화 버튼
    if st.button("🔄 대화 내용 및 분석 데이터 초기화"):
        st.session_state.clear()
        st.rerun()

if not openai_api_key:
    st.warning("⚠️ 서비스 사용을 위해 OpenAI API Key를 입력하거나 Secrets에 등록해주세요.")
    st.stop()

# 5. 기사 텍스트 직접 입력 영역
st.subheader("📝 기사/논문 텍스트 입력")

article_title = st.text_input("기사 제목 또는 출처 (선택 입력)", placeholder="예: [BBC] Global AI Trends")
article_text = st.text_area(
    "분석할 기사 텍스트를 여기에 붙여넣으세요:",
    height=200,
    placeholder="여기에 외국어 뉴스 기사나 논문 텍스트를 전체 복사하여 입력하세요..."
)

btn_analyze = st.button("🚀 기사 분석 및 DB 구축", type="primary")

if btn_analyze:
    if not article_text.strip():
        st.warning("⚠️ 분석할 텍스트를 입력해주세요!")
    else:
        with st.spinner("📄 입력한 텍스트 청크 분할 및 FAISS 데이터베이스 구축 중..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            
            # 텍스트 청크 생성
            chunks = text_splitter.split_text(article_text)
            source_name = article_title.strip() if article_title.strip() else "입력된 기사"
            all_docs = [f"[출처: {source_name}]\n{chunk}" for chunk in chunks]
            
            # 임베딩 모델 생성 및 벡터DB 구축
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=openai_api_key
            )
            vectorstore = FAISS.from_texts(all_docs, embeddings)
            
            # 세션에 벡터스토어 저장
            st.session_state["vectorstore"] = vectorstore
            st.session_state["article_loaded"] = True
            st.success(f"✅ 기사 분석 완료! (총 {len(all_docs)}개 텍스트 블록 생성) 이제 아래에서 질문해보세요.")

st.divider()

# 6. 기사 질의응답 (QA 챗봇) 영역
st.subheader("💬 기사 기반 질의응답")

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화 내용 출력
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 질문 입력
if user_query := st.chat_input("기사 내용 중 궁금한 점을 질문하세요..."):
    # 1) 입력된 기사가 없는 경우 방어
    if "vectorstore" not in st.session_state:
        st.error("⚠️ 먼저 위 상단에서 기사 텍스트를 입력하고 '🚀 기사 분석 및 DB 구축' 버튼을 눌러주세요.")
    else:
        # 2) 질문을 채팅 화면에 표시 및 기록
        st.chat_message("user").markdown(user_query)
        st.session_state["messages"].append({"role": "user", "content": user_query})
        
        # 3) 질문에 대한 답변 생성
        with st.chat_message("assistant"):
            with st.spinner("기사 내용을 바탕으로 답변 생성 중..."):
                vectorstore = st.session_state["vectorstore"]
                
                # 상위 4개 관련 청크 검색
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                retrieved_docs = retriever.invoke(user_query)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # Prompt 템플릿 작성
                prompt = ChatPromptTemplate.from_template("""
당신은 다국어 뉴스 기사 분석 전문가입니다.
아래에 제공된 [기사 참고 내용]만을 바탕으로 사용자의 질문에 한국어로 명확하고 친절하게 답변해주세요.
만약 [기사 참고 내용]에서 질문에 대한 답을 찾을 수 없다면 지어내지 말고 "제시된 기사 내용에서는 해당 정보를 찾을 수 없습니다."라고 답변해주세요.

[기사 참고 내용]:
{context}

[질문]:
{question}
""")
                
                # LLM 파이프라인 연동
                llm = ChatOpenAI(
                    model="gpt-5.6",
                    temperature=0.2,
                    openai_api_key=openai_api_key
                )
                
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({"context": context_text, "question": user_query})
                
                st.markdown(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})
