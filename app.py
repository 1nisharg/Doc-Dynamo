import streamlit as st
import PyPDF2
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq

# Function to initialize conversation chain with GROQ language model
groq_api_key = "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"

# Initializing GROQ chat with provided API key, model name, and settings
llm_groq = ChatGroq(
    groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768", temperature=0.2
)

# Streamlit app configuration
st.set_page_config(page_title="DocDynamo", layout="wide")
st.title("DocDynamo🚀")

uploaded_file = st.file_uploader("Please upload a PDF file to begin!", type="pdf")

st.sidebar.title("DocDynamo By OpenRAG")
st.sidebar.markdown(
    """
    🌟 **Introducing DocDynamo by OpenRAG: Your PDF Companion!** 📚

Welcome, esteemed users, to the groundbreaking release of DocDynamo on May 21, 2024. At OpenRAG, we are committed to pioneering solutions for modern challenges, and DocDynamo is our latest triumph.

    """
)
st.sidebar.markdown(
    """
    💡 **How DocDynamo Works**

Simply upload your PDF, and let DocDynamo work its magic. Once processed, you can ask DocDynamo any question pertaining to the content of your PDF. It's like having a personal assistant at your fingertips, ready to provide instant answers.
    """
)
st.sidebar.markdown(
    """
    📧 **Get in Touch**

For inquiries or collaboration proposals, please don't hesitate to reach out to us:
📩 Email: openrag189@gmail.com
🔗 LinkedIn: [OpenRAG](https://www.linkedin.com/company/102036854/admin/dashboard/)
📸 Instagram: [OpenRAG](https://www.instagram.com/open.rag?igsh=MnFwMHd5cjU1OGFj)

Experience the future of PDF interaction with DocDynamo. Welcome to a new era of efficiency and productivity. OpenRAG: Empowering You Through Innovation. 🚀
    """
)

if uploaded_file:
    with st.spinner(f"Processing `{uploaded_file.name}`..."):
        # Read the PDF file
        pdf = PyPDF2.PdfReader(uploaded_file)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        texts = text_splitter.split_text(pdf_text)

        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
        docsearch = Chroma.from_texts(texts, embeddings)

    
        message_history = ChatMessageHistory()


        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )

    
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm_groq,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            memory=memory,
            return_source_documents=True,
        )

    st.success(f"Processing `{uploaded_file.name}` done. You can now ask questions!")

    user_input = st.text_input("Ask a question about the PDF:")

    if user_input:
        res = chain.invoke(user_input)
        answer = res["answer"]
        source_documents = res["source_documents"]

        text_elements = [] 

    
        if source_documents:
            for source_doc in source_documents:
                text_elements.append(source_doc.page_content)

        # Display the results
        st.markdown(f"**Answer:** {answer}")

        for idx, element in enumerate(text_elements):
            with st.expander(f"Source {idx}"):
                st.write(element)
