import streamlit as st
import joblib

# Must be first Streamlit command
st.set_page_config(page_title="News Category Classifier", page_icon="📰")

# Load saved model, vectorizer, and label encoder
@st.cache_resource
def load_model():
    model = joblib.load("news_category_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    label_encoder = joblib.load("label_encoder.joblib")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_model()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Home", "About", "News Prediction"])

# ---------------------- HOME PAGE ---------------------- #
if page == "Home":
    st.title("📰 News Category Classifier")
    st.write(
        "Welcome! Use this app to classify news headlines or short descriptions into categories."
    )
    st.subheader("Example Predictions")
    examples = [
        "Apple releases new iPhone with improved camera technology.",
        "The president signed a new environmental policy today.",
        "The football team won their first championship in decades!"
    ]
    for ex in examples:
        X_ex = vectorizer.transform([ex])
        pred = model.predict(X_ex)
        category = label_encoder.inverse_transform(pred)[0]
        st.write(f"**Text:** {ex}")
        st.write(f"**Predicted Category:** {category}")
        st.write("---")

# ---------------------- ABOUT PAGE ---------------------- #
elif page == "About":
    st.title("About This App")
    st.write(
        """
        This News Category Classifier is built using **Python, Streamlit, and a machine learning model**.
        It classifies news headlines into categories such as Technology, Politics, Sports, etc.
        
        **Features:**
        - Easy-to-use interface
        - Sidebar navigation
        - Example predictions
        - Fast and accurate classification
        """
    )


# ---------------------- NEWS PREDICTION PAGE ---------------------- #
elif page == "News Prediction":
    st.title("📰 News Prediction")
    st.write("Enter a news headline or short description to classify it.")
    
    user_input = st.text_area("Enter your news text here:", height=150)

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            X_input = vectorizer.transform([user_input])
            prediction = model.predict(X_input)
            category = label_encoder.inverse_transform(prediction)[0]
            st.write("### Your Input:")
            st.write(user_input)
            st.write("### Predicted Category:")
            st.success(category)