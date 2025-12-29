
import streamlit as st
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(layout="wide")
st.title("Student Performance Predictor")

# ---------------- Parameter controls in sidebar ----------------
st.sidebar.header("Adjustment Parameters (tweak these)")
study_thresh = st.sidebar.slider("Study hours threshold (study_limit)", min_value=1.0, max_value=40.0, value=20.0, step=1.0)
att_thresh = st.sidebar.slider("Attendance threshold (att_limit %)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
max_boost = st.sidebar.slider("Maximum boost (points)", min_value=0.0, max_value=15.0, value=10.0, step=1.0)
study_weight = st.sidebar.slider("Study weight (0-10)", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
att_weight = st.sidebar.slider("Attendance weight (0-5)", min_value=1.0, max_value=5.0, value=1.0, step=1.0)


# ensure weights sum to 15 (normalize if needed) and show info
total_w = study_weight + att_weight
if total_w == 0:
    study_weight = 2.5
    att_weight = 3.75
else:
    study_weight = total_w / study_weight 
    att_weight = total_w / att_weight

st.sidebar.markdown(f"**Normalized weights:** study {study_weight:.2f}, attendance {att_weight:.2f} (sum=15.00)")


# Load Model 
@st.cache_resource
def load_model():
    return joblib.load("student_model.joblib")

model = load_model()


# Input Form 
st.header("Enter Student Details")

study = st.number_input("Study Hours per Week", 0.0, 80.0, 30.0, step=1.0)
attendance = st.number_input("Attendance %", 0.0, 100.0, 75.0, step=1.0)
overall = st.number_input("Previous Overall Score (%)", 0.0, 100.0, 60.0, step=1.0)
gender = st.selectbox("Gender", ["M", "F", "Other"])

st.markdown("**Optional:** Agar aapke paas per-subject previous marks hain to yahan daal sakte hain (nahi to overall use hoga)")
prev_col1, prev_col2, prev_col3 = st.columns(3)
with prev_col1:
    prev_math = st.number_input("Previous Math Marks (0-100)", min_value=0.0, max_value=100.0, value=overall, step=1.0)
with prev_col2:
    prev_science = st.number_input("Previous Science Marks (0-100)", min_value=0.0, max_value=100.0, value=overall, step=1.0)
with prev_col3:
    prev_english = st.number_input("Previous English Marks (0-100)", min_value=0.0, max_value=100.0, value=overall, step=1.0)
if st.button("Predict"):
    if model is None:
        st.error("Please upload the trained model (.joblib) first from the sidebar.")
    else:
        X = pd.DataFrame([{
            "study_hours": float(study),
            "attendance_percentage": float(attendance),
            "overall_score":float(overall),
            "gender": gender
        }])

        try:
            pred = model.predict(X)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            raise

        math_p = float(pred[0])
        sci_p = float(pred[1])
        eng_p = float(pred[2])


        # compute boost with user-controlled parameters
        def compute_boost(prev, study_val, attendance_val, s_thresh, a_thresh, m_boost, s_w, a_w):
            if study_val < s_thresh or attendance_val < a_thresh:
                return 0.0
            study_norm = min(1.0, (study_val - s_thresh) / max(30.0, 80.0 - s_thresh))
            att_norm = min(1.0, (attendance_val - a_thresh) / max(75.0, 100.0 - a_thresh))
            combined = s_w * study_norm + a_w * att_norm
            prev_factor = (100.0 - prev) / 100.0
            raw_boost = m_boost * combined * prev_factor
            return float(max(0.0, min(m_boost, raw_boost)))

        previous_scores = [
            prev_math if prev_math is not None else overall,
            prev_science if prev_science is not None else overall,
            prev_english if prev_english is not None else overall
        ]

        math_boost = compute_boost(previous_scores[0], study, attendance, study_thresh, att_thresh, max_boost, study_weight, att_weight)
        sci_boost = compute_boost(previous_scores[1], study, attendance, study_thresh, att_thresh, max_boost, study_weight, att_weight)
        eng_boost = compute_boost(previous_scores[2], study, attendance, study_thresh, att_thresh, max_boost, study_weight, att_weight)

        math_p_adj = min(100.0, math_p + math_boost)
        sci_p_adj = min(100.0, sci_p + sci_boost)
        eng_p_adj = min(100.0, eng_p + eng_boost)

        if any([math_boost > 0, sci_boost > 0, eng_boost > 0]):
            st.info(f"🔧 Adjusted due to high study & attendance. Boosts — Math:+{math_boost:.1f}, Sci:+{sci_boost:.1f}, Eng:+{eng_boost:.1f}")

        display_math = math_p_adj
        display_sci = sci_p_adj
        display_eng = eng_p_adj

        st.subheader("Predicted Marks (raw vs adjusted)")
        st.write(f"Math: {display_math:.2f}  (raw: {math_p:.2f})")
        st.write(f"Science: {display_sci:.2f} (raw: {sci_p:.2f})")
        st.write(f"English: {display_eng:.2f} (raw: {eng_p:.2f})")

        st.subheader("Pie Charts (adjusted)")
        subjects = {"Math": display_math, "Science": display_sci, "English": display_eng}
        cols = st.columns(3)
        for (name, score), col in zip(subjects.items(), cols):
            with col:
                fig, ax = plt.subplots()
                ax.pie([score, max(0, 100 - score)],
                       labels=[f"{name}: {score:.1f}", "Remaining"],
                       autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)

        st.subheader("Previous vs Predicted (Line Graph)")
        predicted_scores = [display_math, display_sci, display_eng]
        subjects_list = ["[Math]", "[Science]", "[English]"]

        fig_line, ax_line = plt.subplots(figsize=(8, 4))
        ax_line.plot(subjects_list, previous_scores, marker='o', linestyle='-', linewidth=1.5, label='Previous Marks')
        ax_line.plot(subjects_list, predicted_scores, marker='o', linestyle='--', linewidth=2, label='Predicted Marks (adjusted)')
        for i, (pvs, prd) in enumerate(zip(previous_scores, predicted_scores)):
            ax_line.annotate(f"{pvs:.1f}", (i, pvs), textcoords="offset points", xytext=(0,8), ha='center')
            ax_line.annotate(f"{prd:.1f}", (i, prd), textcoords="offset points", xytext=(0,12), ha='center')
        ax_line.set_xlabel("<-----Subjects----->"); ax_line.set_ylabel("<------Marks------>"); ax_line.set_ylim(0,100)
        ax_line.set_title("Previous vs Predicted Marks"); ax_line.legend(); ax_line.grid(True)
        st.pyplot(fig_line)
        

        st.subheader("Performance Conditions")
        cond_messages = []
        if study >= 14:
            if not (display_math > previous_scores[0] or display_sci > previous_scores[1] or display_eng > previous_scores[2]):
                cond_messages.append("⚠ Aapki study hours kaafi hain, phir bhi predicted marks improve nahi hue — study method check karo.")
        if attendance >= 75:
            if not (display_math > previous_scores[0] or display_sci > previous_scores[1] or display_eng > previous_scores[2]):
                cond_messages.append("⚠ Attendance achhi hai, lekin predicted marks increase nahi hue — class participation / doubt clearing check karo.")
        if display_math < previous_scores[0]:
            cond_messages.append("❌ Math predicted score previous se kam hai.")
        if display_sci < previous_scores[1]:
            cond_messages.append("❌ Science predicted score previous se kam hai.")
        if display_eng < previous_scores[2]:
            cond_messages.append("❌ English predicted score previous se kam hai.")
        if not cond_messages:
            st.success("✅ All performance conditions satisfied — aapka performance theek dikh raha hai!")
        else:
            for c in cond_messages:
                st.warning(c)

        st.subheader("Subject-wise Improvement Suggestions")
        suggestions_math = [
            "📚 Basics revise karo: algebra/fundamentals phir se dekho.",
            "📝 Daily 30 mins solved examples practice karo (previous year questions).",
            "🧩 Weak topics identify karo aur targeted practice karo.",
            "⌛ Timed quizzes lo — speed aur accuracy dono improve hote hain.",
            "👨‍🏫 Zarurat ho toh tutor se weekly 1–2 sessions consider karo.",
            "📺 Short topic videos dekho aur notes banao."
        ]
        suggestions_science = [
            "🔬 Theory + Practical dono par focus karo.",
            "🧪 Diagrams/steps likhkar yaad karo.",
            "🧠 Concept maps banao — linking se retention strong hota hai.",
            "🧾 Previous year questions solve karo.",
            "🔁 Active recall use karo."
        ]
        suggestions_english = [
            "📖 Daily reading + 10 new words learn karo.",
            "✍️ Short answers/essays likhne ki practice karo.",
            "🔄 Grammar basics revise karo.",
            "🗣️ Speaking practice karo — fluency improve hoti hai."
        ]
        any_suggestion = False
        if display_math < previous_scores[0]:
            any_suggestion = True
            with st.expander(f"Math Suggestions (Predicted {display_math:.1f} < Previous {previous_scores[0]:.1f})"):
                for s in suggestions_math:
                    st.write("- " + s)
        if display_sci < previous_scores[1]:
            any_suggestion = True
            with st.expander(f"Science Suggestions (Predicted {display_sci:.1f} < Previous {previous_scores[1]:.1f})"):
                for s in suggestions_science:
                    st.write("- " + s)
        if display_eng < previous_scores[2]:
            any_suggestion = True
            with st.expander(f"English Suggestions (Predicted {display_eng:.1f} < Previous {previous_scores[2]:.1f})"):
                for s in suggestions_english:
                    st.write("- " + s)
        if not any_suggestion:
            st.success("🎉 Predicted marks equal/above previous marks — koi extra subject suggestions nahi hain. Bas aap isi consistency ke sath padhte rahe.")

        st.subheader("Future Performance Potential (Motivation)")
        if study >= 14 and attendance >= 75:
            st.info(
                "🌟 Aapki study habits aur attendance strong hai! "
                "Agar aap isi consistency ko maintain rakhenge, to aap future me **bahut achha perform kar sakte ho**. "
                "Regular practice + correct strategy = better scores."
            )
        elif (display_math >= previous_scores[0] or display_sci >= previous_scores[1] or display_eng >= previous_scores[2]):
            st.info("🚀 Aapka performance trend improving hai! Kuch subjects me predicted marks better aa rahe hain — keep it up!")
        else:
            st.info("💡 Predicted marks thode kam aaye hain, lekin aap me potential hai. Thodi focused practice aur better study methods se aap future me improve kar sakte ho! 💪")
