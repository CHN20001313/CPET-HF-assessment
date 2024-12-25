import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import time
# 加载 JSON 格式模型
model = xgb.Booster()
model.load_model("xgboost_model.json")

# 设置页面宽度
st.set_page_config(layout="wide")

# 页面标题和简介
st.title("AMI Follow-up period HF Probability Predictor")
st.markdown("""
This tool predicts the likelihood of HFrEF and HFmrEF during follow-up period after acute myocardial infarction (AMI) based on patient characteristics.

**Instructions:**
- Fill in your details on the left.
- Click **Predict** to see your HF probability and recommendations.
""")

# 创建两列布局
col1, col2 = st.columns(2)

# 左侧输入区域
with col1:
    st.header("Input Features")
    LVEDD = st.number_input("Left Ventricular End-Diastolic Diameter (LVEDD, mm)", min_value=10.0, max_value=100.0, value=45.0, step=0.1)
    BRRPEAK = st.number_input("Respiratory rate peak(RR peak, bpm)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    LDH = st.number_input("Lactate dehydrogenase (LDH, U/L)", min_value=0.0, max_value=10000.0, value=120.0, step=0.1)
    VO2KGPEAK = st.number_input("Oxygen consumption peak (VO2 peak, ml/kg/min)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    SBPpeak = st.number_input("Systolic blood pressure (SBP peak, mmHg)", min_value=0.0, max_value=300.0, value=150.0, step=1.0)
    VEVCO2SLOPE = st.number_input("Minute ventilation/carbon dioxide production slope (VE/VCO2 slope)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    VEKGpeak = st.number_input("Ventilation peak (VE peak, ml/kg)", min_value=100.0, max_value=2000.0, value=400.0, step=0.1)
    EF = st.number_input("Ejection fraction (EF, %)", min_value=50.0, max_value=100.0, value=55.0, step=0.1)
    TNI = st.number_input("Troponin I (cTNI, μg/L)", min_value=0.0, max_value=100000.0, value=10.0, step=0.1)
    CKMB = st.number_input("CK-MB (U/L)", min_value=0.0, max_value=100000.0, value=1.0, step=0.1)

    if st.button("Predict"):
        with st.spinner("Calculating..."):
            time.sleep(3)  # 模拟计算时间
        st.success("Calculation complete!")
        # 特征编码
        feature_names = ["LVEDD", "RR peak", "LDH", "VO2 peak", "SBPp eak", "VE/VCO2 slope", "VE peak", "EF", "cTNI", "CKMB"]
        encoded_features = [
            LVEDD, BRRPEAK, LDH, VO2KGPEAK, SBPpeak, VEVCO2SLOPE, VEKGpeak, EF, TNI ,CKMB]
        input_features = np.array(encoded_features).reshape(1, -1)
        dmatrix = xgb.DMatrix(input_features)

        # 预测概率
        probabilities = model.predict(dmatrix)
        predicted_probability = probabilities[0]

        # 风险分组逻辑
        if predicted_probability < 0.425285:
            risk_group = "Low Follow-up period HF Probability"
            risk_color = "green"
            advice = (
                "You have a low probability of follow-up period heart failure. Please consult a healthcare professional as soon as possible "
                "for detailed evaluation and treatment guidance."
            )
        elif 0.425285 <= predicted_probability <= 0.544843:
            risk_group = "Medium Follow-up period HF Probability"
            risk_color = "orange"
            advice = (
                "Your probability of follow-up period heart failure is moderate. It is recommended to monitor your health closely "
                "and consider consulting a healthcare professional for further evaluation."
            )
        else:
            risk_group = "High Follow-up period HF Probability"
            risk_color = "red"
            advice = "You have a high probability of follow-up period heart failure."



        # 显示结果在右侧
        with col2:
            st.header("Prediction Results")
            #st.markdown(
             #   f"<h3 style='font-size:24px;'>Prediction Probability: {predicted_probability * 100:.2f}%</h3>",
              #  unsafe_allow_html=True
            #)
            st.markdown(
                f"<h3 style='font-size:24px; color:{risk_color};'>Risk Group: {risk_group}</h3>",
                unsafe_allow_html=True
            )
            st.write(advice)

            # SHAP 力图
            st.header(
                f"Based on feature values, predicted probability of follow-up period HF is {predicted_probability * 100:.2f}%")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pd.DataFrame(input_features, columns=feature_names))
            shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                pd.DataFrame(input_features, columns=feature_names),
                matplotlib=True
            )
            plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=1200)
            st.image("shap_force_plot.png", caption="Feature Contribution (SHAP Force Plot)")