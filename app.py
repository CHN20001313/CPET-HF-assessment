import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import time
# 加载 JSON 格式模型
model = xgb.Booster()
model.load_model("xgboost_model.model")

# 设置页面宽度
st.set_page_config(layout="wide")

# 页面标题和简介
st.title("CPET Based Post-AMI Heart Failure Probability Predictor")
st.markdown("""
This tool predicts the likelihood of heart failure (HF) after acute myocardial infarction (AMI) based on patient characteristics.

**Instructions:**
- Fill in your details on the left.
- Click **Predict** to see your HFrEF/HFmrEF probability and recommendations.
""")

# 创建两列布局，左侧输入，右侧显示预测结果
col1, col2 = st.columns([1, 2])  # 左侧 1/3, 右侧 2/3

# **左侧：输入参数**
with col1:
    st.sidebar.header("Input Features")
    BNP = st.sidebar.number_input("NT-pro BNP (pg/mL)", min_value=0.0, max_value=100000.0, value=1.0, step=0.1)
    LVEDD = st.sidebar.number_input("Left Ventricular End-Diastolic Diameter (LVEDD, mm)", min_value=10.0, max_value=100.0, value=45.0, step=0.1)
    VO2KGPEAK = st.sidebar.number_input("Oxygen consumption peak (VO2 peak, ml/kg/min)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    LDH = st.sidebar.number_input("Lactate dehydrogenase (LDH, U/L)", min_value=0.0, max_value=10000.0, value=120.0, step=0.1)
    VEVCO2SLOPE = st.sidebar.number_input("Minute ventilation/carbon dioxide production slope (VE/VCO2 slope)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    CKMB = st.sidebar.number_input("CK-MB (U/L)", min_value=0.0, max_value=100000.0, value=1.0, step=0.1)
    EF = st.sidebar.number_input("Ejection fraction (EF, %)", min_value=50.0, max_value=100.0, value=55.0, step=0.1)
    TNI = st.sidebar.number_input("Troponin I (cTNI, μg/L)", min_value=0.0, max_value=100000.0, value=10.0, step=0.1)
    PETCO2peak = st.sidebar.number_input("Peak partial pressure of end tidal carbon dioxide (PETCO2 peak, mmHg)", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
    VTpeak = st.sidebar.number_input("Peak tidal volume (VT peak, L/min)", min_value=0.0, max_value=10.0, value=2.0, step=0.01)

    predict_button = st.sidebar.button("Predict")

# **右侧：显示预测结果**
if predict_button:
    with st.spinner("Calculating..."):
        time.sleep(3)  # 模拟计算时间
    st.success("Calculation complete!")

    # 特征编码
    feature_names = ["NT-pro BNP", "LVEDD", "VO2 peak", "LDH", "VE/VCO2 slope", "CK-MB", "EF", "cTNI", "PETCO2 peak", "VT peak"]
    encoded_features = [BNP, LVEDD, VO2KGPEAK, LDH, VEVCO2SLOPE, CKMB, EF, TNI, PETCO2peak, VTpeak]
    input_features = np.array(encoded_features).reshape(1, -1)
    dmatrix = xgb.DMatrix(input_features)

    # 预测概率
    probabilities = model.predict(dmatrix, iteration_range=(0, 12))
    predicted_probability = probabilities[0]

    # 风险分组逻辑
    if predicted_probability < 0.248145:
        risk_group = "Low HF Probability"
        risk_color = "green"
        advice = "You have a low probability of HF."
    elif 0.248145 <= predicted_probability <= 0.338074:
        risk_group = "Medium HF Probability"
        risk_color = "orange"
        advice = (
            "Your probability of HF is moderate. It is recommended to monitor your health closely "
            "and consider consulting a healthcare professional for further evaluation."
        )
    else:
        risk_group = "High HF Probability"
        risk_color = "red"
        advice = (
            "You have a high probability of HF. Please consult a healthcare professional as soon as possible "
            "for detailed evaluation and treatment guidance."
        )

    # **显示结果在右侧**
    with col2:
        st.header("Prediction Results")
        st.markdown(
            f"<h3 style='font-size:24px; color:{risk_color};'>Risk Group: {risk_group}</h3>",
            unsafe_allow_html=True
        )
        st.write(advice)

        # 显示风险评分
        risk_score = predicted_probability * 10
        st.markdown(f"**Your risk score is: {risk_score:.2f}**")

        # **SHAP 解释**
        st.header(f"Predicted probability of HFis {predicted_probability * 100:.2f}%")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(pd.DataFrame(input_features, columns=feature_names), tree_limit=12)

        # **调整 SHAP 力图渲染**
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            pd.DataFrame(input_features, columns=feature_names),
            matplotlib=True
        )
        plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=1200)
        st.image("shap_force_plot.png", caption="Feature Contribution (SHAP Force Plot)")