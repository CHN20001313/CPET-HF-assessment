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
- Click **Predict** to see your HF probability and recommendations.
""")

# 创建两列布局，左侧输入，右侧显示预测结果
col1, col2 = st.columns([1, 2])  # 左侧 1/3, 右侧 2/3

# **左侧：输入参数**
with st.sidebar:
    st.header("Input Features")
    VO2KGPEAK = st.sidebar.number_input("Oxygen consumption peak (VO2 peak, ml/kg/min)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
    BNP = st.sidebar.number_input("NT-pro BNP (pg/mL)", min_value=0.0, max_value=100000.0, value=1.0, step=0.1)
    EF = st.sidebar.number_input("Ejection fraction (EF, %)", min_value=50.0, max_value=100.0, value=55.0, step=0.1)
    VEVCO2SLOPE = st.sidebar.number_input("Minute ventilation/carbon dioxide production slope (VE/VCO2 slope)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    LDH = st.sidebar.number_input("Lactate dehydrogenase (LDH, U/L)", min_value=0.0, max_value=10000.0, value=120.0, step=0.1)
    CKMB = st.sidebar.number_input("CK-MB (U/L)", min_value=0.0, max_value=100000.0, value=1.0, step=0.1)
    LVEDD = st.sidebar.number_input("Left Ventricular End-Diastolic Diameter (LVEDD, mm)", min_value=10.0, max_value=100.0, value=45.0, step=0.1)
    TNI = st.sidebar.number_input("Troponin I (TNI, μg/L)", min_value=0.0, max_value=100000.0, value=10.0, step=0.1)
    VTpeak = st.sidebar.number_input("Peak tidal volume (VT peak, L/min)", min_value=0.0, max_value=10.0, value=2.0, step=0.01)
    Wpeak = st.sidebar.number_input("Power peak (W)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)

    predict_button = st.sidebar.button("Predict")

# **右侧：显示预测结果**
if predict_button:
    with st.spinner("Calculating..."):
        time.sleep(3)  # 模拟计算时间
    st.success("Calculation complete!")

    # 特征编码
    feature_names = ["VO2 peak", "NT-pro BNP", "EF", "VE/VCO2 slope", "LDH", "CKMB", "LVEDD", "TNI", "VT peak", "Power peak"]
    encoded_features = [VO2KGPEAK, BNP, EF, VEVCO2SLOPE, LDH, CKMB, LVEDD, TNI, VTpeak, Wpeak]
    input_features = np.array(encoded_features).reshape(1, -1)
    dmatrix = xgb.DMatrix(input_features)

    # 预测概率
    probabilities = model.predict(dmatrix, iteration_range=(0, 13))
    predicted_probability = probabilities[0]

    # 风险分组逻辑
    if predicted_probability < 0.172039:
        risk_group = "Low HF Probability"
        risk_color = "green"
        advice = "You have a low probability of HF."
    elif 0.172039 <= predicted_probability <= 0.478812:
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
    with col1:
        with st.container():
            st.header("Prediction Results")
            st.markdown(
                f"<h3 style='color:{risk_color};'>Risk Group: {risk_group}</h3>",
                unsafe_allow_html=True
            )
            st.write(advice)

            # **风险评分**
            risk_score = predicted_probability * 10
            st.markdown(f"**Your risk score is: {risk_score:.2f}**")

            # **SHAP 解释**
            st.markdown(
                f"<h3>Predicted probability of HF is {predicted_probability * 100:.2f}%</h3>",
                unsafe_allow_html=True
            )

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(pd.DataFrame(input_features, columns=feature_names), tree_limit=13)

            # **调整 SHAP 力图**
    with col2:
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.force_plot(
                explainer.expected_value,
                shap_values[0],
                pd.DataFrame(input_features, columns=feature_names),
                matplotlib=True
            )
        plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=1200)
        st.image("shap_force_plot.png", caption="Feature Contribution (SHAP Force Plot)")