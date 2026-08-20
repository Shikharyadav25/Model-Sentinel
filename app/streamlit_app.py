"""
ModelSentinel Streamlit Demo Application.
Interactive UI for AI Model Steganography Malware Detection and X-LSB Attack Simulation.
"""

import base64
import io
import os
import sys
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.attack.xlsb_attack import EICAR_PAYLOAD, embed_payload, embedding_rate
from src.pipeline.scan import scan_model
from src.representation.grayscale_fourpart import to_gf_image
from src.utils.weight_io import load_flat_weights, save_flat_weights_as_state_dict

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ModelSentinel | AI Model Steganography Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: Modern Cybersecurity Dark Theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: #0B0F19;
    }

    .stApp {
        background: linear-gradient(180deg, #0B0F19 0%, #111827 100%);
        color: #F3F4F6;
    }

    /* Custom Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(124, 58, 237, 0.15) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }

    /* Cards */
    .card {
        background: rgba(17, 24, 39, 0.85);
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Verdict Badges */
    .verdict-clean {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10B981;
        color: #10B981;
        padding: 16px 24px;
        border-radius: 12px;
        text-align: center;
    }
    .verdict-low-risk {
        background: rgba(251, 191, 36, 0.15);
        border: 2px solid #FBBF24;
        color: #FBBF24;
        padding: 16px 24px;
        border-radius: 12px;
        text-align: center;
    }
    .verdict-suspicious {
        background: rgba(249, 115, 22, 0.15);
        border: 2px solid #F97316;
        color: #F97316;
        padding: 16px 24px;
        border-radius: 12px;
        text-align: center;
    }
    .verdict-malicious {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #EF4444;
        color: #EF4444;
        padding: 16px 24px;
        border-radius: 12px;
        text-align: center;
    }

    /* Score display */
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 32px;
        font-weight: 700;
    }

    .badge-tag {
        font-family: 'JetBrains Mono', monospace;
        background: #1E293B;
        color: #38BDF8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 13px;
        border: 1px solid #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def annotate_gf_image(gf_arr: np.ndarray, title: str = "Grayscale-Fourpart Grid") -> Image.Image:
    """Adds quadrant overlays and annotations to the GF image."""
    h, w = gf_arr.shape
    pil_img = Image.fromarray(gf_arr).convert("RGB")
    draw = ImageDraw.Draw(pil_img)

    half_h, half_w = h // 2, w // 2

    # Draw divider lines
    draw.line([(half_w, 0), (half_w, h)], fill=(59, 130, 246), width=max(1, w // 256))
    draw.line([(0, half_h), (w, half_h)], fill=(59, 130, 246), width=max(1, h // 256))

    # Highlight B3 (bottom-right quadrant) border with red/orange accent
    draw.rectangle([half_w, half_h, w - 1, h - 1], outline=(239, 68, 68), width=max(2, w // 150))

    return pil_img


def render_verdict_card(result: dict):
    verdict = result["verdict"]
    tier = verdict["tier"]
    risk = verdict["risk_score"]
    color = verdict["tier_color"]
    summary = verdict["summary"]

    tier_class_map = {
        "Clean": "verdict-clean",
        "Low-Risk": "verdict-low-risk",
        "Suspicious": "verdict-suspicious",
        "Malicious": "verdict-malicious",
    }
    css_class = tier_class_map.get(tier, "verdict-clean")

    st.markdown(
        f"""
        <div class="{css_class}">
            <div style="font-size: 13px; letter-spacing: 2px; text-transform: uppercase; font-weight: 600;">VERDICT CLASSIFICATION</div>
            <div style="font-size: 40px; font-weight: 800; margin: 4px 0;">{tier.upper()}</div>
            <div style="font-size: 15px; opacity: 0.95;">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="card">
                <div style="color: #9CA3AF; font-size: 13px; font-weight: 500;">OVERALL RISK SCORE</div>
                <div class="metric-value" style="color: {color};">{risk:.1f} <span style="font-size: 16px; color: #6B7280;">/ 100</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        est_x = verdict.get("estimated_severity_x")
        est_er = verdict.get("estimated_embedding_rate_percent")
        sev_display = f"X = {est_x}" if est_x else "None (Clean)"
        er_display = f"{est_er:.1f}%" if est_er else "0.0%"
        st.markdown(
            f"""
            <div class="card">
                <div style="color: #9CA3AF; font-size: 13px; font-weight: 500;">ESTIMATED SEVERITY</div>
                <div class="metric-value" style="color: #38BDF8;">{sev_display} <span style="font-size: 14px; color: #9CA3AF;">({er_display})</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="card">
                <div style="color: #9CA3AF; font-size: 13px; font-weight: 500;">MODEL PARAMETERS</div>
                <div class="metric-value" style="color: #A78BFA;">{result['num_parameters']:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="card">
                <div style="color: #9CA3AF; font-size: 13px; font-weight: 500;">TOTAL SCAN TIME</div>
                <div class="metric-value" style="color: #34D399;">{result['timing']['total_scan_sec']:.3f}s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_detector_breakdown(result: dict):
    verdict = result["verdict"]
    breakdown = verdict["breakdown"]

    st.markdown("### 🔍 Multi-Signal Detector Breakdown")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        for key, det in breakdown.items():
            name = det["name"]
            score_pct = det["percentage"]
            fired = det["fired"]
            badge_color = "#EF4444" if fired else "#10B981"
            badge_label = "ALERT" if fired else "PASS"

            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; margin-top: 12px;">
                    <span style="font-weight: 600; font-size: 14px;">{name}</span>
                    <span>
                        <span style="background: {badge_color}22; color: {badge_color}; border: 1px solid {badge_color}; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 700; font-family: monospace;">{badge_label}</span>
                        <span style="font-family: monospace; font-weight: 700; margin-left: 8px;">{score_pct:.1f}%</span>
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            bar_color = "red" if fired else "green"
            st.progress(score_pct / 100.0)

    with col_right:
        st.markdown("#### 🖼️ Grayscale-Fourpart (GF) Representation")
        if "gf_image_base64" in result:
            b64_data = result["gf_image_base64"].split(",")[1]
            img_bytes = base64.b64decode(b64_data)
            pil_img = Image.open(io.BytesIO(img_bytes))
            st.image(
                pil_img,
                caption="Top-Left: B0 (Sign/Exp) | Top-Right: B1 | Bottom-Left: B2 | Bottom-Right (Red): B3 (Attack Target)",
                use_container_width=True,
            )


# -----------------------------------------------------------------------------
# Main Navigation & App Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="margin: 0; font-size: 28px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px;">
                    🛡️ ModelSentinel <span style="font-size: 16px; font-weight: 500; color: #60A5FA; background: rgba(59,130,246,0.2); padding: 4px 12px; border-radius: 999px; margin-left: 12px; border: 1px solid rgba(59,130,246,0.4);">v1.0 Hackathon Build</span>
                </h1>
                <p style="margin: 6px 0 0 0; color: #9CA3AF; font-size: 15px;">
                    Automated LSB Steganography Malware Detector & Attack Simulator for AI Model Weights
                </p>
            </div>
            <div style="text-align: right;">
                <span class="badge-tag">🔒 Float32 Mantissa Inspection</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_scan, tab_attack, tab_info = st.tabs([
    "🔍 Scan a Model",
    "⚡ Attack Simulator & Live Demo",
    "📘 How It Works & Benchmarks",
])

# -----------------------------------------------------------------------------
# TAB 1: SCAN A MODEL
# -----------------------------------------------------------------------------
with tab_scan:
    st.markdown("### 📤 Upload or Select a Model to Scan")

    # Sample preloaded models
    benign_dir = "data/benign"
    attacked_dir = "data/attacked"
    sample_options = ["-- Upload Custom File --"]

    if os.path.exists(benign_dir):
        for f in sorted(os.listdir(benign_dir)):
            if f.endswith((".pt", ".safetensors")):
                sample_options.append(f"🟢 [Clean Benign] {f}")
    if os.path.exists(attacked_dir):
        for f in sorted(os.listdir(attacked_dir)):
            if f.endswith((".pt", ".safetensors")):
                sample_options.append(f"🔴 [Attacked Sample] {f}")

    selected_sample = st.selectbox("Quick-Select Preloaded Model:", sample_options)

    target_path = None
    temp_target = None

    if selected_sample != "-- Upload Custom File --":
        clean_name = selected_sample.split("] ")[-1]
        if "Clean Benign" in selected_sample:
            target_path = os.path.join(benign_dir, clean_name)
        else:
            target_path = os.path.join(attacked_dir, clean_name)
    else:
        uploaded_file = st.file_uploader(
            "Choose a PyTorch (.pt / .pth) or Safetensors (.safetensors) model file:",
            type=["pt", "pth", "safetensors", "bin"],
        )
        if uploaded_file is not None:
            ext = os.path.splitext(uploaded_file.name)[1]
            temp_target = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            temp_target.write(uploaded_file.getbuffer())
            temp_target.close()
            target_path = temp_target.name

    col_btn, col_mode = st.columns([1, 3])
    with col_btn:
        start_scan = st.button("🛡️ Run ModelSentinel Scan", type="primary", use_container_width=True)
    with col_mode:
        cnn_mode = st.radio("CNN Evaluation Mode:", ["centroid", "1-nn"], horizontal=True)

    if start_scan and target_path:
        with st.spinner("Analyzing byte planes, calculating statistical entropy, and running Few-Shot CNN inference..."):
            try:
                scan_res = scan_model(target_path, cnn_mode=cnn_mode)
                if selected_sample != "-- Upload Custom File --":
                    scan_res["filename"] = selected_sample
                elif uploaded_file is not None:
                    scan_res["filename"] = uploaded_file.name

                st.session_state["last_scan_result"] = scan_res
            except Exception as e:
                st.error(f"Scan failed: {e}")
            finally:
                if temp_target and os.path.exists(temp_target.name):
                    try:
                        os.remove(temp_target.name)
                    except Exception:
                        pass

    if "last_scan_result" in st.session_state:
        st.markdown("<hr style='border-color: #1F2937; margin: 24px 0;'>", unsafe_allow_html=True)
        render_verdict_card(st.session_state["last_scan_result"])
        st.markdown("<br>", unsafe_allow_html=True)
        render_detector_breakdown(st.session_state["last_scan_result"])


# -----------------------------------------------------------------------------
# TAB 2: ATTACK SIMULATOR & LIVE DEMO
# -----------------------------------------------------------------------------
with tab_attack:
    st.markdown("### ⚡ X-LSB-Attack-Fill Steganography Simulator")
    st.info(
        "💡 **Safety Guarantee**: In accordance with the security policy, payloads are strictly limited "
        "to the harmless **EICAR standard antivirus test file string** or synthetic pseudo-random bytes. No malicious code is ever executed or fetched."
    )

    col_model, col_x = st.columns([1, 1])

    with col_model:
        benign_models = []
        if os.path.exists(benign_dir):
            benign_models = sorted([f for f in os.listdir(benign_dir) if f.endswith((".pt", ".safetensors"))])
        if not benign_models:
            benign_models = ["squeezenet1_0.pt"]

        sim_model_choice = st.selectbox("Select Clean Target Model to Attack:", benign_models)
        sim_payload_type = st.radio("Payload Type:", ["Standard EICAR Antivirus Test Marker", "Pseudo-Random Synthetic Bytes"], horizontal=True)

    with col_x:
        slider_x = st.slider("Attack Severity X (Number of Mantissa LSBs to Overwrite):", min_value=1, max_value=23, value=4, step=1)
        er_pct = embedding_rate(slider_x) * 100.0

        st.markdown(
            f"""
            <div class="card" style="border-left: 4px solid #38BDF8;">
                <div style="font-size: 13px; color: #9CA3AF;">CURRENT EMBEDDING RATE (ER = X / 32)</div>
                <div style="font-size: 26px; font-weight: 700; color: #38BDF8; font-family: monospace;">
                    {er_pct:.2f}% <span style="font-size: 14px; color: #9CA3AF;">({slider_x} out of 23 mantissa bits)</span>
                </div>
                <div style="font-size: 12px; color: #6B7280; margin-top: 4px;">
                    At X={slider_x}, float32 numerical values change by ~2<sup>{slider_x-23}</sup> (negligible to model predictions).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    btn_simulate = st.button("🔥 Inject Payload & Generate Attacked Model", type="primary")

    if btn_simulate:
        benign_sim_path = os.path.join(benign_dir, sim_model_choice)
        if not os.path.exists(benign_sim_path):
            st.error(f"Base model {benign_sim_path} not found. Please run 'python cli.py train' to download the model zoo.")
        else:
            with st.spinner("Embedding synthetic payload and generating Grayscale-Fourpart visualizations..."):
                flat_benign = load_flat_weights(benign_sim_path)
                payload = EICAR_PAYLOAD if "EICAR" in sim_payload_type else np.random.bytes(1024)

                attacked_flat = embed_payload(flat_benign, X=slider_x, payload=payload)

                # Save attacked model to temp file
                temp_att = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
                temp_att.close()
                save_flat_weights_as_state_dict(benign_sim_path, attacked_flat, temp_att.name)

                st.session_state["attacked_sim_path"] = temp_att.name
                st.session_state["attacked_sim_x"] = slider_x
                st.session_state["attacked_sim_flat"] = attacked_flat
                st.session_state["benign_sim_flat"] = flat_benign
                st.session_state["attacked_sim_name"] = f"attacked_x{slider_x}_{sim_model_choice}"

    if "attacked_sim_flat" in st.session_state:
        st.markdown("<hr style='border-color: #1F2937; margin: 24px 0;'>", unsafe_allow_html=True)
        st.markdown("### 🔬 Side-by-Side Grayscale-Fourpart (GF) Comparison")
        st.caption("Notice the high-frequency steganographic noise concentrated in the **bottom-right quadrant (B3)** of the attacked model.")

        gf_benign = to_gf_image(st.session_state["benign_sim_flat"])
        gf_attacked = to_gf_image(st.session_state["attacked_sim_flat"])

        col_img_b, col_img_a = st.columns(2)
        with col_img_b:
            st.markdown("#### 🟢 Clean Benign Model (Pristine Weights)")
            ann_b = annotate_gf_image(gf_benign)
            st.image(ann_b, caption="Benign: Smooth natural distributions across B0, B1, B2, B3", use_container_width=True)

        with col_img_a:
            st.markdown(f"#### 🔴 Attacked Model (X={st.session_state['attacked_sim_x']} Mantissa LSBs Overwritten)")
            ann_a = annotate_gf_image(gf_attacked)
            st.image(ann_a, caption=f"Attacked: High-frequency stego noise visible in bottom-right B3 quadrant", use_container_width=True)

        col_action1, col_action2 = st.columns(2)
        with col_action1:
            with open(st.session_state["attacked_sim_path"], "rb") as f:
                model_bytes = f.read()
            st.download_button(
                label=f"💾 Download {st.session_state['attacked_sim_name']}",
                data=model_bytes,
                file_name=st.session_state["attacked_sim_name"],
                mime="application/octet-stream",
                use_container_width=True,
            )
        with col_action2:
            if st.button("⚡ Scan This Attacked Model Now", type="primary", use_container_width=True):
                with st.spinner("Running scan on generated attacked model..."):
                    res = scan_model(st.session_state["attacked_sim_path"])
                    res["filename"] = st.session_state["attacked_sim_name"]
                    st.session_state["last_scan_result"] = res
                    st.success("Scan complete! View results below.")

        if "last_scan_result" in st.session_state:
            st.markdown("<br>", unsafe_allow_html=True)
            render_verdict_card(st.session_state["last_scan_result"])
            st.markdown("<br>", unsafe_allow_html=True)
            render_detector_breakdown(st.session_state["last_scan_result"])


# -----------------------------------------------------------------------------
# TAB 3: HOW IT WORKS & BENCHMARKS
# -----------------------------------------------------------------------------
with tab_info:
    st.markdown("### 📘 How ModelSentinel Works")

    st.markdown(
        """
        #### 1. The Threat: LSB Steganography in AI Weights
        Modern neural networks store parameters as **IEEE-754 32-bit floating point numbers** (1 sign bit, 8 exponent bits, 23 mantissa bits).
        The lowest bits of the mantissa contribute negligibly to the actual numerical value of the weight (e.g. at $X=4$, $\Delta w \approx 2^{-19} \approx 1.9 \times 10^{-6}$).
        
        Attackers can systematically overwrite these low bits with arbitrary bytes (malware executables, scripts, command-and-control payloads) without affecting the model's prediction accuracy or failing standard inference integrity checks.

        #### 2. Grayscale-Fourpart (GF) Representation
        ModelSentinel extracts the full model parameter set in deterministic alphabetical order and deconstructs every float32 into its 4 constituent byte planes:
        - **B0**: Sign + Top 7 Exponent bits
        - **B1**: 1 Exponent bit + Top 7 Mantissa bits
        - **B2**: Middle 8 Mantissa bits
        - **B3**: Lowest 8 Mantissa bits *(The attack target zone)*
        
        Each byte-plane is formatted into a square matrix of side $\lceil\sqrt{n}\rceil$ and tiled into a $2 \\times 2$ grid:
        ```
        +-------------------+-------------------+
        |  B0 (Sign/Exp)    |  B1 (Exp/Mant)    |
        +-------------------+-------------------+
        |  B2 (Mid Mantissa)|  B3 (LSB Attack)  |
        +-------------------+-------------------+
        ```

        #### 3. Dual Detection Engine
        ModelSentinel pairs two complementary detection systems:
        1. **Statistical Baseline Ensemble**: Computes 4 fast, training-free tests (Byte Entropy, Lag-1 Autocorrelation, B3 Histogram KL-Divergence, and Float32 Value Distribution) calibrated against clean models.
        2. **Few-Shot Siamese CNN (OSL-CNN)**: A 4-stage convolutional embedding network trained with Triplet Loss on minimal examples that learns spatial frequency anomalies in the GF representation and generalizes to unseen model architectures.
        """
    )

    plot_path = "results/accuracy_vs_er.png"
    if os.path.exists(plot_path):
        st.markdown("### 📊 Benchmark Evaluation Results")
        st.image(plot_path, caption="ModelSentinel Accuracy vs. Embedding Rate on Held-out Architectures", use_container_width=True)
    else:
        st.info("Run `python cli.py evaluate` to generate and display the benchmark accuracy curve.")
