from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import OilSpillSegmentation


CHECKPOINT = Path(__file__).parent / "weights.ckpt"
DEFAULT_THRESHOLD = 0.55


st.set_page_config(
    page_title="Detección de derrames | SAR",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
      [data-testid="stFileUploader"] section {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Segmentación de derrames")
st.write("Cargá una imagen satelital para ver la detección y su cobertura estimada.")


@st.cache_resource(show_spinner="Cargando modelo de segmentación…")
def load_model(checkpoint_path: str):
    model = OilSpillSegmentation.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    return model


def predict(image: Image.Image, model: OilSpillSegmentation, threshold: float):
    """Return a binary mask and probabilities resized to the uploaded image."""
    image_gray = image.convert("L")
    tensor = transforms.ToTensor()(image_gray).unsqueeze(0)

    with torch.inference_mode():
        probabilities = model(tensor)
        if probabilities.shape[-2:] != tensor.shape[-2:]:
            probabilities = F.interpolate(
                probabilities,
                size=tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        probabilities = probabilities.squeeze().cpu().numpy()

    mask = probabilities >= threshold
    return mask, probabilities


def make_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    color = np.zeros_like(base)
    color[:] = (255, 83, 73)  # coral red
    alpha = 0.48
    base[mask] = (base[mask] * (1 - alpha) + color[mask] * alpha).astype(np.uint8)
    return Image.fromarray(base)


with st.sidebar:
    st.subheader("Parámetros")
    threshold = st.slider(
        "Umbral de segmentación",
        min_value=0.05,
        max_value=0.95,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Un umbral más alto exige mayor probabilidad para marcar un píxel como derrame.",
    )
    st.caption(f"Modelo: `{CHECKPOINT.name}`")
    st.caption("La cobertura se calcula como píxeles detectados / píxeles totales.")

uploaded_file = st.file_uploader(
    "Elegí una imagen satelital",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    help="Se aceptan imágenes JPG, PNG y TIFF.",
)
if not CHECKPOINT.is_file():
    st.error(f"No se encontró el checkpoint del modelo: `{CHECKPOINT}`")
    st.stop()

if uploaded_file is None:
    st.info("Cargá una imagen para comenzar el análisis.")
    st.caption("El porcentaje es la fracción de píxeles clasificados como derrame; no es un área física en km².")
else:
    try:
        image = Image.open(uploaded_file)
        image.load()
        if image.width < 8 or image.height < 8:
            st.error("La imagen debe tener al menos 8 × 8 píxeles para procesarse.")
            st.stop()

        model = load_model(str(CHECKPOINT))
        with st.spinner("Analizando la imagen…"):
            mask, probabilities = predict(image, model, threshold)

        total_pixels = int(mask.size)
        spill_pixels = int(mask.sum())
        coverage = 100 * spill_pixels / total_pixels
        mean_probability = float(probabilities[mask].mean()) if spill_pixels else 0.0

        st.subheader("Resultado")
        metric1, metric2 = st.columns(2)
        metric1.metric("Cobertura detectada", f"{coverage:.2f}%")
        metric2.metric("Umbral utilizado", f"{threshold:.2f}")

        original_col, overlay_col, mask_col = st.columns(3)
        with original_col:
            st.image(image, caption="Original", use_container_width=True)
        with overlay_col:
            st.image(make_overlay(image, mask), caption="Detección", use_container_width=True)
        with mask_col:
            st.image(mask.astype(np.uint8) * 255, caption="Máscara", clamp=True, use_container_width=True)

        if spill_pixels:
            st.caption(f"Probabilidad media en los píxeles marcados: {mean_probability:.1%}.")
        st.caption(
            "La cobertura es una estimación de píxeles, no un área en km². "
            "La calidad depende de la imagen, su preprocesamiento y el umbral elegido."
        )
    except Exception as exc:
        st.error(f"No se pudo procesar la imagen: {exc}")
