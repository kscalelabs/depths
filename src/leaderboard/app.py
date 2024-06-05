"""A leaderboard for the Humanoid at Home Depth Benchmark."""

import os
from typing import List

import gradio as gr
import pandas as pd
from huggingface_hub import HfApi

api = HfApi()

HF_TOKEN = os.getenv("HF_TOKEN")


def clickable(x: str) -> str:
    return (
        f'<a target="_blank" href="https://huggingface.co/{x}" '
        f'style="color: var(--link-text-color); '
        f'text-decoration: underline;text-decoration-style: dotted;">{x}</a>'
    )


def apply_headers(df: pd.DataFrame, headers: List[str]) -> pd.DataFrame:
    tmp = df.copy()
    tmp.columns = headers
    return tmp


def search(search_text: str) -> pd.DataFrame:
    if not search_text:
        return df

    return df[df["model"].str.contains(search_text, case=False, na=False)]


df = pd.read_csv("model_performance.csv")

df_author_copy = df.copy()

df = apply_headers(
    df, ["🏛️ Model", "⚡️ Performance 1", "⚡️ Performance 2", "⚡️ Performance 3"]
)

desc = """

🎯 The Leaderboard aims to evaluate depth estimation in a home environment
from the humanoid perspective.

## 📄 Information

The dataset consists of n frames of depth images and corresponding ground truth
depth images. We provide instrinsic camera parameters for each frame if VLM can
be prompted with this information. The evaluation part of the dataset is hidden
to avoid contamination for the competition.

The evaluation metric is the mean absolute error (MAE) between the predicted
depth and the ground truth depth. The leaderboard is based on the MAE
of the test set. The lower the MAE, the better the performance.


## 📒 Notes

For more information, please contact us at team at kscale.dev.
"""


title = """
<div style="text-align:center">
  <h1 id="space-title"> Humanoid at Home Depth Benchmark </h1>
</div>
"""

with gr.Blocks() as demo:
    gr.Markdown(
        """<h1 align="center" id="space-title">Humanoid at Home Depth Benchmark </h1>"""  # noqa: E501
    )
    gr.Markdown(
        """
    <div style="display: flex; justify-content: center;">
        <img src="https://i.ibb.co/Ns2yFjN/Screenshot-2024-06-05-at-10-32-21.png"
            alt="Image" style="width:200px; margin-right:10px;" border="2px"/>
        <img src="https://media.kscale.dev/stompy.png"
            alt="Image" style="width:140px; margin-right:10px;" border="2px"/>
        <img src="https://i.ibb.co/WHg7mMM/Screenshot-2024-06-05-at-10-16-58.png"
            alt="Image" style="width:500px; margin-right:10px;" border="2px"/>
        <img src="https://i.ibb.co/MPSSqd3/image.png"
            alt="Image" style="width:200px; margin-right:10px;" border="2px"/>
    </div>
    """
    )
    gr.Markdown(desc)
    with gr.Column(min_width=320):
        search_bar = gr.Textbox(
            placeholder="🔍 Search for the model", show_label=False
        )

    gr_followers = gr.Dataframe(
        df, interactive=False, datatype=["number", "markdown", "number"]
    )

    search_bar.submit(fn=search, inputs=search_bar, outputs=gr_followers)


demo.launch()
