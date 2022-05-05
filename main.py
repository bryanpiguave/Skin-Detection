import gradio as gr
from src import hand_utils
import numpy as np

def main(image:np.array):
    return 

if __name__=="__main__":
    iface = gr.Interface(title="Skin detector", description="Given an image of a hand, an estimated area of this hand is calculated",
                         interpretation="Selected skin area",theme="dark",
                        fn=hand_utils.extract_skin,inputs=gr.inputs.Image(type="pil"), outputs=gr.outputs.Image(type="pil"))
    iface.launch()
