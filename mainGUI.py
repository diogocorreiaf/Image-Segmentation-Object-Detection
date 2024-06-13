
import gradio as gr
from Object_Detection import object_detection
from Image_Segmentation import image_segmentation

app = gr.TabbedInterface([object_detection,image_segmentation ], ["Object Detection", "Image Segmentation"])
print(gr.__version__)
app.launch()