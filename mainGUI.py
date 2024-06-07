
import gradio as gr
from Object_Detection import object_detection
from Image_Segmentation import image_segmentation

app = gr.TabbedInterface([image_segmentation, object_detection], ["Image Segmentation", "Object Detection"])


if __name__ == "__main__":
    app.launch(title="Image Segmentation and Object Detection")