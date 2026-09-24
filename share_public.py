import sys
sys.path.insert(0, ".")
import app
app.demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, share=True)
